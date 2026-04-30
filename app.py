import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import shap

# ---------- LOAD MODEL ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model not found. Run: python src/train.py")
    st.stop()

model = joblib.load(model_path)

# Extract pipeline parts
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

# ---------- INR FORMAT ----------
def format_inr(num):
    if num >= 10000000:
        return f"₹ {num/10000000:.2f} Cr"
    elif num >= 100000:
        return f"₹ {num/100000:.2f} Lakh"
    else:
        return f"₹ {int(num):,}"

# ---------- UI ----------
st.set_page_config(page_title="Advanced House Price Predictor", layout="wide")
st.title("🏠 AI House Price Prediction System (India)")

# Sidebar inputs
st.sidebar.header("Enter Property Details")

area = st.sidebar.slider("Area (sq ft)", 500, 5000, 1200)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
floors = st.sidebar.slider("Floors", 1, 3, 1)
age = st.sidebar.slider("Property Age", 0, 30, 5)
parking = st.sidebar.selectbox("Parking", [0, 1])
furnishing = st.sidebar.selectbox("Furnishing", ["furnished", "semi", "unfurnished"])

# Input dataframe
input_data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floors": floors,
    "age": age,
    "parking": parking,
    "furnishing": furnishing
}])

# ---------- PREDICTION ----------
if st.button("Predict Price"):

    price = model.predict(input_data)[0]

    col1, col2 = st.columns(2)
    col1.metric("Estimated Price", format_inr(price))
    col2.metric("Price per sq ft", f"₹ {int(price/area):,}")

    st.markdown("---")

    # ---------- SHAP EXPLAINABILITY ----------
    st.subheader("🔍 Why this price? (AI Explanation)")

    try:
        # Transform input
        X_transformed = preprocessor.transform(input_data)

        # Feature names
        feature_names = preprocessor.get_feature_names_out()

        # SHAP explainer
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)

        # ---------- SUMMARY PLOT ----------
        st.write("### 📊 Feature Importance")
        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig1)

        # ---------- WATERFALL ----------
        st.write("### 📌 Price Increase / Decrease Breakdown")

        shap_val = shap_values[0]

        impact_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_val
        }).sort_values(by="Impact", key=abs, ascending=False)

        # Show table with ₹ impact
        impact_df["Impact ₹"] = impact_df["Impact"].apply(lambda x: f"{'+' if x>0 else ''}{format_inr(x)}")

        st.dataframe(impact_df)

        # ---------- VISUAL BAR ----------
        st.write("### 📈 Feature Contribution Graph")

        fig2, ax2 = plt.subplots()
        colors = ["green" if v > 0 else "red" for v in impact_df["Impact"]]

        ax2.barh(impact_df["Feature"], impact_df["Impact"], color=colors)
        ax2.set_title("Price Increase (Green) / Decrease (Red)")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"SHAP Error: {e}")

    st.markdown("---")

    # ---------- WHAT-IF SIMULATION ----------
    st.subheader("🧪 What-if Simulation")

    new_area = st.slider("Increase Area", 500, 6000, area)

    temp = input_data.copy()
    temp["area"] = new_area

    new_price = model.predict(temp)[0]

    st.write(f"New Price if area changes: {format_inr(new_price)}")

    diff = new_price - price

    if diff > 0:
        st.success(f"📈 Price increases by {format_inr(diff)}")
    else:
        st.error(f"📉 Price decreases by {format_inr(abs(diff))}")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("### 📌 Insights")
st.write("""
- Area has strong positive impact 📈  
- More bedrooms increase value 🏡  
- Older homes reduce price 📉  
- Furnishing adds premium 💰  
""")