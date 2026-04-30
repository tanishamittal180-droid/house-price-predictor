import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------- CREATE DATA ----------
np.random.seed(42)

df = pd.DataFrame({
    "area": np.random.randint(500, 5000, 300),
    "bedrooms": np.random.randint(1, 6, 300),
    "bathrooms": np.random.randint(1, 4, 300),
    "floors": np.random.randint(1, 3, 300),
    "age": np.random.randint(0, 30, 300),
    "parking": np.random.randint(0, 2, 300),
    "furnishing": np.random.choice(["furnished", "semi", "unfurnished"], 300)
})

df["price"] = (
    df["area"] * 3000 +
    df["bedrooms"] * 50000 +
    df["bathrooms"] * 30000 +
    df["floors"] * 20000 -
    df["age"] * 1000 +
    df["parking"] * 15000 +
    np.random.randint(-50000, 50000, 300)
)

df.to_csv("data/housing.csv", index=False)

# ---------- FEATURES ----------
X = df.drop("price", axis=1)
y = df["price"]

cat_cols = ["furnishing"]
num_cols = ["area", "bedrooms", "bathrooms", "floors", "age", "parking"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100))
])

# ---------- TRAIN ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

# ---------- SAVE ----------
joblib.dump(model, "models/model.pkl")

print("✅ Model trained and saved successfully!")