import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "area": np.random.randint(500, 5000, 500),
    "bedrooms": np.random.randint(1, 6, 500),
    "bathrooms": np.random.randint(1, 4, 500),
    "floors": np.random.randint(1, 3, 500),
    "age": np.random.randint(0, 30, 500),
    "parking": np.random.randint(0, 2, 500),
    "furnishing": np.random.choice(["furnished", "semi", "unfurnished"], 500),
}

df = pd.DataFrame(data)

df["price"] = (
    df["area"] * 3000 +
    df["bedrooms"] * 50000 +
    df["bathrooms"] * 30000 +
    df["floors"] * 20000 -
    df["age"] * 1000 +
    df["parking"] * 15000 +
    np.random.randint(-50000, 50000, 500)
)

df.to_csv("data/housing.csv", index=False)
print("Dataset Created!")