import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "country": np.random.choice(["India", "USA", "UK", "Canada"], n),
    "visa_type": np.random.choice(["Student", "Work", "Tourist"], n),
    "processing_office": np.random.choice(["Delhi", "Mumbai", "Chennai"], n),
    "month": np.random.randint(1, 13, n),
    "processing_time": np.random.randint(15, 90, n)
})

# Save raw
data.to_csv("data/raw_data.csv", index=False)

# Simple preprocessing
data.fillna(method="ffill", inplace=True)

# Save processed
data.to_csv("data/processed_data.csv", index=False)

