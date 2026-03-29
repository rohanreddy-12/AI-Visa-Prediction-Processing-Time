import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/processed_data.csv")

encoders = {}
for col in ["country", "visa_type", "processing_office"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("processing_time", axis=1)
y = df["processing_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

# Save artifacts
pickle.dump(model, open("artifacts/model.pkl", "wb"))
pickle.dump(encoders, open("artifacts/encoders.pkl", "wb"))
