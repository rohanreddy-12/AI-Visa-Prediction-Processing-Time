import pickle
import pandas as pd

model = pickle.load(open("artifacts/model.pkl", "rb"))
encoders = pickle.load(open("artifacts/encoders.pkl", "rb"))

def predict(data):
    df = pd.DataFrame([data])

    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    return model.predict(df)[0]