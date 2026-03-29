import pickle
import pandas as pd

model = pickle.load(open("artifacts/model.pkl", "rb"))
encoders = pickle.load(open("artifacts/encoders.pkl", "rb"))

def predict(data):
    df = pd.DataFrame([data])

    for col in encoders:
        if df[col][0] not in encoders[col].classes_:
            df[col] = encoders[col].classes_[0]
        else:
            df[col] = encoders[col].transform(df[col])

    return model.predict(df)[0]
