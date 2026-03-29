from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.data import generate_synthetic_visa_data, add_features, clean_data, get_feature_frame
from src.model import train_and_select_model, save_bundle
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ART_DIR = BASE_DIR / "artifacts"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    ART_DIR.mkdir(exist_ok=True)

    df = generate_synthetic_visa_data(n_rows=6000, seed=42)
    df = add_features(df)
    df = clean_data(df)

    # Save dataset for inspection
    df.to_csv(DATA_DIR / "visa_processing_data.csv", index=False)

    X, y, _ = get_feature_frame(df)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    bundle, best_name = train_and_select_model(X_train, y_train, X_valid, y_valid, random_state=42)
    save_bundle(bundle, ART_DIR)

    report = {
        "best_model": best_name,
        "metrics": bundle.metrics,
    }
    (ART_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()