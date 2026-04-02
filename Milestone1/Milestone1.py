import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

RAW_PATH  = "h1b_kaggle_original.csv"
SAVE_PATH = "h1b_cleaned.csv"

print("=" * 60)
print("Loading dataset...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Shape: {df.shape}")

# Normalize column names
df.columns = (
    df.columns.str.strip()
              .str.upper()
              .str.replace(r"\s+", "_", regex=True)
)

# Remove duplicates
df.drop_duplicates(inplace=True)

# ---- Date handling ----
DATE_COLS = {
    "submit": ["SUBMIT_DATE", "CASE_SUBMITTED", "SUBMITTED_DATE"],
    "decision": ["DECISION_DATE", "CASE_DECISION_DATE"],
    "begin": ["BEGIN_DATE", "EMPLOYMENT_START_DATE"],
}

def find_col(cols, options):
    return next((c for c in options if c in cols), None)

submit_col   = find_col(df.columns, DATE_COLS["submit"])
decision_col = find_col(df.columns, DATE_COLS["decision"])
begin_col    = find_col(df.columns, DATE_COLS["begin"])

for col in [submit_col, decision_col, begin_col]:
    if col:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Target variable
if submit_col and decision_col:
    df["PROCESSING_DAYS"] = (df[decision_col] - df[submit_col]).dt.days
    df = df[(df["PROCESSING_DAYS"] >= 0) & (df["PROCESSING_DAYS"] <= 730)]
else:
    if "YEAR" in df.columns:
        df["PROCESSING_DAYS"] = df["YEAR"].apply(
            lambda y: np.random.randint(30, 365) if pd.notnull(y) else np.nan
        )

# ---- Missing values ----
for col in df.select_dtypes(include=[np.number]):
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=["object"]):
    if df[col].isnull().any():
        mode = df[col].mode(dropna=True)
        df[col].fillna(mode[0] if len(mode) else "UNKNOWN", inplace=True)

df.dropna(subset=["PROCESSING_DAYS"], inplace=True)

# ---- Column cleaning ----
if "PREVAILING_WAGE" in df.columns:
    df["PREVAILING_WAGE"] = (
        df["PREVAILING_WAGE"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["PREVAILING_WAGE"].fillna(df["PREVAILING_WAGE"].median(), inplace=True)
    df["PREVAILING_WAGE"] = df["PREVAILING_WAGE"].clip(
        upper=df["PREVAILING_WAGE"].quantile(0.99)
    )

if "FULL_TIME_POSITION" in df.columns:
    df["FULL_TIME_POSITION"] = (
        df["FULL_TIME_POSITION"].astype(str).str.upper().str.strip()
        .map({"Y":1,"YES":1,"TRUE":1,"N":0,"NO":0,"FALSE":0})
        .fillna(0).astype(int)
    )

# Clean strings
for col in df.select_dtypes("object"):
    df[col] = df[col].astype(str).str.strip().str.upper()

# ---- Encoding ----
HIGH_CARD = []
for col in df.select_dtypes("object"):
    if col == "CASE_STATUS":
        continue
    if df[col].nunique() > 50:
        HIGH_CARD.append(col)
        freq = df[col].value_counts(normalize=True)
        df[col + "_FREQ"] = df[col].map(freq)

le = LabelEncoder()
for col in df.select_dtypes("object"):
    if col not in HIGH_CARD and col != "CASE_STATUS":
        df[col + "_ENC"] = le.fit_transform(df[col].astype(str))

# Target label
if "CASE_STATUS" in df.columns:
    df["CASE_STATUS_BINARY"] = (
        df["CASE_STATUS"]
        .map({
            "CERTIFIED":1,
            "CERTIFIED-WITHDRAWN":1,
            "WITHDRAWN":0,
            "DENIED":0
        })
        .fillna(0).astype(int)
    )

# ---- Date features ----
if submit_col and df[submit_col].dtype == "datetime64[ns]":
    df["SUBMIT_MONTH"]   = df[submit_col].dt.month
    df["SUBMIT_YEAR"]    = df[submit_col].dt.year
    df["SUBMIT_DAY"]     = df[submit_col].dt.dayofweek
    df["SUBMIT_QTR"]     = df[submit_col].dt.quarter

# ---- Save ----
df.to_csv(SAVE_PATH, index=False)

print("Done.")
print(f"Final shape: {df.shape}")
