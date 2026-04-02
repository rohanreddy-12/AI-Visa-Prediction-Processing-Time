import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

COLORS = sns.color_palette("muted")
TARGET = "PROCESSING_DAYS"
FIG_DIR = "."

# ---- Load data ----
try:
    df = pd.read_csv("h1b_cleaned.csv", low_memory=False)
    print(f"Loaded cleaned data: {df.shape}")
except FileNotFoundError:
    print("Cleaned file not found, doing quick clean...")
    df = pd.read_csv("h1b_kaggle.csv", low_memory=False)

    df.columns = df.columns.str.strip().str.upper().str.replace(r"\s+","_", regex=True)
    df.drop_duplicates(inplace=True)

    for c in ["SUBMIT_DATE", "DECISION_DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "SUBMIT_DATE" in df.columns and "DECISION_DATE" in df.columns:
        df[TARGET] = (df["DECISION_DATE"] - df["SUBMIT_DATE"]).dt.days
        df = df[(df[TARGET] >= 0) & (df[TARGET] <= 730)]

    df.dropna(subset=[TARGET], inplace=True)

    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = num_cols.fillna(num_cols.median())

    for c in df.select_dtypes("object"):
        df[c] = df[c].fillna(df[c].mode()[0]).str.strip().str.upper()

# ---- Save helper ----
def save(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---- EDA 1: Processing time ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].hist(df[TARGET].dropna(), bins=60, color=COLORS[0])
ax[0].set(title="Processing Days", xlabel="Days", ylabel="Count")

stats.probplot(df[TARGET].dropna(), dist="norm", plot=ax[1])
ax[1].set_title("Q-Q Plot")

save(fig, "eda1_processing_time")

# ---- EDA 2: Case status ----
if "CASE_STATUS" in df.columns:
    counts = df["CASE_STATUS"].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    counts.plot.bar(ax=ax[0], color=COLORS)
    ax[0].set_title("Case Status Count")

    ax[1].pie(counts, labels=counts.index, autopct="%1.1f%%")
    ax[1].set_title("Case Status Share")

    save(fig, "eda2_case_status")

# ---- EDA 3: Wage ----
if "PREVAILING_WAGE" in df.columns:
    wage = df["PREVAILING_WAGE"].dropna()
    wage = wage[(wage > 0) & (wage < wage.quantile(0.99))]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].hist(wage, bins=60, color=COLORS[2])
    ax[0].set_title("Wage Distribution")

    sample = wage.sample(min(5000, len(wage)), random_state=42)
    ax[1].scatter(sample,
                  df.loc[sample.index, TARGET],
                  alpha=0.3, s=10)
    ax[1].set_title("Wage vs Days")

    save(fig, "eda3_wage")

# ---- EDA 4: Year trends ----
year_col = next((c for c in ["SUBMIT_YEAR", "YEAR"] if c in df.columns), None)

if year_col:
    yearly = df.groupby(year_col)[TARGET].agg(["count", "mean"]).reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].bar(yearly[year_col], yearly["count"], color=COLORS[3])
    ax[0].set_title("Applications per Year")

    ax[1].plot(yearly[year_col], yearly["mean"], marker="o")
    ax[1].set_title("Avg Processing Days")

    save(fig, "eda4_yearly")

# ---- EDA 5: Correlation ----
num_df = df.select_dtypes(include=np.number)
num_df = num_df[[c for c in num_df if num_df[c].nunique() > 1]]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")

save(fig, "eda5_corr")

# ==================================================
# FEATURE ENGINEERING
# ==================================================

print("\nFeature Engineering...")

# Seasonal index
if "SUBMIT_MONTH" in df.columns:
    global_avg = df[TARGET].mean()
    monthly_avg = df.groupby("SUBMIT_MONTH")[TARGET].mean()
    df["SEASONAL_INDEX"] = df["SUBMIT_MONTH"].map(
        lambda m: monthly_avg.get(m, global_avg) / global_avg
    )

# Employer stats
emp_col = next((c for c in ["EMPLOYER_NAME", "EMPLOYER"] if c in df.columns), None)

if emp_col:
    emp_avg = df.groupby(emp_col)[TARGET].mean()
    df["EMPLOYER_AVG_DAYS"] = df[emp_col].map(emp_avg)
    df["EMPLOYER_VOLUME"] = df[emp_col].map(df[emp_col].value_counts())

# Wage bucket
if "PREVAILING_WAGE" in df.columns:
    df["WAGE_BUCKET"] = pd.cut(
        df["PREVAILING_WAGE"],
        bins=[0, 40000, 70000, 100000, 150000, np.inf]
    )
    df["LOG_WAGE"] = np.log1p(df["PREVAILING_WAGE"])

# ---- Feature vs target ----
features = [c for c in [
    "SEASONAL_INDEX", "EMPLOYER_AVG_DAYS", "LOG_WAGE"
] if c in df.columns]

if features:
    fig, ax = plt.subplots(1, len(features), figsize=(5*len(features), 5))
    if len(features) == 1:
        ax = [ax]

    for a, f in zip(ax, features):
        sample = df[[f, TARGET]].dropna().sample(min(3000, len(df)))
        a.scatter(sample[f], sample[TARGET], alpha=0.3)
        a.set_title(f)

    save(fig, "eda_features")

# ---- Save final dataset ----
df.to_csv("h1b_features.csv", index=False)

print("Done.")
print(f"Final shape: {df.shape}")
