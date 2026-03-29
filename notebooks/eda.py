import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed_data.csv")

# -------------------------------
# 1. Distribution Plot
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["processing_time"], kde=True)
plt.title("Distribution of Visa Processing Times")
plt.xlabel("Processing Time (Days)")
plt.ylabel("Frequency")
plt.savefig("processing_time_dist.png")
plt.close()


# -------------------------------
# 2. Visa Type vs Processing Time
# -------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="visa_type", y="processing_time", data=df)
plt.title("Processing Time by Visa Type")
plt.savefig("visa_type_boxplot.png")
plt.close()


# -------------------------------
# 3. Country vs Processing Time
# -------------------------------
plt.figure(figsize=(10,5))
sns.boxplot(x="country", y="processing_time", data=df)
plt.title("Processing Time by Applicant Country")
plt.xticks(rotation=45)
plt.savefig("country_boxplot.png")
plt.close()


# -------------------------------
# 4. Processing Office Analysis
# -------------------------------
plt.figure(figsize=(10,5))
sns.barplot(x="processing_office", y="processing_time", data=df)
plt.title("Avg Processing Time by Office")
plt.xticks(rotation=45)
plt.savefig("office_barplot.png")
plt.close()


# -------------------------------
# 5. Seasonal Analysis
# -------------------------------
# create season column
df["season"] = pd.cut(df["month"],
                     bins=[0,3,6,9,12],
                     labels=["Winter","Spring","Summer","Fall"])

plt.figure(figsize=(8,5))
sns.barplot(x="season", y="processing_time", data=df)
plt.title("Processing Time by Season")
plt.savefig("season_barplot.png")
plt.close()


# -------------------------------
# 6. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("✅ All EDA plots generated successfully")