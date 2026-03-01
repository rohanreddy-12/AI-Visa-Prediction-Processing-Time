import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


input_file = 'h1b_data_cleaned_for_eda.csv'
print(f"Loading {input_file} for Exploratory Data Analysis...")
df = pd.read_csv(input_file)

#country_avg:
country_avg = df.groupby('applicant_country')['processing_time'].mean().to_dict()
df['country_avg_processing_time'] = df['applicant_country'].map(country_avg)

office_workload = df['processing_office'].value_counts().to_dict()
df['office_workload'] = df['processing_office'].map(office_workload)

# EDA:
sns.set_theme(style="whitegrid")
print("Generating charts...")

# Chart 1: Processing Time Distributions
plt.figure(figsize=(10, 5))
sns.histplot(df['processing_time'], bins=25, kde=True, color='dodgerblue')
plt.title('Distribution of Visa Processing Times')
plt.xlabel('Processing Time (Days)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Chart 2: Processing Time by Visa Case Status
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='case_status', y='processing_time', palette='pastel')
plt.title('Processing Time Distribution by Visa Case Status')
plt.xlabel('Case Status')
plt.ylabel('Processing Time (Days)')
plt.tight_layout()
plt.show()

# Chart 3: Processing Time by Regions (Applicant Origin)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['applicant_country'] != 'UNKNOWN'], x='applicant_country', y='processing_time', palette='Set2')
plt.title('Processing Time Distribution by Applicant Origin')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 4: Trends based on Workload at Processing Centers
plt.figure(figsize=(12, 6))
sns.barplot(data=df[df['processing_office'] != 'UNKNOWN'], x='processing_office', y='processing_time', palette='magma', errorbar=None)
plt.title('Average Processing Time by Processing Center Workload')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Chart 5: Trends based on Seasons
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='season', y='processing_time', palette='viridis', errorbar=None)
plt.title('Average Processing Time by Season')
plt.tight_layout()
plt.show()

#correlations:
print("Calculating correlations and feature importance...")

# Temporarily encode the wage category into numbers just for the math
wage_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
df['wage_encoded'] = df['wage_category'].map(wage_map).fillna(0)

numeric_cols = ['processing_time', 'applicant_age', 'prevailing_wage', 'wage_encoded', 'country_avg_processing_time', 'office_workload']

# Chart 6: Correlation Heatmap
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Heatmap: Features vs. Processing Time')
plt.tight_layout()
plt.show()

