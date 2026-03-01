# importing library
import pandas as pd
import numpy as np
# reading the file
input_file = 'h1b_visa_dataset_v2.csv'
print(f"Loading dataset: {input_file}...")
# converting to the dataframe(which is in the form of table)
df = pd.read_csv(input_file)
# printing the dimensions(rows,columns)
print(f"Initial Shape: {df.shape}")
# converting the dates(used pandas datetime for large data)
df['application_date'] = pd.to_datetime(df['application_date'], dayfirst=True, errors='coerce')
df['decision_date'] = pd.to_datetime(df['decision_date'], dayfirst=True, errors='coerce')
# removes the rows which don't have the both dates
df.dropna(subset=['application_date', 'decision_date'], inplace=True)

categorical_cols = ['processing_office', 'applicant_country', 'education_level', 'occupation_category', 'wage_category']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

if 'applicant_age' in df.columns:
    df['applicant_age'] = df['applicant_age'].fillna(df['applicant_age'].median())

df['processing_time'] = (df['decision_date'] - df['application_date']).dt.days

df = df[df['processing_time'] >= 0]
# dividing into seasons
def get_season(date):
    month = date.month
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

df['season'] = df['application_date'].apply(get_season)
df_clean = df.copy()
df_clean.to_csv('h1b_data_cleaned_for_eda.csv', index=False)
# one hot encoding
if 'full_time_position' in df.columns:
    df['full_time_position'] = df['full_time_position'].map({'Y': 1, 'N': 0})
# categorical encoding
wage_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
if 'wage_category' in df.columns:
    df['wage_category_encoded'] = df['wage_category'].map(wage_map).fillna(0)

edu_map = {"ASSOCIATE'S": 0, "BACHELOR'S": 1, "MASTER'S": 2, "DOCTORATE": 3}
df['education_level_encoded'] = df['education_level'].map(edu_map).fillna(0)

cols_to_encode = ['processing_office', 'applicant_country', 'case_status', 'season', 'occupation_category']
df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

drop_cols = ['application_date', 'decision_date', 'job_title', 'education_level', 'wage_category', 'prevailing_wage']
df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], inplace=True)

output_file = 'h1b_data_preprocessed_final.csv'
df_encoded.to_csv(output_file, index=False)

print(f"\nSuccess! Preprocessing complete.")
print(f"Final dataset shape: {df_encoded.shape}")

print(f"Saved cleaned and encoded data to: {output_file}")
