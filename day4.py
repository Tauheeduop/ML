import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Load dataset
df = pd.read_csv("data/loan_dataset.csv")

#Preprocessing and data cleaning

# 2️⃣ Clean column names (remove extra spaces, unify case)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
print(df['loan_status'].value_counts())
# 3️⃣ Check column names and preview
print("Columns:", df.columns.tolist())
print(df.head())
# 4️⃣ Assume target column is 'loan_status' (common name)  
if 'loan_status' in df.columns:
    print(df['loan_status'].value_counts())
else:
    print("⚠️ Target column 'loan_status' not found. Please verify column names.")
# 5️⃣ Continue preprocessing only if target exists
# Numeric & categorical columns
num_cols= df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols=df.select_dtypes(include=['object']).columns.tolist()
# Remove target from feature lists
if 'loan_status' in num_cols: num_cols.remove('loan_status')
if 'loan_status' in cat_cols: cat_cols.remove('loan_status')
print("Numeric features:", num_cols)
print("Categorical features:", cat_cols)
# 6️⃣ Handle missing values
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
# 7️⃣ Remove duplicates
df.drop_duplicates(inplace=True)
# 8️⃣ Encode categorical
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
# 9️⃣ Scale numeric features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
#  🔟 Save cleaned dataset
df.to_csv("data/loan_dataset_cleaned.csv", index=False)
print("✅ Cleaned dataset saved. Shape:", df.shape)