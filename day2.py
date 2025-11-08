import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
#Preprocessing and data cleaning
#print(df.info())
df.dropna(inplace=True)

df['age'].fillna(df['age'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

#print(df.isnull().sum())

#Encoding categorical variables
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

#Feature Scaling
scaler = StandardScaler()
df [['age', 'fare']]= scaler.fit_transform(df[['age', 'fare']])

print(df.head())
print(df.describe())