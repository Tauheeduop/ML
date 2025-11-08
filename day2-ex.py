import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Loading the dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

#Preprocessing and data cleaning
#Count missing values for each column.
#print(df.isnull().sum())
#Fill missing numeric values (age) with median instead of mean.
df['age'].fillna(df['age'].median(), inplace=True)
#Fill missing categorical (embarked) with mode.
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
#Verify there are no missing values left.
#print(df.isnull().sum())

#Encoding categorical variables
le= LabelEncoder()
df['sex']= le.fit_transform(df['sex'])
df['embarked']= le.fit_transform(df['embarked'])

#df[['sex', 'embarked']]=le.fit_transform(df[['sex', 'embarked']])
#print(df.head(5))
#before scaling
#print(df[['age','fare']].describe())
#Feature Scaling
scaler= StandardScaler()
df[['age', 'fare']]= scaler.fit_transform (df[['age', 'fare']])
#after scaling
print (df[['age', 'fare']].describe())
