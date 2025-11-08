import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

df=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
#Preprocessing and data cleaning
#print(df['sex'].value_counts())
#normalize=True gives the proportions/percentages
#.value_counts() categorical data frequency count while numeric for we use .discribe()
#print(df['embarked'].value_counts(normalize=True))

#basic information about the dataset
#df.shape          # rows × columns
#df.info()         # data types, null values
#df.describe()   # stats for numeric columns
#df.describe(include='object')  # stats for categorical columns

#univariate analysis
#numeric columns

sns.histplot(df['age'], kde=True, bins=20)
plt.show()

df['class'].value_counts()
sns.countplot(x='class', data=df)
plt.show()

#sns.barplot(x="sex", y="survived", data=df)
sns.boxplot(x="pclass", y="age", data=df)

plt.show()

df.corr(numeric_only=True)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()