import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
a=np.array([1, 2, 3])
b=np.array([4, 5, 6])
print(f"{a.mean()} and {b.mean()} are the means of a and b respectively.")
'''
df=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#print(df['age'].unique())
sns.countplot(x='sex', data=df)
plt.show()