import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
df=np.round(pd.read_csv('50_Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']]/10000)
np.random.seed(9)
df = df.sample(5)
print(df)
df = df.iloc[:,0:-1]
print(df)
df.iloc[1,0] = np.NaN
df.iloc[3,1] = np.NaN
df.iloc[-1,-1] = np.NaN
df0 = pd.DataFrame()

df0['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].mean())
df0['Administration'] = df['Administration'].fillna(df['Administration'].mean())
df0['Marketing Spend'] = df['Marketing Spend'].fillna(df['Marketing Spend'].mean())
df1 = df0.copy()

df1.iloc[1,0] = np.NaN

print(df1)
X = df1.iloc[[0,2,3,4],1:3]
print(X)
y = df1.iloc[[0,2,3,4],0]
lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[1,1:].values.reshape(1,2))
