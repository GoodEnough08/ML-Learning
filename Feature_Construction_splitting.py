import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import seaborn as sns
# df = pd.read_csv('train.csv')[['Age','Pclass','SibSp','Parch','Survived']]
# print(df.head())
# df.dropna(inplace=True)
# print(df.head())
# X = df.iloc[:,0:4]
# y = df.iloc[:,-1]
# X['Family_size'] = X['SibSp'] + X['Parch'] + 1
# print(X.head())
# def myfunc(num):
#     if num == 1:
#         return 0
#     elif num >1 and num <=4:
#         return 1
#     else:
#         return 2
# X['Family_type'] = X['Family_size'].apply(myfunc)
# print(X.head())
# X.drop(columns=['SibSp','Parch','Family_size'],inplace=True)

# print('Accuracy with feature engineering',np.mean(cross_val_score(Pipeline([('scaler',StandardScaler()),('LogisticRegression',LogisticRegression())]),X,y,scoring='accuracy',cv=20)))
df=pd.read_csv('train.csv')
print(df.head())
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
print(df[['Title','Name']])
print((df.groupby('Title')['Survived'].mean().sort_values(ascending=False)))
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
print(df.groupby('Is_Married')['Survived'].mean())