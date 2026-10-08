from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
df=pd.read_csv('titanic_toy.csv')
# print(df.head())
# print(df.isnull().sum())
X_train,X_test,Y_train,Y_test=train_test_split(df.drop('Survived',axis=1),df['Survived'],test_size=0.2,random_state=42)
# si=SimpleImputer(missing_values=np.nan,strategy='median')
# X_train=si.fit_transform(X_train)
# X_test=si.transform(X_test)
'''Using Column Transformer'''
# imputer1 = SimpleImputer(strategy='median')
# imputer2 = SimpleImputer(strategy='mean')
# trf = ColumnTransformer([
#     ('imputer1',imputer1,['Age']),
#     ('imputer2',imputer2,['Fare'])
# ],remainder='passthrough')
# trf.fit_transform(X_train)
# trf.transform(X_test)
'''Arbitraty Value Imputation'''
imputer1 = SimpleImputer(strategy='constant',fill_value=99)
imputer2 = SimpleImputer(strategy='constant',fill_value=999)
trf=ColumnTransformer([
    ('imputer1',imputer1,['Age']),
    ('imputer2',imputer2,['Fare'])
],remainder='passthrough')
trf.fit_transform(X_train)
trf.transform(X_test)