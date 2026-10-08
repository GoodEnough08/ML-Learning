import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator

df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])
# print(df.head())
# print(df.isnull().mean() * 100)

# X = df.drop('Survived', axis=1)
# Y = df['Survived']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Copy to avoid SettingWithCopyWarning
# X_train = X_train.copy()
# X_test = X_test.copy()

# X_train['Age_imputed'] = X_train['Age']
# X_test['Age_imputed'] = X_test['Age']

# X_train.loc[X_train['Age_imputed'].isnull(), 'Age_imputed'] = X_train['Age'].dropna().sample(X_train['Age'].isnull().sum(), random_state=42).values
# X_test.loc[X_test['Age_imputed'].isnull(), 'Age_imputed'] = X_train['Age'].dropna().sample(X_test['Age'].isnull().sum(), random_state=42).values

# print('Original variable variance: ', X_train['Age'].var())
# print('Variance after random imputation: ', X_train['Age_imputed'].var())

# dt = pd.read_csv('house-train.csv', usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])
# print(dt.head())
# print(dt.isnull().mean() * 100)

# dt['GarageQual_imputed'] = dt['GarageQual']
# dt['FireplaceQu_imputed'] = dt['FireplaceQu']

# dt.loc[dt['GarageQual_imputed'].isnull(), 'GarageQual_imputed'] = dt['GarageQual'].dropna().sample(dt['GarageQual'].isnull().sum(), random_state=42).values
# dt.loc[dt['FireplaceQu_imputed'].isnull(), 'FireplaceQu_imputed'] = dt['FireplaceQu'].dropna().sample(dt['FireplaceQu'].isnull().sum(), random_state=42).values

# print('\nOriginal GarageQual category proportions:\n', dt['GarageQual'].value_counts(normalize=True))
# print('Imputed GarageQual category proportions:\n', dt['GarageQual_imputed'].value_counts(normalize=True))

# print('\nOriginal FireplaceQu category proportions:\n', dt['FireplaceQu'].value_counts(normalize=True))
# print('Imputed FireplaceQu category proportions:\n', dt['FireplaceQu_imputed'].value_counts(normalize=True))
'''Using Missing Indicator'''
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf,y_train)

y_pred = clf.predict(X_test_trf)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
mi = MissingIndicator()

mi.fit(X_train)
X_train_missing = mi.transform(X_train)
X_test_missing = mi.transform(X_test)
X_train['Age_NA'] = X_train_missing
X_test['Age_NA'] = X_test_missing
si = SimpleImputer()

X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf2,y_train)

y_pred = clf.predict(X_test_trf2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)