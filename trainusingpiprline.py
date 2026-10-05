import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("train.csv")
print(df.head())
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)
print(X_train.head())
tr1=ColumnTransformer([('impute_age',SimpleImputer(),[2]),('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])],remainder='passthrough')
tr2=ColumnTransformer([('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[1,3])],remainder='passthrough')
tr3=ColumnTransformer([('scale',MinMaxScaler(),slice(0,10))])
tr4 = SelectKBest(score_func=chi2,k=8)
tr5=DecisionTreeClassifier()
pipe=make_pipeline(tr1,tr2,tr3,tr4,tr5)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))