import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
df=pd.read_csv('train.csv',usecols=['Age','Fare','Survived']).dropna()
print(df.head())
print(df.shape)
x=df.drop("Survived", axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(np.mean(cross_val_score(clf,x,y,cv=10,scoring='accuracy')))
k_bin_age=KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
k_bin_fare=KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
t=ColumnTransformer([
    ('k_bin_age',k_bin_age,['Age']),
    ('k_bin_fare',k_bin_fare,['Fare'])
])
x_train_t=t.fit_transform(x_train)
x_test_t=t.transform(x_test)

clf.fit(x_train_t,y_train)
y_pred=clf.predict(x_test_t)
print(accuracy_score(y_test,y_pred))
print(np.mean(cross_val_score(clf,x_train_t,y_train,cv=10,scoring='accuracy')))

#manualbinning
x_train['age_bin']=pd.cut(x_train['Age'],bins=10,labels=False)
x_test['age_bin']=pd.cut(x_test['Age'],bins=10,labels=False)
x_train['fare_bin']=pd.cut(x_train['Fare'],bins=10,labels=False)
x_test['fare_bin']=pd.cut(x_test['Fare'],bins=10,labels=False)
clf.fit(x_train[['age_bin','fare_bin']],y_train)
y_pred=clf.predict(x_test[['age_bin','fare_bin']])
print(accuracy_score(y_test,y_pred))
print(np.mean(cross_val_score(clf,x_train[['age_bin','fare_bin']],y_train,cv=10,scoring='accuracy')))