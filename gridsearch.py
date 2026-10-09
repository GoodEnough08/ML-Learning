import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset('iris')
print(df.head())
X = df.iloc[:,0:4]
y = df.iloc[:,4]
print(X.head())
# print(y.head())
# model_svm=SVC()
# pipeline = Pipeline([('scaler', StandardScaler()), ('LogisticRegression', LogisticRegression())])
# trf=Pipeline([('scaler',StandardScaler()),('SVC',SVC())])
# param_grid = {
#     'LogisticRegression__C': [0.1, 1, 10, 100]
# }
# param_grid_svm = {
#     'SVC__C': [0.1, 1, 10, 100],
#     'SVC__kernel': ['rbf', 'linear']
# }

# grid_search_svm = GridSearchCV(trf, param_grid_svm, cv=10)

     
# grid_search = GridSearchCV(pipeline, param_grid, cv=10)

# grid_search.fit(X, y)
# grid_search_svm.fit(X, y)
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
# print("Best parameters: ", grid_search_svm.best_params_)
# print("Best score: ", grid_search_svm.best_score_)

'''RandomizedSearchCV'''
from sklearn.model_selection import RandomizedSearchCV
pipeline=Pipeline([['scaler',StandardScaler()],['LogisticRegression',LogisticRegression()]])
param_distributions = {
    'LogisticRegression__C': [0.1, 1, 10, 100]
}

random_search = RandomizedSearchCV(pipeline, param_distributions, cv=10, n_iter=5)
random_search.fit(X, y)

print("Best parameters (RandomizedSearchCV): ", random_search.best_params_)
print("Best score (RandomizedSearchCV): ", random_search.best_score_)
