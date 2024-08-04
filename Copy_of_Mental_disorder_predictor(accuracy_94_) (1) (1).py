#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.subplots as sp
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# In[21]:


# Importing dataset
data=pd.read_csv('Dataset-Mental-Disorders.csv')
data.head(2)


# In[22]:


data.dropna()


# In[23]:


data['Mood Swing'].replace('YES ', 'YES', inplace=True)


# In[24]:


data['Mood Swing'].value_counts()


# In[25]:


data
data.rename(columns={'Ignore & Move-On': 'ignore_and__move_on'}, inplace=True)


# In[26]:


Yes_No_column = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation', 'Aggressive Response', 'ignore_and__move_on', 'Nervous Break-down', 'Admit Mistakes', 'Overthinking']

for column in Yes_No_column:
    data[column] = data[column].replace({'YES ': 'YES', ' NO ': 'NO', 'NO ': 'NO', 'YES': 'YES', 'NO': 'NO'})
    data[column] = data[column].map({'YES': 1, 'NO': 0})


# In[27]:


data.drop(columns=['Patient Number'], inplace=True)


# In[28]:


for column in ['Overthinking', 'Sexual Activity', 'Concentration','Optimisim']:
    data[column] = data[column].astype(str).str.extract(r'(\d+)').astype(int)


# In[29]:


X=data.drop(columns=['Expert Diagnose'])
y=data['Expert Diagnose']


# In[30]:


X=pd.get_dummies(X, columns=['Sadness','Euphoric','Exhausted','Sleep dissorder'])


# In[31]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[32]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

feature_selection = SelectKBest(score_func=mutual_info_classif, k=10)
feature_selection = feature_selection.fit(X, y)


features_scores = pd.Series(feature_selection.scores_, index=X.columns)
top_features = features_scores.nlargest(10).index


x_train, x_test, y_train, y_test = train_test_split(X[top_features], y, test_size=0.3, random_state=2, shuffle=True)


param_grid = {
    'bootstrap': [True],
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 13, 22],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [4, 7, 9],
    'max_features': ['sqrt', 'log2']
}


model = RandomForestClassifier()

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(x_train, y_train)


y_predict = grid_search.predict(x_test)


print(classification_report(y_test, y_predict))


# In[38]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

feature_selection = SelectKBest(score_func=mutual_info_classif, k=10)
feature_selection = feature_selection.fit(X, y)


features_scores = pd.Series(feature_selection.scores_, index=X.columns)
top_features = features_scores.nlargest(10).index


x_train, x_test, y_train, y_test = train_test_split(X[top_features], y, test_size=0.3, random_state=2, shuffle=True)



param_grid = {
    'criterion': ['gini'], 'max_depth': [4], 'max_features': ['auto'], 'n_estimators': [200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')


# Fit the model
grid_search.fit(x_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)



# Evaluate the model with the best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)
print(classification_report(y_test, y_pred))


# In[40]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

feature_selection = SelectKBest(score_func=mutual_info_classif, k=10)
feature_selection = feature_selection.fit(X, y)


features_scores = pd.Series(feature_selection.scores_, index=X.columns)
top_features = features_scores.nlargest(10).index


x_train, x_test, y_train, y_test = train_test_split(X[top_features], y, test_size=0.2, random_state=42, shuffle=True)



param_grid = {
    'criterion': ['gini'], 'max_depth': [4], 'max_features': ['auto'], 'n_estimators': [200]
}

# Initialize the model
# rf = LogisticRegression()
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# Define the parameter grid
param_grid = {
    'logreg__C': [ 11],
    'logreg__penalty': ['l1'],
    'logreg__solver': [ 'liblinear'],
    'logreg__max_iter': [102]
}


# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy',verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(x_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)



# Evaluate the model with the best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)
print(classification_report(y_test, y_pred))


# In[42]:


import pickle
best_model = grid_search.best_estimator_
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


y_pred_loaded = loaded_model.predict(x_test)
print(classification_report(y_test, y_pred_loaded))


# In[ ]:




