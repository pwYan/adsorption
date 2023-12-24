# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:12:00 2019

@author: carrt
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
#import xgboost as xgb
#from xgboost import XGBRegressor
#from xgboost import plot_importance
#plt.style.use("seaborn-paper")
#sns.set_style('white')

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')
plt.style.use("seaborn-paper")
from time import time


filepath = 'the absolute path of the data files'
df = pd.read_csv(filepath)
#print(df.columns)
df = df.set_index('Biomass_feedstock')

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
mt_lb = LabelEncoder()
df['Metal_type'] = mt_lb.fit_transform(df['Metal_type'])


features = ['pH_H2O', 'C', '(O+N)/C', 'O/C', 'H/C', 'Ash',  'SA', 'CEC', 'T',
       'pH_sol', 'C0','Metal_type']
X = df[features]
y = df['AE']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)


def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

start = time()   
params = {       
    'colsample_bytree': [0.7],
    'gamma': [0],
    'learning_rate': [0.06], 
    'max_depth': [5],
    'min_child_weight':[5] ,
    'n_estimators': [1900],
    'seed': [8],
    'subsample': [0.8],
    'verbosity':[0]
    }
#np.arange(a,b,c)
R2 = make_scorer(r2_score,greater_is_better = True)

gbm = XGBRegressor()
clf3 = GridSearchCV(gbm, params, verbose=0, cv=5, scoring = R2)
clf3.fit(X_train,y_train)
XGBoost = XGBRegressor(**clf3.best_params_)

XGBoost.fit(X_train, y_train)
y_pred_xgb = XGBoost.predict(X_test)
y_pred_xgb = pd.Series(y_pred_xgb,index = y_test.index)
end = time()
print('run time：{:.3f}s'.format(end-start))

RMSE_xgb = rmse(y_test,y_pred_xgb)
R2_xgb = r2_score(y_test,y_pred_xgb)
MAE_xgb = mean_absolute_error(y_test,y_pred_xgb)
print('RMSE{:.5f}'.format(RMSE_xgb))
print('R2{:.5f}'.format(R2_xgb))
print('MAE{:.5f}'.format(MAE_xgb))
print('Optimal parameters,score'.format(clf3.best_params_,clf3.best_score_))







