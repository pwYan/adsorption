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

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import plot_importance
from mlens.ensemble import SuperLearner
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
import warnings
from time import time
warnings.filterwarnings('ignore')
plt.style.use("seaborn-paper")


filepath = 'the absolute path of the data files'
df = pd.read_csv(filepath)
#print(df.columns)
df = df.set_index('Biomass_feedstock')

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
mt_lb = LabelEncoder()
df['Metal_type'] = mt_lb.fit_transform(df['Metal_type'])

features = ['pH_H2O', 'C', '(O+N)/C', 'O/C', 'H/C', 'Ash',  'SA', 'CEC', 'T',
       'pH_sol', 'C0', 'Metal_type']
X = df[features]
y = df['AE']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)



params = {       
    'colsample_bytree': 0.7,
    'gamma': 0,
    'learning_rate': 0.06, 
    'max_depth': 7,
    'min_child_weight': 5,
    'n_estimators': 1700,
    'seed': 8,
    'subsample': 0.8,
    'verbosity':0
    }


parameters1 = {
    'n_estimators':400,
    'learning_rate':0.1, 
    'max_depth':6,  
    'min_samples_leaf':1, 
    'min_samples_split':2, 
    'loss':'huber', 
    'random_state' :42,
    }



start = time()
xgb = XGBRegressor(**params)
svr = SVR(kernel='linear')
#rf = RandomForestRegressor(**parameters2)
DT = DecisionTreeRegressor(max_depth = 8)
from sklearn.ensemble import GradientBoostingRegressor
GBDT = GradientBoostingRegressor(**parameters1)

ensemble = SuperLearner(scorer = mean_squared_error,verbose = 2)
ensemble.add([xgb,GBDT,DT])
ensemble.add_meta(svr)

ensemble.fit(X_train,y_train)

pred_stack = ensemble.predict(X_test)
pred_stack = pd.DataFrame(pred_stack,index = X_test.index)
end = time()
print('run time：{:.3f}s'.format(end-start))



def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))


RMSE_xgb = rmse(y_test,pred_stack)
R2_xgb = r2_score(y_test,pred_stack)
MAE_xgb = mean_absolute_error(y_test,pred_stack)
print('Stacking{:.5f}'.format(RMSE_xgb))
print('Stacking{:.5f}'.format(R2_xgb))
print('Stacking{:.5f}'.format(MAE_xgb))



f,ax = plt.subplots(figsize = (15,10))
plt.scatter(pred_stack, y_test, c = "blue",  label = "Test data",s = 100)
plt.grid()
ax.set_title("Stacking Regression",fontsize = '20')
ax.set_xlabel("Predicted values",fontsize = '15')
ax.set_ylabel("Real values",fontsize = '15')
plt.xlim(0.01,1.7)
plt.ylim(0,1.7)
ax.legend(loc = "upper left",fontsize = '20')
ax.tick_params(labelsize=15)
x = np.arange(-0.5,2,0.01)
plt.plot(x, x, c = "red")
plt.xlim(0,1.6)
plt.ylim(0,1.6)
plt.show()


