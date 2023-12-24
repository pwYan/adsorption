

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import cmath
#from sklearn import linear_model
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.svm import SVR
#from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate

#from sklearn.linear_model import  Ridge 
from sklearn.model_selection import GridSearchCV
from time import time
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('the absolute path of the data files') 
#print(data.columns)

data = data.set_index('Biomass_feedstock')
mt_lb = LabelEncoder()
data['Metal_type'] = mt_lb.fit_transform(data['Metal_type'].values)

features = ['pH_H2O','C', '(O+N)/C', 'O/C', 'H/C', 'Ash', 'SA', 'CEC', 'T',
       'pH_sol', 'HM/Bio', 'Metal_type']
X_data = data[features]
Y_data = data['AE']


X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size = 0.20,random_state = 42)

def rmse(Y_data,Y_pred):
    return np.sqrt(mean_squared_error(Y_data,Y_pred))
R2 = make_scorer(r2_score,greater_is_better = True)


start = time()
knn_model = KNeighborsRegressor(n_neighbors=2, weights='uniform', 
                                algorithm='auto', leaf_size=38, p=1, metric='minkowski',
                                metric_params=None, n_jobs=None)



knn_model.fit(X_train, Y_train)
Y_train_pred = knn_model.predict(X_train)
Y_test_pred = knn_model.predict(X_test)
end = time()
print('run time：{:.4f}s'.format(end-start))
score_test = cross_val_score(knn_model,X_test,Y_test, cv=5)
print('score：{:.3f}s', score_test)  
print("Accuracy_score: %0.2f (+/- %0.2f)" % (score_test.mean(), score_test.std() * 2))  

score_train = cross_val_score(knn_model,X_train,Y_train, cv=n)  
print('score：{:.3f}s', score_train)  
print("Accuracy_score: %0.2f (+/- %0.2f)" % (score_train.mean(), score_train.std() * 2))


RMSE_test = rmse(Y_test,Y_test_pred)
R2_test = r2_score(Y_test,Y_test_pred)
MAE_test = mean_absolute_error(Y_test,Y_test_pred)
print('RMSE_test:{:.4f}'.format(RMSE_test))
print('R2_test:{:.4f}'.format(R2_test))
print('MAE_test:{:.4f}'.format(MAE_test))
Y_test_pred = pd.Series(Y_test_pred,index = Y_test.index)

RMSE_train = rmse(Y_train,Y_train_pred)
R2_train = r2_score(Y_train,Y_train_pred)
MAE_train = mean_absolute_error(Y_train,Y_train_pred)
print('RMSE_train:{:.4f}'.format(RMSE_train))
print('R2_train:{:.4f}'.format(R2_train))
print('MAE_train:{:.4f}'.format(MAE_train))
Y_train_pred = pd.Series(Y_train_pred,index = Y_train.index)

