
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from time import time


filepath = 'the absolute path of the data files'
df = pd.read_csv(filepath)
#print(df.columns)
df = df.set_index('Biomass_feedstock')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
mt_lb = LabelEncoder()
df['Metal_type'] = mt_lb.fit_transform(df['Metal_type'].values)

features = [ 'pH_H2O','C', '(O+N)/C', 'O/C', 'H/C', 'Ash', 'SA', 'CEC', 'T',
       'pH_sol', 'C0', 'Metal_type']
X = df[features]
y = df['AE']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)

def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

R2 = make_scorer(r2_score,greater_is_better = True)

start = time()
from sklearn.tree import DecisionTreeRegressor
parameters2 = {
        'max_depth': [8],        
        }
dt = DecisionTreeRegressor()
clf2 = GridSearchCV(dt, parameters2, verbose=0, cv=5,scoring = R2)
clf2.fit(X_train, y_train)
dt = DecisionTreeRegressor(**clf2.best_params_)
#np.arange(a,b,c)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
end = time()
print('run time：{:.3f}s'.format(end-start))
#np.arange(1,10,1)

RMSE_dt = rmse(y_test,y_pred_dt)
R2_dt = r2_score(y_test,y_pred_dt)
MAE_dt = mean_absolute_error(y_test,y_pred_dt)
print('RMSE{:.4f}'.format(RMSE_dt))
print('R2{:.4f}'.format(R2_dt))
print('MAE{:.4f}'.format(MAE_dt))
print('Optimal parameters：{}'.format(clf2.best_params_))
y_pred_dt = pd.Series(y_pred_dt,index = y_test.index)






