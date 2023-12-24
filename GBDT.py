import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
plt.style.use("seaborn-paper")

from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from time import time


filepath = 'the absolute path of the data files'
df = pd.read_csv(filepath)
#print(df.columns)
df = df.set_index('Biomass_feedstock')

from sklearn.preprocessing import LabelEncoder
mt_lb = LabelEncoder()
df['Metal_type'] = mt_lb.fit_transform(df['Metal_type'].values)


#sns.pairplot(df[df.columns],diag_kind = 'kde')

start = time()
from sklearn.model_selection import train_test_split

features = ['pH_H2O','C', '(O+N)/C', 'O/C', 'H/C', 'Ash',  'SA', 'CEC', 'T',
       'pH_sol', 'C0', 'Metal_type']
X = df[features]
y = df['AE']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)


def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))


from sklearn.ensemble import GradientBoostingRegressor

parameters1 = {
    'n_estimators':[990],
    'learning_rate':[0.01], 
    'max_depth':[3],  
    'min_samples_leaf':[10], 
    'min_samples_split':[5], 
    'loss':['huber'], 
    'random_state' :[42],
    }

R2 = make_scorer(rmse,greater_is_better = False)
gbr = GradientBoostingRegressor()
clf1 = GridSearchCV(gbr, parameters1, verbose=0, cv=10,scoring = R2)
clf1.fit(X_train, y_train)
gbr = GradientBoostingRegressor(**clf1.best_params_)

gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
y_pred = pd.Series(y_pred,index = y_test.index)
end = time()
print('run time：{:.3f}'.format(end-start))

RMSE = rmse(y_test,y_pred)
R2 = r2_score(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
print('RMSE{:.3f}'.format(RMSE))
print('GBDT{:.2f}'.format(R2))
print('GBDT{:.3f}'.format(MAE))
print('GBDT:{}'.format(clf1.best_params_))







