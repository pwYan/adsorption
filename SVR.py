
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


filepath = 'the absolute path of the data files'

df = pd.read_csv(filepath)
print(df.columns)
df = df.set_index('Biomass_feedstock')
#sns.pairplot(df[df.columns],diag_kind = 'kde')


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
mt_lb = LabelEncoder()
df['Metal_type'] = mt_lb.fit_transform(df['Metal_type'].values)

features = ['pH_H2O', 'C', '(O+N)/C', 'O/C', 'H/C', 'Ash',  'SA', 'CEC', 'T',
       'pH_sol', 'HM/Bio', 'Metal_type']
X = df[features]
y = df['AE']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)

def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

R2 = make_scorer(r2_score,greater_is_better = True)

from sklearn.svm import SVR
parameters2 = {
        'C': [8],
        'gamma': [1],      
        }


svr = SVR()
clf2 = GridSearchCV(svr, parameters2, verbose=0, cv=10,scoring = R2)
clf2.fit(X_train, y_train)
svr = SVR(**clf2.best_params_)

svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
y_pred_svr = pd.Series(y_pred_svr,index = y_test.index)
#y = pd.concat([y_test,y_pred_svr],axis=1)
#y.rename(columns={'AE':'Truth',0:'Prediction'},inplace = True)
#y.to_excel('Save the data at path')



RMSE_svr = rmse(y_test,y_pred_svr)
R2_svr = r2_score(y_test,y_pred_svr)
MAE_svr = mean_absolute_error(y_test,y_pred_svr)
print('SVR{:.4f}'.format(RMSE_svr))
print('SVR{:.4f}'.format(R2_svr))
print('SVR{:.4f}'.format(MAE_svr))
print('{}'.format(clf2.best_params_))





