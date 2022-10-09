import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
time_start = time.time()


'''
1.load and check the data
'''
data = pd.read_csv('insurance.csv')
print('View missing values ------------------------------------------------------------\n', data.isnull().sum()) # Checking for missing values in the dataset
print('\n output the first 5 data -----------------------------------------------------------------\n', data.head())
print('different values of sex：', data.sex.unique())
print('the number of different values of sex：', data.sex.nunique())
print('different values of smoker：', data.smoker.unique())
print('the number of different values of smoker：', data.smoker.nunique())
print('different values of region：', data.region.unique())
print('the number of different values of region：', data.region.nunique())
print('different values of children：', data.children.unique())
print('the number of different values of children：', data.children.nunique())


'''
2.preprocessing
'''
print('\n Convert non-numeric data to numeric：')
data_new = data.copy()
class_le = LabelEncoder()
# data_label_num.iloc[:,1] = class_le.fit_transform(data_label_num.iloc[:,1].values)
# label= np.unique(data_label_num.iloc[:,1])
data_new['sex'] = class_le.fit_transform(data_new['sex'].values)
data_new['smoker'] = class_le.fit_transform(data_new['smoker'].values)
data_new['region'] = class_le.fit_transform(data_new['region'].values)
print('label sex:', np.unique(data_new['sex']))
print('label smoker:', np.unique(data_new['smoker']))
print('label region:', np.unique(data_new['region']))
print('\n output the first 5 data --------------------------------------------------------\n', data_new.head())


'''
3. dividing X and Y
'''
y = data_new.charges
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
X = data_new[features]
y = y.to_frame()


'''
4. data normalization
'''
from sklearn import preprocessing
scaler_x = preprocessing.MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
print('\n check the shape of X and Y --------------------------------------------')
print(X.shape)

scaler_y = preprocessing.MinMaxScaler()
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1,1))

print(y_scaled.shape)
y_scaled = y_scaled.reshape(-1)
print(y_scaled.shape)

# check
print('\n check data normalization----------------------------------------')
print('X_scaled[0:5]:\n', X_scaled[0:5])
print('y_scaled[0:5]:', y_scaled[0:5])


# save data: X_scaled(1338, 6), y_scaled(1338, )
print(X_scaled.shape)
print(y_scaled.shape)
insurance_afterpreprocess = {}
insurance_afterpreprocess['X'] = X_scaled
insurance_afterpreprocess['y'] = y_scaled
np.save(os.getcwd() + '/result_data/insurance_afterpreprocess.npy', insurance_afterpreprocess)
print('save insurance_afterpreprocess.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)

loadData = np.load(os.getcwd() + '/result_data/insurance_afterpreprocess.npy', allow_pickle=True)
insurance_afterpreprocess = loadData.tolist()
X = insurance_afterpreprocess['X']
y = insurance_afterpreprocess['y']

trails = 20
for th in range(trails):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=338, random_state=th)  # shuffle:bool, default=True
    print('\n the division of training and testing data --------------------------------trail:', th)
    print('X_train:', X_train.shape)
    print('y_train:', y_test.shape)
    print('X_train[0]:', X_train[0])
    print('y_train[0]:', y_train[0])



