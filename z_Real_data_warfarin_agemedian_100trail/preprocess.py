import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


'''
1.load and check the data
'''
data0 = pd.read_csv('warfarin_addINR.csv')
print('View missing values ------------------------------------------------------------  warfarin_addINR.csv')
print(data0.isnull().sum())     # Checking for missing values in the dataset
print('different values of Race_Reported：', data0.Race_Reported.unique())
print('the number of different values of Race_Reported：', data0.Race_Reported.nunique())
print('different values of Race_OMB：', data0.Race_OMB.unique())
print('the number of different values of Race_OMB：', data0.Race_OMB.nunique())
print('different values of Age：', data0.Age.unique())
print('the number of different values of Age：', data0.Age.nunique())
print('different values of Amiodarone：', data0.Amiodarone.unique())
print('the number of different values of Amiodarone：', data0.Amiodarone.nunique())


data = pd.read_csv('warfarin_noINR_enzyme.csv')
print('View missing values ------------------------------------------------------------  warfarin_noINR_enzyme.csv')
print(data.isnull().sum())
print('\n output the first 5 data -----------------------------------------------------------------\n', data.head())
print(data.describe())
print(data.dtypes)


'''
2.preprocessing
'''
# Age
data['Age'].replace('0 - 9', 4.5, inplace=True)  # age's type: str --> int
data['Age'].replace('10 - 19', 14.5, inplace=True)
data['Age'].replace('20 - 29', 24.5, inplace=True)
data['Age'].replace('30 - 39', 34.5, inplace=True)
data['Age'].replace('40 - 49', 44.5, inplace=True)
data['Age'].replace('50 - 59', 54.5, inplace=True)
data['Age'].replace('60 - 69', 64.5, inplace=True)
data['Age'].replace('70 - 79', 74.5, inplace=True)
data['Age'].replace('80 - 89', 84.5, inplace=True)
data['Age'].replace('90+', 95.0, inplace=True)
print('\n output the first 5 data ----------------------------------------------------------------- transform Age\n', data.head())


# Race
class_le = LabelEncoder()
data['Race'] = class_le.fit_transform(data['Race'].values)
print('\n output the first 5 data -------------------------------------------------------- transform Race\n', data.head())


# Medications
data['Medications'] = data.Medications.astype(str).map(lambda x:1 if "carbamazepine" in x or "phenytoin" in x or "rifampin" in x else 0)
# print(data['Medications'])
print('\n output the first 5 data ------------------------------------------------- transform Medications(finish)\n', data.head())

a = list(data['Medications'])
print('Enzyme inducer status(from feature Medications) is 1:', a.count(1))  # 0：5676； 1：24
print('Enzyme inducer status is 0:', a.count(0))  # 0：5676； 1：24


# delate dose that larger than 315 (extremely value)
print(data.shape[0])
data = data.drop(data[data['Dose'] >= 315].index)
print(data.shape[0])


print('------------------------------------------------- before imputation:\n')
print(data.isnull().sum())
print(data[1243:1251])

data['Amiodarone'] = data['Amiodarone'].fillna(data['Amiodarone'].mode()[0])
data['Height'] = data['Height'].fillna(data['Height'].mean())
data['Weight'] = data['Weight'].fillna(data['Weight'].mean())
data['Dose'] = data['Dose'].fillna(data['Dose'].mean())


#check
# print(data['Height'][455:463])
# print(data['Weight'][453:458])
# print(data['Dose'][3390:3396])
# print(data['Amiodarone'][3137:3141])
print('------------------------------------------------- after imputation:\n')
print(data.isnull().sum())
print(data[1243:1251])


'''
3. dividing X and Y
'''
y = data.Dose
features = ['Race', 'Age', 'Height', 'Weight', 'Medications', 'Amiodarone']
X = data[features]
y = y.to_frame()
print(type(X))  # <class 'pandas.core.frame.DataFrame'>
print(type(y))  # <class 'pandas.core.frame.DataFrame'>

print('\n--------------------------------------------------------------- X.describe and type')
print(X.describe())
# print(X.dtypes)
print('--------------------------------------------------------------- y.describe and type')
print(y.describe())
print(y.dtypes)


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
# print(y_scaled.shape)
y_scaled = y_scaled.reshape(-1)  # (20640, 1) or (1, 20640) --> (20640,)
print(y_scaled.shape)


# check
print('\n check data normalization ----------------------------------------')
print('X_scaled[0:5]:\n', X_scaled[0:5])
print('y_scaled[0:5]:', y_scaled[0:5])
# print(data['Dose'][3390:3396])
# print(y_scaled[3390:3396])
# print(data['Amiodarone'][95:101])


# save data: X_scaled(1338, 6), y_scaled(1338, )
warfarin_afterpreprocess = {}
warfarin_afterpreprocess['X'] = X_scaled
warfarin_afterpreprocess['y'] = y_scaled
np.save(os.getcwd() + '/result_data/warfarin_afterpreprocess.npy', warfarin_afterpreprocess)
print('save warfarin_afterpreprocess.npy done')


loadData = np.load(os.getcwd() + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(X[1240:1250])

trails = 20
for th in range(trails):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    print('\n the division of training and testing data --------------------------------trail:', th)
    print('X_train:', X_train.shape)
    print('y_train:', y_test.shape)
    print('X_train[0]:', X_train[0])
    print('y_train[0]:', y_train[0])


