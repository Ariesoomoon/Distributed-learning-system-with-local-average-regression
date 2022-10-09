import os, time
import numpy as np
from Function_same_block_warfarin_epan import clinical_percentage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


'''
1. Initialization
'''
# load the data
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData.tolist()

loadData_war = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData_war.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(warfarin_afterpreprocess.keys())


# other algorithm
LR = linear_model.LinearRegression()
# RFR = RandomForestRegressor(n_estimators = 100)
# Tree = DecisionTreeRegressor()
# Knn = KNeighborsRegressor()
# Ada = AdaBoostRegressor()
# Gbr = GradientBoostingRegressor()

trails = 100
MSE_lr, MSE_rfr, MSE_tree, MSE_knn, MSE_ada, MSE_gbr = [], [], [], [], [], []
percs_lr = np.empty(shape=(9, 100))


'''
2. calculating
'''
for th in range(trails):
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    # linear regression
    LR.fit(X_train,y_train)
    fit_lr = LR.predict(X_test)
    # print(fit_lr)
    print(fit_lr.shape)
    perc_lr = clinical_percentage(fit_lr, y_test)  # GE means the fit of y_test
    percs_lr[:, th] = np.squeeze(perc_lr)
    print('percs_lr:', percs_lr)

    # mse_lr = mean_squared_error(y_test, LR.predict(X_test))
    # MSE_lr.append(mse_lr)

    # Random Forest Regressor
    # RFR.fit(X_train, y_train)
    # mse_rfr = mean_squared_error(y_test, RFR.predict(X_test))
    # MSE_rfr.append(mse_rfr)

    # Decision tree
    # Tree.fit(X_train, y_train)
    # mse_tree = mean_squared_error(y_test, Tree.predict(X_test))
    # MSE_tree.append(mse_tree)

    # KNeighbors Regressor
    # Knn.fit(X_train, y_train)
    # mse_knn = mean_squared_error(y_test, Knn.predict(X_test))
    # MSE_knn.append(mse_knn)

    # AdaBoost Regressor
    # Ada.fit(X_train, y_train)
    # mse_ada = mean_squared_error(y_test, Ada.predict(X_test))
    # MSE_ada.append(mse_ada)

    # Gradient Boosting Regressor
    # Gbr.fit(X_train, y_train)
    # mse_gbr = mean_squared_error(y_test, Gbr.predict(X_test))
    # MSE_gbr.append(mse_gbr)

warfarin_hybrid['percs_lr'] = percs_lr
# warfarin_hybrid['MSE_lr'] = MSE_lr
# warfarin_hybrid['MSE_rfr'] = MSE_rfr
# warfarin_hybrid['MSE_tree'] = MSE_tree
# warfarin_hybrid['MSE_knn'] = MSE_knn
# warfarin_hybrid['MSE_ada'] = MSE_ada
# warfarin_hybrid['MSE_gbr'] = MSE_gbr

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
print('--------------------------------------------------------------save warfarin_hybrid.npy done')



# check
# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
# warfarin_hybrid = loadData.tolist()
# print(warfarin_hybrid['percs_lr'])
# print('--------')
# percs_GE_lr = np.mean(warfarin_hybrid['percs_lr'], axis=1)
# print('percs_GE_lr:', percs_GE_lr)

# [[0.01851852 0.00557103 0.02253521 0.024      0.01336898 0.01149425
#   0.01312336 0.03089888 0.0188172  0.00539084 0.00773196 0.02011494
#   0.01436782 0.00512821 0.01329787 0.01626016 0.00777202 0.00795756
#   0.02279202 0.00828729]
#  [0.85714286 0.89415042 0.8        0.83466667 0.83957219 0.83333333
#   0.8503937  0.82022472 0.82258065 0.83557951 0.84793814 0.82758621
#   0.82183908 0.87948718 0.84574468 0.83197832 0.85751295 0.84880637
#   0.83475783 0.84254144]

# percs_GE_lr: [0.01437141 0.84129181 0.14433678 0.87119002 0.         0.12880998
#  0.25226311 0.21814326 0.52959363]

