import os, time
import numpy as np
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

loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(warfarin_afterpreprocess.keys())

# other algorithm
LR = linear_model.LinearRegression()
RFR = RandomForestRegressor(n_estimators = 100)
Tree = DecisionTreeRegressor()
Knn = KNeighborsRegressor()
Ada = AdaBoostRegressor()
Gbr = GradientBoostingRegressor()

trails = 100
MSE_lr, MSE_rfr, MSE_tree, MSE_knn, MSE_ada, MSE_gbr = [], [], [], [], [], []


'''
2. calculating
'''
for th in range(trails):
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    # linear regression
    LR.fit(X_train,y_train)
    mse_lr = mean_squared_error(y_test, LR.predict(X_test))
    MSE_lr.append(mse_lr)

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

warfarin_hybrid['MSE_lr'] = MSE_lr
# warfarin_hybrid['MSE_rfr'] = MSE_rfr
# warfarin_hybrid['MSE_tree'] = MSE_tree
# warfarin_hybrid['MSE_knn'] = MSE_knn
# warfarin_hybrid['MSE_ada'] = MSE_ada
# warfarin_hybrid['MSE_gbr'] = MSE_gbr

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
print('--------------------------------------------------------------save warfarin_hybrid.npy done')



