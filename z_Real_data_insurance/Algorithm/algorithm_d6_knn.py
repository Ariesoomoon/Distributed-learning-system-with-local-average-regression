import os, time
from Function_same_block_insurance import generate_data, GE_Knn, LE_Knn_test, AE_Knn_test, AE_adapt_Knn_test, AE_active_Knn_test
# from Function_diff_block import generate_data, GE_Knn, LE_Knn_test, AE_Knn_test, AE_adapt_Knn_test, AE_active_Knn_test
import numpy as np
from sklearn.model_selection import train_test_split
time_start = time.time()


'''
1. Initialization
'''
d = 6
f = 5
s = 2
trails = 20
fen_num, gap = 5, 1 # m = 2,4,...,60

# 30 type of divisions, 20trails, average by axis=1(by trail axis)
# 4: to prove the effective of logarithmic operator, m = 2,3,4,5
GEs = []
AEs = np.empty(shape=(4, 20))
LEs = np.empty(shape=(4, 20))
AE_adapts = np.empty(shape=(4, 20))
AE_logs = np.empty(shape=(4, 20))
# AE_logs = np.empty(shape=(30, 20))
AE_log_actives = np.empty(shape=(30, 20))

# load the data
loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/insurance_afterpreprocess.npy', allow_pickle=True)
insurance_afterpreprocess = loadData.tolist()
X = insurance_afterpreprocess['X']
y = insurance_afterpreprocess['y']
print(insurance_afterpreprocess.keys())

# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_knn.npy', allow_pickle=True)
insurance_knn = loadData.tolist()
global_ks = insurance_knn['global_k']
local_ks = insurance_knn['local_k_m5']
# local_ks = insurance_knn['local_diff_k']
print(insurance_knn.keys())


'''
2. Calculating different types of MSE for 20 trails by KNN
'''
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=338, random_state=th)  # shuffle:bool, default=True
    global_k = global_ks[th]    # for th trail, select the corresponding global_h[th]
    local_k = local_ks[th]

    # (2) calculating
    GE = GE_Knn(X_train, y_train, X_test, y_test, global_k, d)
    GEs.append(GE)
    print('GEs:', GEs)

    LE = LE_Knn_test(X_train, y_train, X_test, y_test, fen_num, gap, global_k, d)
    AE = AE_Knn_test(X_train, y_train, X_test, y_test, fen_num, gap, global_k, d)             # AE use the optimal h from global

    AE_adapt = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)  # with h_adapt
    AE_log_active = AE_active_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)[0]

    # (3) for saving
    LE = np.array(LE)
    LEs[:, th] = np.squeeze(LE)

    AE = np.array(AE)
    AEs[:, th] = np.squeeze(AE)

    AE_adapt = np.array(AE_adapt)
    AE_adapts[:, th] = np.squeeze(AE_adapt)

    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)

    # (4) calculate AE_log separately, remember to change h_adapt to h^log in AE_adapt_gaussiank
    # AE_log = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)  # with h^log
    # AE_log = np.array(AE_log)
    # AE_logs[:, th] = np.squeeze(AE_log)


'''
3. save MSE
'''
insurance_knn['GEs'] = GEs
insurance_knn['LEs'] = LEs
insurance_knn['AEs'] = AEs
insurance_knn['AE_adapts_m5'] = AE_adapts
# insurance_knn['AE_logs_m5'] = AE_logs

# # insurance_knn['AE_logs'] = AE_logs
# # insurance_knn['AE_log_actives'] = AE_log_actives


# np.save(os.path.dirname(os.getcwd()) + '/Result_data/insurance_knn.npy', insurance_knn)
# print('--------------------------------------------------------------save insurance_knn.npy done')
# time_total = time.time() - time_start
# print('runing time:', time_total)
