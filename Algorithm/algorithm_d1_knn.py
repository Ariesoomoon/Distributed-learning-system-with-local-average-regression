import os, time
from Function_same_block import generate_data, GE_Knn, LE_Knn_test, AE_Knn_test, AE_adapt_Knn_test, AE_active_Knn_test
# from Function_diff_block import generate_data, GE_Knn, LE_Knn_test, AE_Knn_test, AE_adapt_Knn_test, AE_active_Knn_test
import numpy as np
time_start = time.time()


'''
0. remember to choose the Function_same_block.py or Function_diff_block.py, 
and use the corresponding localization parameters: local_k or local_k_diff
'''


'''
1. Initialization
'''
f = 5
trails = 20
train, test, d = 10000, 1000, 1
fen_num, gap = 70, 5   # blocks num: 5, 10, ..., 350
fen_num_forknn = 48    # d = 1, min(k_opt) in global of 20 trails is 244, so we set fen_num_forknn=48 for LE, AE


# 70 type of divisions, 20trails
GEs = []
AEs = np.empty(shape=(48, 20))  # fen_num_forknn = 48
LEs = np.empty(shape=(48, 20))  # fen_num_forknn = 48
AE_adapts = np.empty(shape=(70, 20))
AE_logs = np.empty(shape=(70, 20))
AE_log_actives = np.empty(shape=(70, 20))

# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_knn.npy', allow_pickle=True)
trails_d1_knn = loadData.tolist()
global_ks = trails_d1_knn['global_k']
local_ks = trails_d1_knn['local_k']
# local_ks = trails_d1_knn['local_k_diff']
print(trails_d1_knn.keys())


'''
2. Calculating different types of MSE for 20 trails by knn
'''
for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    global_k = global_ks[th]    # for th trail, select the corresponding global_k[th]
    local_k = local_ks[th]
    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    # print('global_k:', global_k)
    # print('local_k[trail][0]:', local_k[0])
    # print('--------------------------------------------------------------------------------')

    # (2) calculating
    GE = GE_Knn(X_train, y_train, X_test, y_test, global_k, d)
    GEs.append(GE)
    print('GEs:', GEs)

    LE = LE_Knn_test(X_train, y_train, X_test, y_test, fen_num_forknn, gap, global_k, d)
    AE = AE_Knn_test(X_train, y_train, X_test, y_test, fen_num_forknn, gap, global_k, d)             # AE use the optimal k from global
    AE_adapt = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)         # with k_adapt
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

    ## (4) calculate AE_log separately, remember to change k_adapt to k^log in AE_adapt_knn
    # AE_log = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)  # with k^log
    # AE_log = np.array(AE_log)
    # AE_logs[:, th] = np.squeeze(AE_log)


'''
3. save MSE
'''
trails_d1_knn['GEs'] = GEs
trails_d1_knn['LEs'] = LEs
trails_d1_knn['AEs'] = AEs
trails_d1_knn['AE_adapts'] = AE_adapts
trails_d1_knn['AE_log_actives'] = AE_log_actives
trails_d1_knn['AE_logs'] = AE_logs


np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_knn.npy', trails_d1_knn)
print('--------------------------------------------------------------save trails_d1_knn.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)
