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
train, test, d = (10000, 5), (1000, 5), 5
fen_num, fen_num_forknn = 70, 20   # d = 5, min(k_opt) in global of 20 trails is , so we set fen_num_forknn=20 for LE, AE
gap, gap_forknn = 5, 1


# 20trails, average by axis=1
GEs = []
# to prove logarithmic operator is effective
AEs_m20 = np.empty(shape=(19, 20))
LEs_m20 = np.empty(shape=(19, 20))
AE_adapts_m20 = np.empty(shape=(19, 20))
AE_logs_m20 = np.empty(shape=(19, 20))

# to prove active rule is effective
AE_logs = np.empty(shape=(70, 20))
AE_log_actives = np.empty(shape=(70, 20))

# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_knn.npy', allow_pickle=True)
trails_d5_knn = loadData.tolist()
global_ks = trails_d5_knn['global_k']
local_ks = trails_d5_knn['local_k']
local_k_m20s = trails_d5_knn['local_k_m20']
print(trails_d5_knn.keys())



'''
2. Calculating different types of MSE for 20 trails by knn
'''
for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    global_k = global_ks[th]  # for th trail, select the corresponding global_k[th]
    local_k = local_ks[th]
    local_k_m20 = local_k_m20s[th]
    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    # print('global_k:', global_k)
    # print('local_k[trail][0]:', local_k[0])
    # print('--------------------------------------------------------------------------------')

    # (2) calculating
    GE = GE_Knn(X_train, y_train, X_test, y_test, global_k, d)
    GEs.append(GE)
    print('GEs:', GEs)

    LE_m20 = LE_Knn_test(X_train, y_train, X_test, y_test, fen_num_forknn, gap_forknn, global_k, d)
    AE_m20 = AE_Knn_test(X_train, y_train, X_test, y_test, fen_num_forknn, gap_forknn, global_k, d)                   # AE use the optimal k from global
    AE_adapt_m20 = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num_forknn, gap_forknn, local_k_m20)    # with k_adapt
    AE_log_active = AE_active_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)[0]

    LE_m20 = np.array(LE_m20)
    LEs_m20[:, th] = np.squeeze(LE_m20)

    AE_m20 = np.array(AE_m20)
    AEs_m20[:, th] = np.squeeze(AE_m20)

    AE_adapt_m20 = np.array(AE_adapt_m20)
    AE_adapts_m20[:, th] = np.squeeze(AE_adapt_m20)

    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)

    ## (4) calculate AE_log_m20 and AE_logs separately, remember to change k_adapt to k^log in AE_adapt_Knn
    # AE_log_m20 = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num_forknn, gap_forknn, local_k_m20)  # with k^log
    # AE_log = AE_adapt_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)      # with k^log
    #
    # AE_log_m20 = np.array(AE_log_m20)
    # AE_logs_m20[:, th] = np.squeeze(AE_log_m20)
    #
    # AE_log = np.array(AE_log)
    # AE_logs[:, th] = np.squeeze(AE_log)


'''
3. save MSE
'''
trails_d5_knn['GEs'] = GEs
trails_d5_knn['LEs_m20'] = LEs_m20
trails_d5_knn['AEs_m20'] = AEs_m20
trails_d5_knn['AE_adapts_m20'] = AE_adapts_m20
trails_d5_knn['AE_log_actives'] = AE_log_actives

trails_d5_knn['AE_logs_m20'] = AE_logs_m20
trails_d5_knn['AE_logs'] = AE_logs


np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_knn.npy', trails_d5_knn)
print('--------------------------------------------------------------save trails_d5_knn.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)
