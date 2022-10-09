import os, time
# from Function_same_block import generate_data, hybrid_algorithm_test, MSE_pri_TQMA, TQMA, RL_measure
from Function_diff_block import generate_data, hybrid_algorithm_test, MSE_pri_TQMA, LE_hybrid_algorithm_test
import numpy as np
time_start = time.time()



'''
1. load local_h or local_k parameter of each algorithm, and save all of them in trails_d5_hybrid.npy
'''
# loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_epan.npy', allow_pickle=True)
# loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_gau2.npy', allow_pickle=True)
# loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_naive.npy', allow_pickle=True)
# loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_knn.npy', allow_pickle=True)
#
# trails_d5_hybrid = {}
# trails_d5_hybrid['local_h_epan'] = loadData_epan.tolist()['local_h']
# trails_d5_hybrid['local_h_gau'] = loadData_gau.tolist()['local_h']
# trails_d5_hybrid['local_h_naive'] = loadData_naive.tolist()['local_h']
# trails_d5_hybrid['local_k'] = loadData_knn.tolist()['local_k']
# trails_d5_hybrid['AE_lc_epan'] = loadData_epan.tolist()['AE_log_actives']  # AE_lc_epan: AE_log_actives_epan
# trails_d5_hybrid['AE_lc_gau'] = loadData_gau.tolist()['AE_log_actives']
# trails_d5_hybrid['AE_lc_naive'] = loadData_naive.tolist()['AE_log_actives']
# trails_d5_hybrid['AE_lc_k'] = loadData_knn.tolist()['AE_log_actives']
#
# print(trails_d5_hybrid.keys())
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_hybrid.npy', trails_d5_hybrid)
# print('--------------------------------------------------------------save trails_d5_hybrid.npy done')



'''
2.1 Calculating the hybrid algorithm in the equal-sized setting 
'''
f = 5
trails = 20
train, test, d = (10000, 5), (1000, 5), 5
fen_num, gap = 70, 5   # blocks num: 5, 10, ..., 350

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_hybrid.npy', allow_pickle=True)
trails_d5_hybrid = loadData.tolist()
adapt_epan = loadData.tolist()['local_h_epan']
adapt_gau = loadData.tolist()['local_h_gau']
adapt_nai = loadData.tolist()['local_h_naive']
adapt_knn = loadData.tolist()['local_k']
AE_lc_hybrid = np.empty(shape=(70, 20))


for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    adapt_epan_1 = adapt_epan[th]
    adapt_gau_1 = adapt_gau[th]
    adapt_nai_1 = adapt_nai[th]
    adapt_knn_1 = adapt_knn[th]
    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    AE = hybrid_algorithm_test(X_train, y_train, X_test, y_test, adapt_gau_1, adapt_epan_1, adapt_nai_1, adapt_knn_1, d, gap, fen_num)
    AE = np.array(AE)
    AE_lc_hybrid[:, th] = np.squeeze(AE)

trails_d5_hybrid['AE_lc_hybrid'] = AE_lc_hybrid
np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_hybrid.npy', trails_d5_hybrid)
print('save trails_d5_hybrid.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)



'''
2.2 Calculating the hybrid algorithm in the unequal-sized setting 
'''
# f = 5
# trails = 20
# train, test, d = (10000, 5), (1000, 5), 5
# fen_num, gap = 70, 5   # blocks num: 5, 10, ..., 350
# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_hybrid.npy', allow_pickle=True)
# trails_d5_hybrid = loadData.tolist()
# adapt_epan = loadData.tolist()['local_h_diff_epan']
# adapt_gau = loadData.tolist()['local_h_diff_gau']
# adapt_nai = loadData.tolist()['local_h_diff_naive']
# adapt_knn = loadData.tolist()['local_k_diff_knn']
# AE_lc_hybrid = np.empty(shape=(70, 20))
# LE_hybrid = np.empty(shape=(70, 20))
#
#
# for th in range(trails):
#     np.random.seed(th)
#     print('                                                                                  trail:', th + 1)
#     # (1) create data and load parameter
#     X_train, y_train, X_test, y_test = generate_data(train, test, d)
#     adapt_epan_1 = adapt_epan[th]
#     adapt_gau_1 = adapt_gau[th]
#     adapt_nai_1 = adapt_nai[th]
#     adapt_knn_1 = adapt_knn[th]
#     print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
#     AE = hybrid_algorithm_test(X_train, y_train, X_test, y_test, adapt_gau_1, adapt_epan_1, adapt_nai_1, adapt_knn_1, d, gap, fen_num)
#     AE = np.array(AE)
#     AE_lc_hybrid[:, th] = np.squeeze(AE)
#     LE = LE_hybrid_algorithm_test(X_train, y_train, X_test, y_test, adapt_gau_1, adapt_epan_1, adapt_nai_1, adapt_knn_1, d, gap, fen_num)
#     LE = np.array(LE)
#     LE_hybrid[:, th] = np.squeeze(LE)
#
# trails_d5_hybrid['AE_lc_hybrid_diff'] = AE_lc_hybrid
# trails_d5_hybrid['LE_hybrid_diff'] = LE_hybrid
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_hybrid.npy', trails_d5_hybrid)
# print('save trails_d5_hybrid.npy done')
# time_total = time.time() - time_start
# print('runing time:', time_total)
#
#
