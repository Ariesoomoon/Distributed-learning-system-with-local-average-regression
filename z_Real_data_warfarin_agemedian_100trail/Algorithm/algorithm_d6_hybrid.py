import os, time
from Function_same_block_warfarin_epan import generate_data, hybrid_algorithm_test, MSE_pri_TQMA
# from Function_diff_block import generate_data, hybrid_algorithm_test, MSE_pri_TQMA
import numpy as np
from sklearn.model_selection import train_test_split
time_start = time.time()


'''
1. load the data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(warfarin_afterpreprocess.keys())


# load local_h or local_k parameter of each algorithm, and save all of them in trails_d5_hybrid.npy
loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', allow_pickle=True)
loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', allow_pickle=True)
loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData.tolist()

# warfarin_hybrid = {}
# warfarin_hybrid['local_h_epan'] = loadData_epan.tolist()['local_h']
# warfarin_hybrid['local_h_gau'] = loadData_gau.tolist()['local_h']
# warfarin_hybrid['local_h_naive'] = loadData_naive.tolist()['local_h']
# warfarin_hybrid['local_k'] = loadData_knn.tolist()['local_k']
# warfarin_hybrid['AE_lc_epan'] = loadData_epan.tolist()['AE_log_actives']  # AE_lc_epan: AE_log_actives_epan
# warfarin_hybrid['AE_lc_gau'] = loadData_gau.tolist()['AE_log_actives']
# warfarin_hybrid['AE_lc_naive'] = loadData_naive.tolist()['AE_log_actives']
# warfarin_hybrid['AE_lc_k'] = loadData_knn.tolist()['AE_log_actives']
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
# print('--------------------------------------------------------------save warfarin_hybrid.npy done')
print(warfarin_hybrid.keys())
print(loadData_naive.tolist()['AE_log_actives'].shape) # (50, 100)


'''
2. Initialization
'''
d = 6
f = 5
s = 2
trails = 100
fen_num, gap = 50, 2

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData.tolist()
adapt_epan = loadData.tolist()['local_h_epan']
adapt_gau = loadData.tolist()['local_h_gau']
adapt_nai = loadData.tolist()['local_h_naive']
adapt_knn = loadData.tolist()['local_k']
AE_lc_hybrid = np.empty(shape=(50, 100))


'''
3.Calculating the hybrid algorithm
'''
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True

    adapt_epan_1 = adapt_epan[th]
    adapt_gau_1 = adapt_gau[th]
    adapt_nai_1 = adapt_nai[th]
    adapt_knn_1 = adapt_knn[th]

    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    AE = hybrid_algorithm_test(X_train, y_train, X_test, y_test, adapt_gau_1, adapt_epan_1, adapt_nai_1, adapt_knn_1, d, gap, fen_num)
    AE = np.array(AE)
    AE_lc_hybrid[:, th] = np.squeeze(AE)

warfarin_hybrid['AE_lc_hybrid'] = AE_lc_hybrid
np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
print('save warfarin_hybrid.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)




