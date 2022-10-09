import os, time
from Function_same_block_warfarin import generate_data, hybrid_algorithm_test, MSE_pri_TQMA, clinical_percentage, clinical_num
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


# 1.1 load local_h or local_k parameter of each algorithm, and save all of them in trails_d5_hybrid.npy
# loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', allow_pickle=True)
# loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
# loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', allow_pickle=True)
# loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
#
# warfarin_hybrid = {}
# warfarin_hybrid['local_h_epan'] = loadData_epan.tolist()['local_h']
# warfarin_hybrid['local_h_gau'] = loadData_gau.tolist()['local_h']
# warfarin_hybrid['local_h_naive'] = loadData_naive.tolist()['local_h']
# warfarin_hybrid['local_k'] = loadData_knn.tolist()['local_k']
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
# print('--------------------------------------------------------------save warfarin_hybrid.npy done')
# print(warfarin_hybrid.keys())


'''
2. Initialization
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData.tolist()
adapt_epan = loadData.tolist()['local_h_epan']
adapt_gau = loadData.tolist()['local_h_gau']
adapt_nai = loadData.tolist()['local_h_naive']
adapt_knn = loadData.tolist()['local_k']

d = 6
f = 5
s = 2
trails = 100
fen_num, gap = 21, 1  # remember: change 'range(fen_num)' to 'range(fen_num-1, fen_num)'
percs_AE = np.empty(shape=(9, 100))  # nine Comparative Measure according to the clinical value


'''
3. Calculating the percentage
'''
# note: change 'k = ((adapt_knn[fen_num][i]) ** LOG) / m' to 'k = ((adapt_knn[i]) ** LOG) / m'
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True

    adapt_epan_1 = adapt_epan[th][20]  # adapt_epan shape = (20, 1, 21): 20 trails, one 21th, 21 localization parameters
    adapt_gau_1 = adapt_gau[th][20]
    adapt_nai_1 = adapt_nai[th][20]
    adapt_knn_1 = adapt_knn[th][20]

    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    AE = hybrid_algorithm_test(X_train, y_train, X_test, y_test, adapt_gau_1, adapt_epan_1, adapt_nai_1, adapt_knn_1, d, gap, fen_num)[1]
    AE = np.array(AE)
    perc_AE = clinical_percentage(AE, y_test)
    percs_AE[:, th] = np.squeeze(perc_AE)
    print('percs_AE:', percs_AE)
warfarin_hybrid['percs_AE'] = percs_AE
np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
print('save warfarin_hybrid.npy done')


'''
4. calculating the number of testing patients requiring low, intermediate and high dose in 20 trails, then average
'''
num_lows, num_highs, num_intermediates = [], [], []
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    a = clinical_num(y_test)
    num_low, num_high, num_intermediate = a[0], a[1], a[2]
    num_lows.append(num_low)
    num_highs.append(num_high)
    num_intermediates.append(num_intermediate)

warfarin_hybrid['num_lows'] = num_lows
warfarin_hybrid['num_highs'] = num_highs
warfarin_hybrid['num_intermediates'] = num_intermediates
np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', warfarin_hybrid)
print('save warfarin_hybrid.npy done')


