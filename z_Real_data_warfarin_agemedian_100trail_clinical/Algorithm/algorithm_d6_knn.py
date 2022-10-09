import os, time
from Function_same_block_warfarin import generate_data, GE_Knn, LE_Knn_test, clinical_percentage, AE_Knn_test, AE_adapt_Knn_test, AE_active_Knn_test
# from Function_diff_block import generate_data, GE_Knn, LE_Knn_test, AE_Knn_test, AE_adapt_Knn_test, AE_active_Knn_test
import numpy as np
from sklearn.model_selection import train_test_split
time_start = time.time()


# load the data
loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(warfarin_afterpreprocess.keys())

# load the localization parameters
loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
warfarin_knn = loadData_knn.tolist()
global_ks = warfarin_knn['global_k']
local_ks = warfarin_knn['local_k']
print(warfarin_knn.keys())


'''
1. Initialization
'''
d = 6
f = 5
s = 2
trails = 100
fen_num, gap = 21, 1  # remember: change 'range(fen_num)' to 'range(fen_num-1, fen_num)'
fen_num_knn = 21
# GEs = np.empty(shape=(1140, 20))
# LEs = np.empty(shape=(1140, 20))
# AE_log_actives = np.empty(shape=(1140, 20))
percs_GE = np.empty(shape=(9, 100))
percs_LE = np.empty(shape=(9, 100))
percs_AE = np.empty(shape=(9, 100))


'''
2. calculating the MSE for 20 trails
'''
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    global_k = global_ks[th]    # for th trail, select the corresponding global_h[th]
    local_k = local_ks[th]

    # (2) calculating
    GE = GE_Knn(X_train, y_train, X_test, y_test, global_k, d)[1]
    GE = np.array(GE)
    perc_GE = clinical_percentage(GE, y_test)  # GE means the fit of y_test
    percs_GE[:, th] = np.squeeze(perc_GE)
    print('percs_GE:', percs_GE)

    LE = LE_Knn_test(X_train, y_train, X_test, y_test, fen_num_knn, gap, global_k, d)[1]
    LE = np.array(LE)
    perc_LE = clinical_percentage(LE, y_test)  # GE means the fit of y_test
    percs_LE[:, th] = np.squeeze(perc_LE)
    print('percs_LE:', percs_LE)

    AE_log_active = AE_active_Knn_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_k)[2]
    AE_log_active = np.array(AE_log_active)
    perc_AE = clinical_percentage(AE_log_active, y_test)  # GE means the fit of y_test
    percs_AE[:, th] = np.squeeze(perc_AE)
    print('percs_AE:', percs_AE)


'''
3. save MSE
'''
warfarin_knn['percs_GE'] = percs_GE
warfarin_knn['percs_LE'] = percs_LE
warfarin_knn['percs_AE'] = percs_AE

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', warfarin_knn)
print('--------------------------------------------------------------save warfarin_knn.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


# check
print(warfarin_knn['percs_AE'])
print(warfarin_knn['percs_AE'].shape)
a = np.mean(warfarin_knn['percs_AE'], axis=1)
print('percs_AE', a)
b = np.mean(warfarin_knn['percs_LE'], axis=1)
print('percs_LE',b)
c = np.mean(warfarin_knn['percs_GE'], axis=1)
print('percs_GE',c)

'''
percs_AE [0.00187844 0.83442205 0.16369951 0.89116096 0.         0.10883904
 0.24870534 0.21527363 0.53602102]
percs_LE [0.04069871 0.78106515 0.17823614 0.81934408 0.01001792 0.170638
 0.32016059 0.26099556 0.41884386]
percs_GE [1.56678797e-03 8.31680945e-01 1.66752267e-01 8.79501848e-01
 7.14285714e-05 1.20426723e-01 2.51680316e-01 2.20361116e-01
 5.27958568e-01]
'''

