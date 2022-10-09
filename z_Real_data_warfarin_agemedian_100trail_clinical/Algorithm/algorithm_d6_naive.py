import os, time
from Function_same_block_warfarin import generate_data, GE_naivek, LE_naivek_test, clinical_percentage, AE_naivek_test, AE_adapt_naivek_test, AE_active_naivek_test
# from Function_diff_block import generate_data, GE_naivek, LE_naivek_test, AE_naivek_test, AE_adapt_naivek_test, AE_active_naivek_test
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
loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', allow_pickle=True)
warfarin_naive = loadData_naive.tolist()
global_hs = warfarin_naive['global_h']
local_hs = warfarin_naive['local_h']
print(warfarin_naive.keys())


'''
1. Initialization
'''
d = 6
f = 5
s = 2
trails = 100
fen_num, gap = 21, 1  # remember: change 'range(fen_num)' to 'range(fen_num-1, fen_num)'

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
    global_h = global_hs[th]    # for th trail, select the corresponding global_h[th]
    local_h = local_hs[th]

    # (2) calculating
    GE = GE_naivek(X_train, y_train, X_test, y_test, global_h, d)[1]
    GE = np.array(GE)
    perc_GE = clinical_percentage(GE, y_test)  # GE means the fit of y_test
    percs_GE[:, th] = np.squeeze(perc_GE)
    print('percs_GE:', percs_GE)

    LE = LE_naivek_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)[1]
    LE = np.array(LE)
    perc_LE = clinical_percentage(LE, y_test)  # GE means the fit of y_test
    percs_LE[:, th] = np.squeeze(perc_LE)
    print('percs_LE:', percs_LE)

    AE_log_active = AE_active_naivek_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)[1]
    AE_log_active = np.array(AE_log_active)
    perc_AE = clinical_percentage(AE_log_active, y_test)  # GE means the fit of y_test
    percs_AE[:, th] = np.squeeze(perc_AE)
    print('percs_AE:', percs_AE)


'''
3. save MSE
'''
warfarin_naive['percs_GE'] = percs_GE
warfarin_naive['percs_LE'] = percs_LE
warfarin_naive['percs_AE'] = percs_AE

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', warfarin_naive)
print('--------------------------------------------------------------save warfarin_naive.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


# check
print(warfarin_naive['percs_AE'])
print(warfarin_naive['percs_AE'].shape)
a = np.mean(warfarin_naive['percs_AE'], axis=1)
print('percs_AE', a)
b = np.mean(warfarin_naive['percs_LE'], axis=1)
print('percs_LE',b)
c = np.mean(warfarin_naive['percs_GE'], axis=1)
print('percs_GE',c)

'''
percs_AE [0.00125237 0.87138143 0.1273662  0.93686462 0.00228327 0.06085211
 0.22836084 0.20521927 0.56641989]
percs_LE [0.02194744 0.82923348 0.14881909 0.8967982  0.0061628  0.097039
 0.26976663 0.21476644 0.51546693]
percs_GE [0.00233285 0.83726842 0.16039873 0.9011885  0.00411964 0.09469185
 0.24051406 0.20913399 0.55035195]
'''