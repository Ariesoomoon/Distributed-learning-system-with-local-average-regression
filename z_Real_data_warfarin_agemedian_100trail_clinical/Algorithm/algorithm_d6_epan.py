import os, time
from Function_same_block_warfarin import generate_data, GE_Epank, LE_Epank_test, clinical_percentage, AE_Epank_test, AE_adapt_epank_test, AE_active_epank_test
# from Function_diff_block import generate_data, GE_Epank, LE_Epank_test, AE_Epank_test, AE_adapt_epank_test, AE_active_epank_test
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
loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', allow_pickle=True)
warfarin_epan = loadData_epan.tolist()
global_hs = warfarin_epan['global_hs']
local_hs = warfarin_epan['local_h']
print(warfarin_epan.keys())


'''
1. Initialization
'''
d = 6
f = 5
s = 2
trails = 100
fen_num, gap = 21, 1  # remember: change 'range(fen_num)' to 'range(fen_num-1, fen_num)'
# GEs = np.empty(shape=(1140, 20))
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
    GE = GE_Epank(X_train, y_train, X_test, y_test, global_h, d)[1]
    GE = np.array(GE)
    perc_GE = clinical_percentage(GE, y_test)  # GE means the fit of y_test
    percs_GE[:, th] = np.squeeze(perc_GE)
    print('percs_GE:', percs_GE)


    LE = LE_Epank_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)[1]
    LE = np.array(LE)
    perc_LE = clinical_percentage(LE, y_test)  # GE means the fit of y_test
    percs_LE[:, th] = np.squeeze(perc_LE)
    print('percs_LE:', percs_LE)

    AE_log_active = AE_active_epank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)[4]
    AE_log_active = np.array(AE_log_active)
    perc_AE = clinical_percentage(AE_log_active, y_test)  # GE means the fit of y_test
    percs_AE[:, th] = np.squeeze(perc_AE)
    print('percs_AE:', percs_AE)


'''
3. save MSE
'''
warfarin_epan['percs_GE'] = percs_GE
warfarin_epan['percs_LE'] = percs_LE
warfarin_epan['percs_AE'] = percs_AE

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', warfarin_epan)
print('--------------------------------------------------------------save warfarin_epan.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


# check
print(warfarin_epan['percs_AE'])
print(warfarin_epan['percs_AE'].shape)
a = np.mean(warfarin_epan['percs_AE'], axis=1)
print('percs_AE', a)
b = np.mean(warfarin_epan['percs_LE'], axis=1)
print('percs_LE',b)
c = np.mean(warfarin_epan['percs_GE'], axis=1)
print('percs_GE',c)

'''
percs_AE [0.00141714 0.85219359 0.14638926 0.92391257 0.00235225 0.07373518
 0.2342855  0.20494814 0.56076636]
percs_LE [0.01994304 0.82834494 0.15171202 0.89464655 0.00653276 0.09882068
 0.26735257 0.21674047 0.51590697]
percs_GE [0.00249714 0.83794069 0.15956217 0.90286334 0.00419643 0.09294023
 0.24172098 0.20909761 0.54918141]
'''