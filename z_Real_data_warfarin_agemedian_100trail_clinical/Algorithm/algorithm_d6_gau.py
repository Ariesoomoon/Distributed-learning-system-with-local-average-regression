import os, time
from Function_same_block_warfarin import GE_gaussiank, LE_gaussiank_test, clinical_percentage, AE_gaussiank_test, AE_adapt_gaussiank_test, AE_active_gaussiank_test
# from Function_diff_block import generate_data, GE_gaussiank, LE_gaussiank_test, AE_gaussiank_test, AE_adapt_gaussiank_test, AE_active_gaussiank_test
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
loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
warfarin_gau = loadData_gau.tolist()
global_hs = warfarin_gau['global_h']
local_hs = warfarin_gau['local_h']
print(warfarin_gau.keys())


'''1. Initialization'''
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
    GE = GE_gaussiank(X_train, y_train, X_test, y_test, global_h, d, s)[1]
    GE = np.array(GE)
    perc_GE = clinical_percentage(GE, y_test)  # GE means the fit of y_test
    percs_GE[:, th] = np.squeeze(perc_GE)
    print('percs_GE:', percs_GE)

    LE = LE_gaussiank_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d,s)[1]
    LE = np.array(LE)
    perc_LE = clinical_percentage(LE, y_test)  # GE means the fit of y_test
    percs_LE[:, th] = np.squeeze(perc_LE)
    print('percs_LE:', percs_LE)

    AE_log_active = AE_active_gaussiank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, s, local_h)[4]
    AE_log_active = np.array(AE_log_active)
    perc_AE = clinical_percentage(AE_log_active, y_test)  # GE means the fit of y_test
    percs_AE[:, th] = np.squeeze(perc_AE)
    print('percs_AE:', percs_AE)


'''
3. save MSE
'''
warfarin_gau['percs_GE'] = percs_GE
warfarin_gau['percs_LE'] = percs_LE
warfarin_gau['percs_AE'] = percs_AE

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', warfarin_gau)
print('--------------------------------------------------------------save warfarin_gau.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


# check
print(warfarin_gau['percs_AE'])
print(warfarin_gau['percs_AE'].shape)
a = np.mean(warfarin_gau['percs_AE'], axis=1)
print('percs_AE', a)
b = np.mean(warfarin_gau['percs_LE'], axis=1)
print('percs_LE',b)
c = np.mean(warfarin_gau['percs_GE'], axis=1)
print('percs_GE',c)


'''
percs_AE [1.68167618e-03 8.34991950e-01 1.63326374e-01 8.86728227e-01
 6.51310079e-04 1.12620463e-01 2.48229364e-01 2.13233850e-01
 5.38536786e-01]
percs_LE [0.00711865 0.83809774 0.15478362 0.8900913  0.00656195 0.10334675
 0.25893377 0.21812836 0.52293787]
percs_GE [0.00127422 0.83498744 0.16373834 0.90719726 0.00442322 0.08837952
 0.24119009 0.20729335 0.55151656]
'''