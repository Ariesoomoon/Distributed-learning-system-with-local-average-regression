import os, time
from Function_same_block_warfarin_epan import generate_data, GE_naivek, LE_naivek_test, AE_naivek_test, AE_adapt_naivek_test, AE_active_naivek_test
# from Function_diff_block import generate_data, GE_naivek, LE_naivek_test, AE_naivek_test, AE_adapt_naivek_test, AE_active_naivek_test
import numpy as np
from sklearn.model_selection import train_test_split
time_start = time.time()


'''
1. Initialization
'''
d = 6
f = 5
s = 2
trails = 100
fen_num, gap = 50, 2 # m = 2,4,...,60

# 50 type of divisions, 20trails, average by axis=1(by trail axis)
GEs = []
LEs = np.empty(shape=(50, 100))
AE_log_actives = np.empty(shape=(50, 100))

# load the data
loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(warfarin_afterpreprocess.keys())

# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', allow_pickle=True)
warfarin_naive = loadData.tolist()
global_hs = warfarin_naive['global_h']
local_hs = warfarin_naive['local_h']
print(warfarin_naive.keys())


'''
2. Calculating different types of MSE for 20 trails by NWK(naive)
'''
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    global_h = global_hs[th]    # for th trail, select the corresponding global_h[th]
    local_h = local_hs[th]

    # (2) calculating
    GE = GE_naivek(X_train, y_train, X_test, y_test, global_h, d)
    GEs.append(GE)
    print('GEs:', GEs)

    LE = LE_naivek_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)
    LE = np.array(LE)
    LEs[:, th] = np.squeeze(LE)

    AE_log_active = AE_active_naivek_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)
    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)


'''
3. save MSE
'''
warfarin_naive['GEs'] = GEs
warfarin_naive['LEs'] = LEs
warfarin_naive['AE_log_actives'] = AE_log_actives

np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', warfarin_naive)
print('--------------------------------------------------------------save warfarin_naive.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


