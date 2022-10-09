import os, time
from Function_same_block_insurance import generate_data, GE_naivek, LE_naivek_test, AE_naivek_test, AE_adapt_naivek_test, AE_active_naivek_test
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
trails = 20
fen_num, gap = 30, 2 # m = 2,4,...,60

# 30 type of divisions, 20trails, average by axis=1(by trail axis)
GEs = []
AEs = np.empty(shape=(30, 20))
LEs = np.empty(shape=(30, 20))
AE_adapts = np.empty(shape=(30, 20))
AE_logs = np.empty(shape=(30, 20))
AE_log_actives = np.empty(shape=(30, 20))

# load the data
loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/insurance_afterpreprocess.npy', allow_pickle=True)
insurance_afterpreprocess = loadData.tolist()
X = insurance_afterpreprocess['X']
y = insurance_afterpreprocess['y']
print(insurance_afterpreprocess.keys())

# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_naive.npy', allow_pickle=True)
insurance_naive = loadData.tolist()
global_hs = insurance_naive['global_h']
local_hs = insurance_naive['local_h']
# local_hs = insurance_naive['local_diff_h']
print(insurance_naive.keys())


'''
2. Calculating different types of MSE for 20 trails by NWK(naive)
'''
for th in range(trails):
    # (1) create data and load parameter
    print('                                                                                  trail:', th + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=338, random_state=th)  # shuffle:bool, default=True
    global_h = global_hs[th]    # for th trail, select the corresponding global_h[th]
    local_h = local_hs[th]

    # (2) calculating
    GE = GE_naivek(X_train, y_train, X_test, y_test, global_h, d)
    GEs.append(GE)
    print('GEs:', GEs)

    LE = LE_naivek_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)
    AE = AE_naivek_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)             # AE use the optimal h from global

    AE_adapt = AE_adapt_naivek_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)  # with h_adapt
    AE_log_active = AE_active_naivek_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)

    # (3) for saving
    LE = np.array(LE)
    LEs[:, th] = np.squeeze(LE)

    AE = np.array(AE)
    AEs[:, th] = np.squeeze(AE)

    AE_adapt = np.array(AE_adapt)
    AE_adapts[:, th] = np.squeeze(AE_adapt)

    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)

    ## (4) calculate AE_log separately, remember to change h_adapt to h^log in AE_adapt_gaussiank
    # AE_log = AE_adapt_naivek_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)  # with h^log
    # AE_log = np.array(AE_log)
    # AE_logs[:, th] = np.squeeze(AE_log)


'''
3. save MSE
'''
insurance_naive['GEs'] = GEs
insurance_naive['LEs'] = LEs
insurance_naive['AEs'] = AEs
insurance_naive['AE_adapts'] = AE_adapts
insurance_naive['AE_log_actives'] = AE_log_actives
# insurance_naive['AE_logs'] = AE_logs

# np.save(os.path.dirname(os.getcwd()) + '/Result_data/insurance_naive.npy', insurance_naive)
# print('--------------------------------------------------------------save insurance_naive.npy done')
# time_total = time.time() - time_start
# print('runing time:', time_total)


