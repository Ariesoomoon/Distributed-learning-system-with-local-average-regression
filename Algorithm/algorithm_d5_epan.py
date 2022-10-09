import os, time
from Function_same_block import generate_data, GE_Epank, LE_Epank_test, AE_Epank_test, AE_adapt_epank_test, AE_active_epank_test
# from Function_diff_block import generate_data, GE_Epank, LE_Epank_test, AE_Epank_test, AE_adapt_epank_test, AE_active_epank_test
import numpy as np
time_start = time.time()



'''
0. remember to choose the Function_same_block.py or Function_diff_block.py, 
and use the corresponding localization parameters: local_h or local_diff_h 
'''


'''
1. Initialization
'''
f = 5
trails = 20
train, test, d = (10000, 5), (1000, 5), 5
fen_num, gap = 70, 5
s = 2


# 70 type of divisions, 20trails
GEs = []
AEs = np.empty(shape=(70, 20))
LEs = np.empty(shape=(70, 20))
AE_adapts = np.empty(shape=(70, 20))
AE_logs = np.empty(shape=(70, 20))
AE_log_actives = np.empty(shape=(70, 20))

# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_epan.npy', allow_pickle=True)
trails_d5_epan = loadData.tolist()
global_hs = trails_d5_epan['global_h']
local_hs = trails_d5_epan['local_h']
# local_hs = trails_d5_epan['local_diff_h']
print(trails_d5_epan.keys())


'''
2. Calculating different types of MSE for 20 trails by NWK(Epanechnikov)
'''
for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    global_h = global_hs[th]    # for th trail, select the corresponding global_h[th]
    local_h = local_hs[th]
    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    # print('global_h:', global_h)
    # print('local_h[trail][0]:', local_h[0])
    # print('--------------------------------------------------------------------------------')

    # (2) calculating
    GE = GE_Epank(X_train, y_train, X_test, y_test, global_h, d)
    GEs.append(GE)
    print('GEs:', GEs)

    LE = LE_Epank_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)
    AE = AE_Epank_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d)               # AE use the optimal h from global
    AE_adapt = AE_adapt_epank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)    # with h_adapt
    AE_log_active = AE_active_epank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)[0]

    # (3) for saving
    LE = np.array(LE)
    LEs[:, th] = np.squeeze(LE)

    AE = np.array(AE)
    AEs[:, th] = np.squeeze(AE)

    AE_adapt = np.array(AE_adapt)
    AE_adapts[:, th] = np.squeeze(AE_adapt)

    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)

    ## (4) calculate AE_log separately, remember to change h_adapt to h^log in AE_adapt_epank
    # AE_log = AE_adapt_epank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, local_h)  # with h^log
    # AE_log = np.array(AE_log)
    # AE_logs[:, th] = np.squeeze(AE_log)


'''
3. save MSE
'''
trails_d5_epan['GEs'] = GEs
trails_d5_epan['LEs'] = LEs
trails_d5_epan['AEs'] = AEs
trails_d5_epan['AE_adapts'] = AE_adapts
trails_d5_epan['AE_log_actives'] = AE_log_actives
# trails_d5_epan['AE_logs'] = AE_logs

np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_epan.npy', trails_d5_epan)
print('--------------------------------------------------------------save trails_d5_epan.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)
