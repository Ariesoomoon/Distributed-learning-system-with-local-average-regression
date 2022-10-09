import os, time
from Function_same_block_insurance import generate_data, global_parameter_h, local_parameter_h_test
# from Function_diff_block import generate_data, global_parameter_h, local_parameter_h_test
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
insurance_gau = {}
global_h = []
local_h = {}

loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/insurance_afterpreprocess.npy', allow_pickle=True)
insurance_afterpreprocess = loadData.tolist()
X = insurance_afterpreprocess['X']
y = insurance_afterpreprocess['y']
print(insurance_afterpreprocess.keys())


'''
2. training parameter: 20 random seed for 20 trails
'''
for th in range(trails):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=338, random_state=th)  # shuffle:bool, default=True
    print('\n--------------------------------------------------------------trail:', th)

    gh_trail1 = global_parameter_h(X_train, y_train, f, d)
    global_h.append(gh_trail1)
    lh_trail1 = local_parameter_h_test(X_train, y_train, f, d, fen_num, gap)
    local_h[th] = lh_trail1


'''
3. save parameters
'''
insurance_gau['global_h'] = global_h
insurance_gau['local_h'] = local_h
# # insurance_epan['local_h_diff'] = local_h
np.save(os.path.dirname(os.getcwd()) + '/Result_data/insurance_gau.npy', insurance_gau)
print('--------------------------------------------------------------save insurance_gau.npy done')
#pdb.set_trace()
