import os, time
from Function_same_block_warfarin_epan import generate_data, global_parameter_h, local_parameter_h_test
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
trails = 100
fen_num, gap = 50, 2 # m = 2,4,...,100
warfarin_epan = {}
global_h = []
local_h = {}

loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
warfarin_afterpreprocess = loadData.tolist()
X = warfarin_afterpreprocess['X']
y = warfarin_afterpreprocess['y']
print(warfarin_afterpreprocess.keys())


'''
2. training parameter: 20 random seed for 20 trails
'''
for th in range(trails):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=th)  # shuffle:bool, default=True
    print('\n--------------------------------------------------------------trail:', th)

    gh_trail1 = global_parameter_h(X_train, y_train, f, d)
    global_h.append(gh_trail1)
    print(global_h)
    lh_trail1 = local_parameter_h_test(X_train, y_train, f, d, fen_num, gap)
    local_h[th] = lh_trail1


'''
3. save parameters
'''
warfarin_epan['global_h'] = global_h
warfarin_epan['local_h'] = local_h
# warfarin_epan['local_h_diff'] = local_h
np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', warfarin_epan)
print('--------------------------------------------------------------save warfarin_epan.npy done')
#pdb.set_trace()
