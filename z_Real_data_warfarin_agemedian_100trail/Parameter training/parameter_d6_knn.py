import os, time
from Function_same_block_warfarin_epan import generate_data, global_parameter_k_d5, local_parameter_k_d_test
# from Function_diff_block import generate_data, global_parameter_k_d5, local_parameter_k_d_test
import numpy as np
from sklearn.model_selection import train_test_split
time_start = time.time()


'''
1. Initialization
'''
d = 6
f = 5
trails = 100
fen_num, gap = 50, 2 # m = 2,4,...,100
p1, p2 = 30, 90
warfarin_knn = {}
global_k = []
local_k = {}

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
    print('\n训练集和测试集的划分（input X和output y） --------------------------------trail:', th)

    gk_trail1 = global_parameter_k_d5(X_train, y_train, d)
    gk_trail1 = int(gk_trail1)
    global_k.append(gk_trail1)
    print(global_k)

    lk_trail1 = local_parameter_k_d_test(X_train, y_train, p1, p2, d, fen_num, gap)
    local_k[th] = lk_trail1
    print(local_k)


'''
3. save parameters
'''
warfarin_knn['global_k'] = global_k
warfarin_knn['local_k'] = local_k
# warfarin_knn['local_k_diff'] = local_k
np.save(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', warfarin_knn)
print('--------------------------------------------------------------save warfarin_knn.npy done')
#pdb.set_trace()





