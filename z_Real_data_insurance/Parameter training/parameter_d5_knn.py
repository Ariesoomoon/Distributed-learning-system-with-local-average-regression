import os, time
from Function_same_block_insurance import generate_data, global_parameter_k_d5, local_parameter_k_d_test
# from Function_diff_block import generate_data, global_parameter_k_d5, local_parameter_k_d_test
import numpy as np
from sklearn.model_selection import train_test_split
time_start = time.time()



'''
1. Initialization
'''
d = 6
f = 5
trails = 20
fen_num, gap = 30, 2 # m = 2,4,...,60
p1, p2 = 5, 15
insurance_knn = {}
global_k = []
local_k = {}

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
    print('\n训练集和测试集的划分（input X和output y） --------------------------------trail:', th)

    gk_trail1 = global_parameter_k_d5(X_train, y_train, d)
    gk_trail1 = int(gk_trail1)
    global_k.append(gk_trail1)
    lk_trail1 = local_parameter_k_d_test(X_train, y_train, p1, p2, d, fen_num, gap)
    local_k[th] = lk_trail1



'''
3. save parameters
'''
insurance_knn['global_k'] = global_k
insurance_knn['local_k'] = local_k
# insurance_knn['local_k_diff'] = local_k
np.save(os.path.dirname(os.getcwd()) + '/Result_data/insurance_knn.npy', insurance_knn)
print('--------------------------------------------------------------save insurance_knn.npy done')
#pdb.set_trace()



# '''4. training parameter on m=2,3,4,5'''
# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_knn.npy', allow_pickle=True)
# insurance_knn = loadData.tolist()
# d = 6
# f = 5
# trails = 20
# fen_num, gap = 5, 1   # m=2,3,4,5
# p1, p2 = 5, 15
# local_k_m5 = {}
#
# for th in range(trails):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=338, random_state=th)  # shuffle:bool, default=True
#     print('\n--------------------------------------------------------------trail:', th)
#     lk_trail1 = local_parameter_k_d_test(X_train, y_train, p1, p2, d, fen_num, gap)
#     local_k_m5[th] = lk_trail1
#
# insurance_knn['local_k_m5'] = local_k_m5
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/insurance_knn.npy', insurance_knn)
# print('--------------------------------------------------------------save insurance_knn.npy done')
# #pdb.set_trace()




