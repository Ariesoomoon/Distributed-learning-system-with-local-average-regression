import os, time
from Function_same_block import generate_data, global_parameter_k_d5, local_parameter_k_d_test
#from Function_diff_block import generate_data, global_parameter_k_d5, local_parameter_k_d_test
import numpy as np
time_start = time.time()


'''
1. Initialization
'''
f = 5                  # cv-5
trails = 20
train, test, d = (10000,5), (1000,5), 5
fen_num, gap = 70, 5   # blocks num: 5, 10, ..., 350, because AE_opt and LE, we need design the blocks number not to exceed the optimal k in global
p1, p2 = 20, 60      # for tarining parameter k
# trails_d5_knn = {}
global_k = []
local_k = {}

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_knn.npy', allow_pickle=True)
trails_d5_knn = loadData.tolist()
print(trails_d5_knn.keys())


'''
2. training parameter: 20 random seed for 20 trails
'''
for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    print('\nThe first training data:\nX_train:%s,y_train:%s' % (X_train[0:1], y_train[0:1]))
    gk_trail1 = global_parameter_k_d5(X_train, y_train, d)
    # gk_trail1 = int(gk_trail1)
    # global_k.append(gk_trail1)
    # print('global_k:', global_k)
    lk_trail1 = local_parameter_k_d_test(X_train, y_train, p1, p2, d, fen_num, gap)
    local_k[trail] = lk_trail1
    print('local_k:', local_k)


'''
3. save parameters
'''
# trails_d5_knn['global_k'] = global_k
# trails_d5_knn['local_k'] = local_k
trails_d5_knn['local_k_diff'] = local_k

np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_knn.npy', trails_d5_knn)
print('--------------------------------------------------------------save trails_d5_knn.npy done')

