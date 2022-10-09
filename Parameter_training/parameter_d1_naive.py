import os, time
#from Function_same_block import generate_data, global_parameter_h, local_parameter_h_test
from Function_diff_block import generate_data, global_parameter_h, local_parameter_h_test
import numpy as np
time_start = time.time()



'''
1. Initialization
'''
f = 5                  # cv-5
trails = 20
train, test, d = 10000, 1000, 1
fen_num, gap = 70, 5   # blocks num: 5, 10, ..., 350
p1, p2 = 0.25, 0.75    # for tarining parameter h
s = 2                  # for gaussian kernel, s = 1,2,3, here just choose s = 2
# trails_d1_naive = {}
# global_h = []
local_h = {}

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_naive.npy', allow_pickle=True)
trails_d1_naive = loadData.tolist()
print(trails_d1_naive.keys())


'''
2. training parameter: 20 random seed for 20 trails
'''
for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    print('\nThe first training data:\nX_train:%s,y_train:%s' % (X_train[0:1], y_train[0:1]))
    # gh_trail1 = global_parameter_h(X_train, y_train, f, d)
    # gh_trail1 = int(gh_trail1)
    # global_h.append(gh_trail1)
    # print('global_h:', global_h)

    lh_trail1 = local_parameter_h_test(X_train, y_train, f, d, fen_num, gap)
    local_h[trail] = lh_trail1
    print('local_h:', local_h)


'''
3. save parameters
'''
# trails_d1_naive['global_h'] = global_h
# trails_d1_naive['local_h'] = local_h
trails_d1_naive['local_h_diff'] = local_h
np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_naive.npy', trails_d1_naive)
print('--------------------------------------------------------------save trails_d1_naive.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)

