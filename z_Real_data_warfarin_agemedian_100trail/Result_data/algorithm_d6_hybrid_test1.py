import os, time
import numpy as np


'''检查是否求参数时cv中的函数及时更换（warfarin有一次没有更换！已改正），已更换！'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData.tolist()
adapt_epan = loadData.tolist()['local_h_epan']
adapt_gau = loadData.tolist()['local_h_gau']
adapt_nai = loadData.tolist()['local_h_naive']
adapt_knn = loadData.tolist()['local_k']
print(adapt_epan[0][2])
print('--------------------------------------------------------------')
print(adapt_gau[0][2])
print('--------------------------------------------------------------')
print(adapt_nai[0][2])
print('--------------------------------------------------------------')
print(adapt_knn[0][2])
# wrong
# [0.375      0.515625   0.375      0.50048828 0.40625    0.37890625]
# --------------------------------------------------------------
# [0.375      0.515625   0.375      0.50048828 0.40625    0.37890625]
# --------------------------------------------------------------
# [0.375      0.515625   0.375      0.50048828 0.40625    0.37890625]
# --------------------------------------------------------------
# [38 38 38 56 24 38]

# correct
# [0.375      0.515625   0.375      0.50048828 0.40625    0.37890625]
# --------------------------------------------------------------
# [0.18408203 0.18212891 0.19482422 0.18505859 0.17578125 0.19384766]
# --------------------------------------------------------------
# [0.34765625 0.375      0.328125   0.50048828 0.27197266 0.33886719]
# --------------------------------------------------------------
# [38 38 38 56 24 38]


