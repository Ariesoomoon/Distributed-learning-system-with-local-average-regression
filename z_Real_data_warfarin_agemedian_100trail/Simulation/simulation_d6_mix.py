import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os
time_start = time.time()


'''
1. load data
'''
loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
warfarin_gau = loadData_gau.tolist()
MSE_gau_mean = [np.mean(warfarin_gau['GEs'])] * 50
LE_gau_mean = np.mean(warfarin_gau['LEs'], axis=1)

loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
warfarin_knn = loadData_knn.tolist()
MSE_knn_mean = [np.mean(warfarin_knn['GEs'])] * 18
LE_knn_mean = np.mean(warfarin_knn['LEs'], axis=1)
# RMSE
# MSE_gau_mean = np.array(MSE_gau_mean)**0.5
# LE_gau_mean = LE_gau_mean ** 0.5

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData.tolist()
AE_lc_epan_mean = np.mean(loadData.tolist()['AE_lc_epan'], axis=1)
AE_lc_gau_mean = np.mean(loadData.tolist()['AE_lc_gau'], axis=1)
AE_lc_naive_mean = np.mean(loadData.tolist()['AE_lc_naive'], axis=1)
AE_lc_knn_mean = np.mean(loadData.tolist()['AE_lc_k'], axis=1)
AE_lc_hybrid_mean = np.mean(loadData.tolist()['AE_lc_hybrid'], axis=1)
MSE_lr_mean = [np.mean(warfarin_hybrid['MSE_lr'])] * 50      # linear regression
print(warfarin_hybrid.keys())
print(MSE_lr_mean)
# RMSE
# MSE_lr_mean = np.array(MSE_lr_mean)**0.5
# AE_lc_knn_mean = AE_lc_knn_mean ** 0.5
# AE_lc_hybrid_mean = AE_lc_hybrid_mean ** 0.5


'''
figure 1: compare the AE_log_active with optimal algorithm (AE) gaussian kernel
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(2, 101, 2)]
ax.plot(M_nwk, LE_gau_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_naive_mean, c='royalblue', linestyle=':', linewidth=1.4)
# ax.plot(M_nwk, AE_lc_epan_mean, c='royalblue', linestyle='--', linewidth=1.4)
# ax.plot(M_nwk, AE_lc_knn_mean[0:30], c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_lc_gau_mean, c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, AE_lc_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, MSE_lr_mean, c='forestgreen', linestyle='-.', linewidth=2.0)         # LR
ax.plot(M_nwk, MSE_gau_mean, c='black', linestyle='-.', linewidth=2.0)   # GE with gaussian kernel

plt.legend(['$LE_{opt, Gau}$', '$AE_{active, Gau}$', '$AE_{active, Hybrid}$','$GE_{lr}$', '$GE_{Gau}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_AE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: compare the AE_log_active with optimal algorithm(GE) knn
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(2, 37, 2)]
ax.plot(M_nwk, LE_knn_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_naive_mean, c='royalblue', linestyle=':', linewidth=1.4)
# ax.plot(M_nwk, AE_lc_epan_mean, c='royalblue', linestyle='--', linewidth=1.4)
# ax.plot(M_nwk, AE_lc_knn_mean[0:30], c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_lc_knn_mean[0:18], c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, AE_lc_hybrid_mean[0:18], c='brown', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, MSE_lr_mean[0:18], c='forestgreen', linestyle='-.', linewidth=2.0)         # LR
ax.plot(M_nwk, MSE_knn_mean, c='black', linestyle='-.', linewidth=2.0)   # GE with gaussian kernel
plt.legend(['$LE_{opt, Knn}$', '$AE_{active, Knn}$', '$AE_{active, Hybrid}$','$GE_{lr}$', '$GE_{Knn}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_AE_knn.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


