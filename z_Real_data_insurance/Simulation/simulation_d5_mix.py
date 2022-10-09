import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os
time_start = time.time()


'''
1. load data
'''
loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_gau.npy', allow_pickle=True)
insurance_gau = loadData_gau.tolist()
MSE_gaussian_mean = [np.mean(insurance_gau['GEs'])] * 30

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_hybrid.npy', allow_pickle=True)
insurance_hybrid = loadData.tolist()

AE_lc_epan_mean = np.mean(loadData.tolist()['AE_lc_epan'], axis=1)
AE_lc_gau_mean = np.mean(loadData.tolist()['AE_lc_gau'], axis=1)
AE_lc_naive_mean = np.mean(loadData.tolist()['AE_lc_naive'], axis=1)
AE_lc_knn_mean = np.mean(loadData.tolist()['AE_lc_k'], axis=1)
AE_lc_hybrid_mean = np.mean(loadData.tolist()['AE_lc_hybrid'], axis=1)

LE_epan_mean = np.mean(loadData.tolist()['LE_epan'], axis=1)
LE_gau_mean = np.mean(loadData.tolist()['LE_gau'], axis=1)
LE_naive_mean = np.mean(loadData.tolist()['LE_naive'], axis=1)
LE_knn_mean = np.mean(loadData.tolist()['LE_knn'], axis=1)


# other methods
MSE_lr_mean = [np.mean(insurance_hybrid['MSE_lr'])] * 30       # linear regression
MSE_rfr_mean = [np.mean(insurance_hybrid['MSE_rfr'])] * 30     # RandomForestRegressor
MSE_tree_mean = [np.mean(insurance_hybrid['MSE_tree'])] * 30   # Decision tree
MSE_knn_mean = [np.mean(insurance_hybrid['MSE_knn'])] * 30     # KNeighbors Regressor
MSE_ada_mean = [np.mean(insurance_hybrid['MSE_ada'])] * 30     # AdaBoost Regressor
MSE_gbr_mean = [np.mean(insurance_hybrid['MSE_gbr'])] * 30     # Gradient Boosting Regressor



'''
figure 1: compare the AE_log_active on each algorithm
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(2, 61, 2)]
ax.plot(M_nwk, AE_lc_gau_mean, c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, AE_lc_naive_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, AE_lc_epan_mean, c='royalblue', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, AE_lc_knn_mean, c='royalblue', linestyle=':', linewidth=2.0)

ax.plot(M_nwk, AE_lc_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, MSE_lr_mean, c='forestgreen', linestyle='--', linewidth=2.0)         # LR
ax.plot(M_nwk, MSE_gaussian_mean, c='black', linestyle='-.', linewidth=2.0)   # GE with gaussian kernel
# ax.plot(M_nwk, MSE_rfr_mean, c='forestgreen', linestyle=':', linewidth=1.4)       # RandomForestRegressor
# ax.plot(M_nwk, MSE_tree_mean, c='forestgreen', linestyle=':', linewidth=1.4)      # Decision tree
# ax.plot(M_nwk, MSE_knn_mean, c='forestgreen', linestyle=':', linewidth=1.4)       # KNeighbors Regressor
# ax.plot(M_nwk, MSE_ada_mean, c='forestgreen', linestyle=':', linewidth=1.4)       # AdaBoost Regressor
# ax.plot(M_nwk, MSE_gbr_mean, c='forestgreen', linestyle='--', linewidth=1.4)      # Gradient Boosting Regressor
plt.ylim(0.0065, 0.013)
plt.legend(['$AE_{active, Gau}$', '$AE_{active, Naive}$', '$AE_{active, Epan}$', '$AE_{active, Knn}$', '$AE_{active, Hybrid}$','$GE_{lr}$', '$GE_{Gau}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_AE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
# print(AE_lc_hybrid_mean-MSE_lr_mean)  # our algorithm outperforms linear regression when m <= 12



'''
figure 2: compare the AE_log_active on each algorithm
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(2, 61, 2)]
ax.plot(M_nwk, LE_gau_mean, c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, LE_naive_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, LE_epan_mean, c='royalblue', linestyle='-.', linewidth=2.0)
# ax.plot(M_nwk, LE_knn_mean, c='royalblue', linestyle=':', linewidth=2.0)

ax.plot(M_nwk, AE_lc_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
plt.ylim(0.005, 0.06)
plt.legend(['$LE_{opt, Gau}$', '$LE_{opt, Naive}$', '$LE_{opt, Epan}$', '$AE_{active, Hybrid}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/LE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


