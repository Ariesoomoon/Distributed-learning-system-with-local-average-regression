import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import pdb, time, os
import sys


'''
1. load data
'''
loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', allow_pickle=True)
warfarin_naive = loadData_naive.tolist()
print(warfarin_naive.keys())

GEs_naive = [np.mean(warfarin_naive['GEs'])] * 50
LEs_naive = np.mean(warfarin_naive['LEs'], axis=1)
AE_log_actives_naive = np.mean(warfarin_naive['AE_log_actives'], axis=1)
# RMSE
# GEs_naive = np.array(GEs_naive)**0.5
# LEs_naive = np.array(LEs_naive)**0.5
# AE_log_actives_naive = AE_log_actives_naive ** 0.5


# M_nwk = [i for i in range(2, 101, 2)]
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# ax.plot(M_nwk, GEs_naive, c='black', linestyle='-.', linewidth=1.4)
# ax.plot(M_nwk, LEs_naive, c='royalblue', linestyle='--', linewidth=1.4)
# ax.plot(M_nwk, AE_log_actives_naive, c='brown', linestyle='--', linewidth=1.4)
# plt.legend(['GE', 'LE', 'AE_active'], loc='upper left', fontsize='medium')
# ax.set_xlabel('Blocks number \n(warfarin data, d=6, same block size)', fontsize='13')
# ax.set_ylabel('MSE (Naive)', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/naive_LE_AE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()



loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
warfarin_gau = loadData_gau.tolist()
print(warfarin_gau.keys())
GEs_gau = [np.mean(warfarin_gau['GEs'])] * 50
LEs_gau = np.mean(warfarin_gau['LEs'], axis=1)
AE_log_actives_gau = np.mean(warfarin_gau['AE_log_actives'], axis=1)
# RMSE
# GEs_gau = np.array(GEs_gau)**0.5
# LEs_gau = np.array(LEs_gau)**0.5
# AE_log_actives_gau = AE_log_actives_gau ** 0.5

loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', allow_pickle=True)
warfarin_epan = loadData_epan.tolist()
GEs_epan = [np.mean(warfarin_epan['GEs'])] * 50
LEs_epan = np.mean(warfarin_epan['LEs'], axis=1)
AE_log_actives_epan = np.mean(warfarin_epan['AE_log_actives'], axis=1)
# RMSE
# GEs_epan = np.array(GEs_epan)**0.5
# LEs_epan = np.array(LEs_epan)**0.5
# AE_log_actives_epan = AE_log_actives_epan ** 0.5

loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
warfarin_knn = loadData_knn.tolist()
GEs_knn = [np.mean(warfarin_knn['GEs'])] * 50
LEs_knn = np.mean(warfarin_knn['LEs'], axis=1)
AE_log_actives_knn = np.mean(warfarin_knn['AE_log_actives'], axis=1)
print(min(warfarin_knn['global_k']))  # 36
# RMSE
# GEs_knn = np.array(GEs_knn)**0.5
# LEs_knn = np.array(LEs_knn)**0.5
# AE_log_actives_knn = AE_log_actives_knn ** 0.5


loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData_knn.tolist()
GEs_lr = [np.mean(warfarin_hybrid['MSE_lr'])] * 50
AE_lc_hybrid_mean = np.mean(warfarin_hybrid['AE_lc_hybrid'], axis=1)
# RMSE
# GEs_lr = np.array(GEs_lr)**0.5


'''
figure 1: compare AE--MSE
'''
M_nwk = [i for i in range(2, 101, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
# ax.plot(M_nwk, GEs_mean, c='black', linestyle='-', linewidth=1.4)
# ax.plot(M_nwk, GEs_gau, c='black', linestyle='-.', linewidth=1.4)
# ax.plot(M_nwk, GEs_epan, c='black', linestyle='--', linewidth=1.4)
# ax.plot(M_nwk, GEs_knn, c='black', linestyle=':', linewidth=1.4)
# ax.plot(M_nwk, GEs_lr, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_gau, c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_naive, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_epan, c='royalblue', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_knn, c='royalblue', linestyle=':', linewidth=2.0)

plt.legend(['$AE_{active, Gau}$', '$AE_{active, Naive}$', '$AE_{active, Epan}$', '$AE_{active, Knn}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/compare_AE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: compare LE--MSE
'''
M_nwk = [i for i in range(2, 37, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, LEs_gau[0:18], c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, LEs_naive[0:18], c='royalblue', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, LEs_epan[0:18], c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, LEs_knn, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AE_lc_hybrid_mean[0:18], c='brown', linestyle='--', linewidth=2.0)
plt.ylim(0.00365, 0.0050)
plt.legend(['$LE_{opt, Gau}$', '$LE_{opt, Naive}$', '$LE_{opt, Epan}$', '$LE_{opt, Knn}$', '$AE_{active, Hybrid}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/compare_LE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 3: Hybrid_AE
'''
M_nwk = [i for i in range(2, 101, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
# ax.plot(M_nwk, GEs_mean, c='black', linestyle='-', linewidth=1.4)
# ax.plot(M_nwk, GEs_gau, c='black', linestyle='-.', linewidth=1.4)
# ax.plot(M_nwk, GEs_epan, c='black', linestyle='--', linewidth=1.4)
# ax.plot(M_nwk, GEs_knn, c='black', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_gau, c='royalblue', linestyle='-', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_naive, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_epan, c='royalblue', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_knn, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AE_lc_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, GEs_lr, c='Forestgreen', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, GEs_knn, c='Black', linestyle='--', linewidth=2.0)
plt.ylim(0.00365, 0.0045)
plt.legend(['$AE_{active, Gau}$', '$AE_{active, Naive}$', '$AE_{active, Epan}$',
            '$AE_{active, Knn}$', '$AE_{active, Hybrid}$', '$GE_{LR}$', '$GE_{Knn}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/Hybrid_AE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()



