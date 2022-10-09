import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_knn.npy', allow_pickle=True)
trails_d5_knn = loadData.tolist()
print(trails_d5_knn.keys())

GEs_mean = [np.mean(trails_d5_knn['GEs'])] * 19
AEs_m20_mean = np.mean(trails_d5_knn['AEs_m20'], axis=1)
LEs_m20_mean = np.mean(trails_d5_knn['LEs_m20'], axis=1)
AE_adapts_m20_mean = np.mean(trails_d5_knn['AE_adapts_m20'], axis=1)
AE_logs_m20_mean = np.mean(trails_d5_knn['AE_logs_m20'], axis=1)
AE_log_actives_20mean = np.mean(trails_d5_knn['AE_log_actives'], axis=1)


'''
figure 1: k^log is effective
'''
M_nwk = [i for i in range(2, 21, 1)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, LEs_m20_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AEs_m20_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, AE_adapts_m20_mean, c='brown', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AE_logs_m20_mean, c='brown', linestyle='--', linewidth=2.0)
plt.legend(['$GE$', '$LE_{opt}$', '$AE_{opt}$', '$AE_{log}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents\n(d=5)', fontsize='13')
ax.set_ylabel('MSE (Knn)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_d5_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 3: active rule is effective
'''
# GEs_mean = [np.mean(trails_d5_knn['GEs'])] * 70
# AE_logs_mean = np.mean(trails_d5_knn['AE_logs'], axis=1)
# AE_log_actives_mean = np.mean(trails_d5_knn['AE_log_actives'], axis=1)
# M_nwk = [i for i in range(5, 351, 5)]
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
# ax.plot(M_nwk, AE_logs_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=2.0)
# plt.legend(['$GE$', '$AE_{log}$', '$AE_{active}$'], loc='upper left', fontsize='medium')
# ax.set_xlabel('Number of local agents\n(d=5)', fontsize='13')
# ax.set_ylabel('MSE (Knn)', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_d5_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()
# print('picture is done')

