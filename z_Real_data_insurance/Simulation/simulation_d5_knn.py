import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_knn.npy', allow_pickle=True)
insurance_knn = loadData.tolist()
print(insurance_knn.keys())

GEs_m5_mean = [np.mean(insurance_knn['GEs'])] * 4
AEs_m5_mean = np.mean(insurance_knn['AEs'], axis=1)
LEs_m5_mean = np.mean(insurance_knn['LEs'], axis=1)
AE_adapts_m5_mean = np.mean(insurance_knn['AE_adapts_m5'], axis=1)
AE_logs_m5_mean = np.mean(insurance_knn['AE_logs_m5'], axis=1)


'''
figure 1: k^log is effective
'''
M_nwk = [i for i in range(2, 6, 1)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_m5_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_m5_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AEs_m5_mean, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_adapts_m5_mean, c='brown', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_logs_m5_mean, c='brown', linestyle='--', linewidth=1.4)
ax.set_xticks([2, 3, 4, 5])
plt.legend(['GE', 'LE', 'AE_opt_k', 'AE_adapt', 'AE_log'], loc='upper left', fontsize='medium')

ax.set_xlabel('Blocks number \n(insurance data, d=6, same block size)', fontsize='13')
ax.set_ylabel('MSE (Knn)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_log_m5.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()



'''
figure 2: active rule is effective
'''
GEs_mean = [np.mean(insurance_knn['GEs'])] * 30
AE_log_actives_mean = np.mean(insurance_knn['AE_log_actives'], axis=1)
AE_logs_mean = np.mean(insurance_knn['AE_logs'], axis=1)

M_nwk = [i for i in range(2, 61, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GEs', 'AE_log', 'AE_log_active'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(insurance data, d=6, same block size)', fontsize='13')
ax.set_ylabel('MSE (Knn)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
print('picture is done')

