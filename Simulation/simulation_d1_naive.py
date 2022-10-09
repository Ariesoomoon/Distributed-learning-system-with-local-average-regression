import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_naive.npy', allow_pickle=True)
trails_d1_naive = loadData.tolist()
print(trails_d1_naive.keys())

GEs_mean = [np.mean(trails_d1_naive['GEs'])] * 70
AEs_mean = np.mean(trails_d1_naive['AEs'], axis=1)
LEs_mean = np.mean(trails_d1_naive['LEs'], axis=1)
AE_adapts_mean = np.mean(trails_d1_naive['AE_adapts'], axis=1)
AE_logs_mean = np.mean(trails_d1_naive['AE_logs'], axis=1)
AE_log_actives_mean = np.mean(trails_d1_naive['AE_log_actives'], axis=1)


'''
figure 1: h^log is effective
'''
M_nwk = [i for i in range(5, 355, 5)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AEs_mean, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_adapts_mean, c='brown', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean, c='brown', linestyle='--', linewidth=1.4)
plt.axvline(x=15,ls="-",c="green", linewidth=1.2)
plt.legend(['GE', 'LE', 'AE_opt_h', 'AE_adapt', 'AE_log'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(d=1, same block size)', fontsize='13')
ax.set_ylabel('MSE (Naive)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/naive_d1_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: h^log is effective, in derail
'''
M_nwk = [i for i in range(5, 16, 5)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean[:3], c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean[:3], c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AEs_mean[:3], c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_adapts_mean[:3], c='brown', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean[:3], c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GE', 'LE', 'AE_opt_h', 'AE_adapt', 'AE_log'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(d=1, same block size)', fontsize='13')
ax.set_ylabel('MSE (Naive)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/naive_d1_log_15.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 3: active rule is effective
'''
M_nwk = [i for i in range(5, 351, 5)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GEs', 'AE_log', 'AE_log_active'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(d=1, same block size)', fontsize='13')
ax.set_ylabel('MSE (Naive)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/naive_d1_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
print('picture is done')


