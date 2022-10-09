import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_epan.npy', allow_pickle=True)
trails_d1_epan = loadData.tolist()
print(trails_d1_epan.keys())

GEs_mean = [np.mean(trails_d1_epan['GEs'])] * 70
AEs_mean = np.mean(trails_d1_epan['AEs'], axis=1)
LEs_mean = np.mean(trails_d1_epan['LEs'], axis=1)
AE_adapts_mean = np.mean(trails_d1_epan['AE_adapts'], axis=1)
AE_logs_mean = np.mean(trails_d1_epan['AE_logs'], axis=1)
AE_log_actives_mean = np.mean(trails_d1_epan['AE_log_actives'], axis=1)


'''
figure 1: h^log is effective
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 355, 5)]
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AEs_mean, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_adapts_mean, c='brown', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean, c='brown', linestyle='--', linewidth=1.4)
plt.axvline(x=30,ls="-",c="green", linewidth=1.2)
plt.legend(['GE', 'LE', 'AE_opt_h', 'AE_adapt', 'AE_log'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(d=1, same block size)', fontsize='13')
ax.set_ylabel('MSE (Epanechnikov)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/epan_d1_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: h^log is effective, in derail
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 31, 5)]
ax.plot(M_nwk, GEs_mean[:6], c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean[:6], c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AEs_mean[:6], c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_adapts_mean[:6], c='brown', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean[:6], c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GE', 'LE', 'AE_opt_h', 'AE_adapt', 'AE_log'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(d=1, same block size)', fontsize='13')
ax.set_ylabel('MSE (Epanechnikov)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/epan_d1_log_30.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 3: active rule is effective
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 351, 5)]
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GEs', 'AE_log', 'AE_log_active'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(d=1, same block size)', fontsize='13')
ax.set_ylabel('MSE (Epanechnikov)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/epan_d1_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
print('picture is done')