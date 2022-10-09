import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/insurance_epan.npy', allow_pickle=True)
insurance_epan = loadData.tolist()
print(insurance_epan.keys())

GEs_mean = [np.mean(insurance_epan['GEs'])] * 30
AEs_mean = np.mean(insurance_epan['AEs'], axis=1)
LEs_mean = np.mean(insurance_epan['LEs'], axis=1)
AE_adapts_mean = np.mean(insurance_epan['AE_adapts'], axis=1)
AE_logs_mean = np.mean(insurance_epan['AE_logs'], axis=1)
AE_log_actives_mean = np.mean(insurance_epan['AE_log_actives'], axis=1)
print(AE_log_actives_mean)
print(np.mean(insurance_epan['GEs']))
print(AEs_mean[10])
print(LEs_mean[0])


'''
figure 1: h^log is effective
'''
M_nwk = [i for i in range(2, 61, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean, c='royalblue', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AEs_mean, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_adapts_mean, c='brown', linestyle=':', linewidth=1.4)
ax.plot(M_nwk, AE_logs_mean, c='brown', linestyle='--', linewidth=1.4)
plt.axvline(x=12,ls="-",c="green", linewidth=1.2)
plt.legend(['GE', 'LE', 'AE_opt_h', 'AE_adapt', 'AE_log'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(insurance data, d=6, same block size)', fontsize='13')
ax.set_ylabel('MSE (Epanechnikov)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/epan_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: active rule is effective
'''
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
ax.set_ylabel('MSE (Epanechnikov)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/epan_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
print('picture is done')
