import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import pdb, time, os
import sys


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
warfarin_knn = loadData.tolist()
print(warfarin_knn.keys())

GEs_mean = [np.mean(warfarin_knn['GEs'])] * 18
LEs_mean = np.mean(warfarin_knn['LEs'], axis=1)
AE_log_actives_mean = np.mean(warfarin_knn['AE_log_actives'], axis=1)


M_nwk = [i for i in range(2, 37, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_mean[0:18], c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GE', 'LE', 'AE_active'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(warfarin data, d=6, same block size)', fontsize='13')
ax.set_ylabel('MSE (Knn)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_LE_AE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()



