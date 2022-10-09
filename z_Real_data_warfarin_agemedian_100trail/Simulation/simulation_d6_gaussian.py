import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import pdb, time, os
import sys


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
warfarin_gau = loadData.tolist()
print(warfarin_gau.keys())

GEs_mean = [np.mean(warfarin_gau['GEs'])] * 50
LEs_mean =  np.mean(warfarin_gau['LEs'], axis=1)
AE_log_actives_mean = np.mean(warfarin_gau['AE_log_actives'], axis=1)
# RMSE
# GEs_mean = np.array(GEs_mean)**0.5
# LEs_mean = np.array(LEs_mean)**0.5
# AE_log_actives_mean = AE_log_actives_mean ** 0.5

# 1.1 gaussian
M_nwk = [i for i in range(2, 101, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=1.4)
ax.plot(M_nwk, LEs_mean, c='royalblue', linestyle='--', linewidth=1.4)
ax.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=1.4)
plt.legend(['GE', 'LE', 'AE_active'], loc='upper left', fontsize='medium')
ax.set_xlabel('Blocks number \n(warfarin data, d=6, same block size)', fontsize='13')
ax.set_ylabel('MSE (Gaussian)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau_LE_AE.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()





