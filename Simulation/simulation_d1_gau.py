import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
trails_d1_gau2 = loadData.tolist()
print(trails_d1_gau2.keys())

GEs_mean = [np.mean(trails_d1_gau2['GEs'])] * 70
AEs_mean = np.mean(trails_d1_gau2['AEs'], axis=1)  # average AE, LE, GE of 20 trails, axis=1 means averaging on trail axis
LEs_mean = np.mean(trails_d1_gau2['LEs'], axis=1)
AE_adapts_mean = np.mean(trails_d1_gau2['AE_adapts'], axis=1)
AE_logs_mean = np.mean(trails_d1_gau2['AE_logs'], axis=1)
AE_log_actives_mean = np.mean(trails_d1_gau2['AE_log_actives'], axis=1)
LE_adapts_mean = np.mean(trails_d1_gau2['LE_adapts'], axis=1)
AE_log_actives_diff_mean = np.mean(trails_d1_gau2['AE_log_actives_diff'], axis=1)
last_block_size = trails_d1_gau2['last_block_size']


'''
figure 1: h^log is effective
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 355, 5)]     # m = 5,10, ..., 350
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, LEs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AEs_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, AE_adapts_mean, c='brown', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AE_logs_mean, c='brown', linestyle='--', linewidth=2.0)
# plt.axvline(x=150,ls="-",c="green", linewidth=2.0)
plt.legend(['$GE$', '$LE_{opt}$', '$AE_{opt}$', '$AE_{log}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
ax.set_ylabel('MSE (Gaussian)', fontsize='13')
# plt.yscale('log')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()

# +logscale
left, bottom, width, height = 0.5, 0.7, 0.3, 0.25
ax1 = fig.add_axes([left, bottom, width, height])
# ax1.grid(linestyle='-.')
M_nwk = [i for i in range(5, 355, 5)]     # m = 5,10, ..., 350
ax1.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax1.plot(M_nwk, LEs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax1.plot(M_nwk, AEs_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax1.plot(M_nwk, AE_logs_mean, c='brown', linestyle='--', linewidth=2.0)
ax1.set_ylabel('log(MSE)')
plt.yscale('log')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_log(logscale).pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


# left, bottom, width, height = 0.6, 0.6, 0.25, 0.25
# ax1 = fig.add_axes([left, bottom, width, height])
# # ax1.grid(linestyle='-.')
# M_nwk = [i for i in range(5, 151, 5)]
# ax1.plot(M_nwk, GEs_mean[:30], c='black', linestyle='-.', linewidth=2.0)
# ax1.plot(M_nwk, LEs_mean[:30], c='royalblue', linestyle=':', linewidth=2.0)
# ax1.plot(M_nwk, AEs_mean[:30], c='royalblue', linestyle='--', linewidth=2.0)
# # ax.plot(M_nwk, AE_adapts_mean[:30], c='brown', linestyle=':', linewidth=2.0)
# ax1.plot(M_nwk, AE_logs_mean[:30], c='brown', linestyle='--', linewidth=2.0)
#
# ax1.set_ylabel('MSE')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_log(detail).pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()


'''
figure 2: h^log is effective, in derail
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 151, 5)]
ax.plot(M_nwk, GEs_mean[:30], c='black', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, LEs_mean[:30], c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AEs_mean[:30], c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, AE_adapts_mean[:30], c='brown', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AE_logs_mean[:30], c='brown', linestyle='--', linewidth=2.0)
plt.legend(['$GE$', '$LE_{opt}$', '$AE_{opt}$', '$AE_{log}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents\n(d=1)', fontsize='13')
ax.set_ylabel('MSE (Gaussian)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_log_150.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 3: active rule is effective
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 351, 5)]
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax.plot(M_nwk, AE_logs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=2.0)
plt.legend(['$GE$', '$AE_{log}$', '$AE_{active}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
ax.set_ylabel('MSE (Gaussian)', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()


# +logscale
left, bottom, width, height = 0.5, 0.7, 0.3, 0.25
ax1 = fig.add_axes([left, bottom, width, height])
# ax1.grid(linestyle='-.')
M_nwk = [i for i in range(5, 355, 5)]     # m = 5,10, ..., 350
ax1.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax1.plot(M_nwk, AE_logs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax1.plot(M_nwk, AE_log_actives_mean, c='brown', linestyle='--', linewidth=2.0)

ax1.set_ylabel('log(MSE)')
plt.yscale('log')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_active(logscale).pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()



'''
figure 4: MSE of same block size and different block size
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
M_nwk = [i for i in range(5, 351, 5)]

ax.plot(M_nwk, LE_adapts_mean, c='forestgreen', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, AE_log_actives_diff_mean, c='brown', linestyle='--', linewidth=2.0)
ax.plot(M_nwk, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
# plt.ylim(0.00002, 0.00020)
plt.legend(['$LE_{adapt}$', '$AE_{active,same}$', '$AE_{active,diff}$', '$GE$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
ax.set_ylabel('MSE (Gaussian)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_same_diff_compare(correct).pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 5: last block size changes with blocks number
'''
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# M_nwk = [i for i in range(5, 351, 5)]
# ax.plot(M_nwk, last_block_size, c='black', linestyle='-.', linewidth=1.4)
# ax.set_xlabel('Blocks number \n(d=1, diff block size)', fontsize='13')
# ax.set_ylabel('Block size', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/last_block_size.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()





