import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_knn.npy', allow_pickle=True)
trails_d1_knn = loadData.tolist()
print(trails_d1_knn.keys())

GEs_mean = [np.mean(trails_d1_knn['GEs'])] * 70
AEs_mean = np.mean(trails_d1_knn['AEs'], axis=1)
LEs_mean = np.mean(trails_d1_knn['LEs'], axis=1)
AE_adapts_mean = np.mean(trails_d1_knn['AE_adapts'], axis=1)
AE_logs_mean = np.mean(trails_d1_knn['AE_logs'], axis=1)
AE_log_actives_mean = np.mean(trails_d1_knn['AE_log_actives'], axis=1)


'''
figure 1: k^log is effective
'''
M_knn = [i for i in range(5, 241, 5)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_knn, GEs_mean[:48], c='black', linestyle='-.', linewidth=2.0)
ax.plot(M_knn, LEs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(M_knn, AEs_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_knn, AE_adapts_mean[:48], c='brown', linestyle=':', linewidth=2.0)
ax.plot(M_knn, AE_logs_mean[:48], c='brown', linestyle='--', linewidth=2.0)
plt.legend(['$GE$', '$LE_{opt}$', '$AE_{opt}$', '$AE_{log}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents\n(d=1)', fontsize='13')
ax.set_ylabel('MSE (Knn)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_d1_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: k^log is effective, adopt LEs_mean
'''
M_knn = [i for i in range(5, 241, 5)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(M_knn, GEs_mean[:48], c='black', linestyle='-.', linewidth=2.0)
ax.plot(M_knn, AEs_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_knn, AE_adapts_mean[:48], c='brown', linestyle=':', linewidth=2.0)
ax.plot(M_knn, AE_logs_mean[:48], c='brown', linestyle='--', linewidth=2.0)
plt.ylim(0.00002, 0.0004)
plt.legend(['$GE$', '$AE_{opt}$', '$AE_{log}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents\n(d=1)', fontsize='13')
ax.set_ylabel('MSE (Knn)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_d1_log-.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
print('picture is done')
plt.show()


'''
figure 3: active rule is effective
'''
# M_knn = [i for i in range(5, 351, 5)]
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# ax.plot(M_knn, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
# ax.plot(M_knn, AE_logs_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax.plot(M_knn, AE_log_actives_mean, c='brown', linestyle='--', linewidth=2.0)
# plt.ylim(0.00002, 0.00014)
# plt.legend(['$GE$', '$AE_{log}$', '$AE_{active}$'], loc='upper left', fontsize='medium')
# ax.set_xlabel('Number of local agents\n(d=1)', fontsize='13')
# ax.set_ylabel('MSE (Knn)', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/knn_d1_active.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()
# print('picture is done')
