import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_hybrid.npy', allow_pickle=True)
trails_d1_hybrid = loadData.tolist()
print(trails_d1_hybrid.keys())
AE_lc_epan_mean = np.mean(loadData.tolist()['AE_lc_epan'], axis=1)
AE_lc_gau_mean = np.mean(loadData.tolist()['AE_lc_gau'], axis=1)
AE_lc_naive_mean = np.mean(loadData.tolist()['AE_lc_naive'], axis=1)
AE_lc_knn_mean = np.mean(loadData.tolist()['AE_lc_k'], axis=1)
AE_lc_hybrid_mean = np.mean(loadData.tolist()['AE_lc_hybrid'], axis=1)
GE_gau_mean = [np.mean(loadData.tolist()['GE_gau'])] * 70



'''
figure 1: compare the AE_log_active on each algorithm
'''
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# M_nwk = [i for i in range(5, 355, 5)]
# ax.plot(M_nwk, AE_lc_knn_mean, c='forestgreen', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_naive_mean, c='forestgreen', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_epan_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_gau_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, GE_gau_mean, c='black', linestyle='--', linewidth=2.0)
#
# plt.ylim(0.00001, 0.00023)
# plt.legend(['$AE_{active, Knn}$', '$AE_{active, Naive}$', '$AE_{active, Epan}$', '$AE_{active, Gau}$', '$AE_{active, Hybrid}$', '$GE_{Gau}$'], loc='upper left', fontsize='medium')
# ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
# ax.set_ylabel('MSE', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_d1_AE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()



'''
figure 2: prove the effective of hybrid algorithm. Compare the LE on each algorithm, GE of NWK(gaussian) as our baseline.
'''
# # (1) save LE of each algorithm to trails_d1_hybrid.npy
# loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_epan.npy', allow_pickle=True)
# loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
# loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_naive.npy', allow_pickle=True)
# loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_knn.npy', allow_pickle=True)
# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_hybrid.npy', allow_pickle=True)
# trails_d1_hybrid = loadData.tolist()
# trails_d1_hybrid['LE_epan'] = loadData_epan.tolist()['LEs']
# trails_d1_hybrid['LE_gau'] = loadData_gau.tolist()['LEs']
# trails_d1_hybrid['LE_naive'] = loadData_naive.tolist()['LEs']
# trails_d1_hybrid['LE_knn'] = loadData_knn.tolist()['LEs']
# trails_d1_hybrid['GE_gau'] = loadData_gau.tolist()['GEs']
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_hybrid.npy', trails_d1_hybrid)
# print('save trails_d1_hybrid.npy done')


# (2) load data and plot
# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_hybrid.npy', allow_pickle=True)
# trails_d1_hybrid = loadData.tolist()
# LE_epan_mean = np.mean(loadData.tolist()['LE_epan'], axis=1)
# LE_gau_mean = np.mean(loadData.tolist()['LE_gau'], axis=1)
# LE_naive_mean = np.mean(loadData.tolist()['LE_naive'], axis=1)
# LE_knn_mean = np.mean(loadData.tolist()['LE_knn'], axis=1)
# GE_gau_mean = [np.mean(loadData.tolist()['GE_gau'])] * 48
# AE_lc_hybrid_mean = np.mean(loadData.tolist()['AE_lc_hybrid'], axis=1)
#
# M_nwk = [i for i in range(5, 241, 5)]   # change 350 to 240, beacuse of Knn
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# ax.plot(M_nwk, LE_knn_mean[:48], c='forestgreen', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, LE_naive_mean[:48], c='forestgreen', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, LE_epan_mean[:48], c='royalblue', linestyle=':', linewidth=2.0)
# ax.plot(M_nwk, LE_gau_mean[:48], c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(M_nwk, AE_lc_hybrid_mean[:48], c='brown', linestyle='--', linewidth=2.0)
# # ax.plot(M_nwk, GE_gau_mean, c='black', linestyle='--', linewidth=2.0)
# plt.legend(['$LE_{opt, Knn}$', '$LE_{opt, Naive}$', '$LE_{opt, Epan}$', '$LE_{opt, Gau}$', '$AE_{active, Hybrid}$'], loc='upper left', fontsize='medium')
# ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
# ax.set_ylabel('MSE', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_d1_LE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()



'''
figure 3: conpare the MSE of same block size and different block size
'''
# loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
# trails_d1_gau2 = loadData_gau.tolist()
# LE_adapt_gau = np.mean(trails_d1_gau2['LE_adapts'], axis=1)
#
# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_hybrid.npy', allow_pickle=True)
# trails_d1_hybrid = loadData.tolist()
# LE_hybrid_diff = np.mean(loadData.tolist()['LE_hybrid_diff'], axis=1)  # LE_adapt of hybird algorithm
# AE_lc_hybrid = np.mean(loadData.tolist()['AE_lc_hybrid'], axis=1)
# AE_lc_hybrid_diff = np.mean(loadData.tolist()['AE_lc_hybrid_diff'], axis=1)
# GE_gau = [np.mean(loadData.tolist()['GE_gau'])] * 70
# print(trails_d1_gau2.keys())
# print(trails_d1_hybrid.keys())
#
#
# M_nwk = [i for i in range(5, 351, 5)]
# fig = plt.figure(tight_layout=True)
# gs = gridspec.GridSpec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# ax.grid(linestyle='-.')
# ax.plot(M_nwk, LE_adapt_gau, c='forestgreen', linestyle=':', linewidth=1.4)
# ax.plot(M_nwk, LE_hybrid_diff, c='forestgreen', linestyle='-.', linewidth=1.4)
# ax.plot(M_nwk, AE_lc_hybrid_diff, c='brown', linestyle='-.', linewidth=1.4)
# ax.plot(M_nwk, AE_lc_hybrid, c='royalblue', linestyle='-.', linewidth=1.4)
# ax.plot(M_nwk, GE_gau, c='black', linestyle='-.', linewidth=1.4)
# # plt.axvline(x=25,ls="-",c="green", linewidth=1.2)
# plt.legend(['LE_adapt_gau', 'LE_adapt', 'Hybrid_diff', 'Hybrid_same', 'GE_gau'], loc='upper left', fontsize='medium')
# ax.set_ylabel('MSE', fontsize='13')
# ax.set_xlabel('Blocks number \n(d=1)', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/compare_same_diff_size_d1.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()

