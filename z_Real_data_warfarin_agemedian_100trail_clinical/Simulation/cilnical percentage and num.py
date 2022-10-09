import numpy as np
from Function_same_block_warfarin import clinical_percentage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import pdb, time, os
import sys



'''1. load data'''
# loadData = np.load(os.path.dirname(os.getcwd()) + '/result_data/warfarin_afterpreprocess.npy', allow_pickle=True)
# warfarin_afterpreprocess = loadData.tolist()
# X = warfarin_afterpreprocess['X']
# y = warfarin_afterpreprocess['y']
# print(warfarin_afterpreprocess.keys())


print('-------------------------------------------------------------- warfarin_epan')
loadData_epan = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_epan.npy', allow_pickle=True)
warfarin_epan = loadData_epan.tolist()
# print(warfarin_epan.keys())
# print(warfarin_epan['percs_AE'])
print(warfarin_epan['percs_AE'].shape)
percs_AE_epan = np.mean(warfarin_epan['percs_AE'], axis=1)
print('percs_AE_epan:', percs_AE_epan)
percs_LE_epan = np.mean(warfarin_epan['percs_LE'], axis=1)
print('percs_LE_epan:', percs_LE_epan)
percs_GE_epan = np.mean(warfarin_epan['percs_GE'], axis=1)
print('percs_GE_epan:',percs_GE_epan)


print('-------------------------------------------------------------- warfarin_gau')
loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_gau.npy', allow_pickle=True)
warfarin_gau = loadData_gau.tolist()
# print(warfarin_gau.keys())
# print(warfarin_gau['percs_AE'])
print(warfarin_gau['percs_AE'].shape)
percs_AE_gau = np.mean(warfarin_gau['percs_AE'], axis=1)
print('percs_AE_gau:', percs_AE_gau)
percs_LE_gau = np.mean(warfarin_gau['percs_LE'], axis=1)
print('percs_LE_gau:', percs_LE_gau)
percs_GE_gau = np.mean(warfarin_gau['percs_GE'], axis=1)
print('percs_GE_gau:', percs_GE_gau)


print('-------------------------------------------------------------- warfarin_naive')
loadData_naive = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_naive.npy', allow_pickle=True)
warfarin_naive = loadData_naive.tolist()
# print(warfarin_naive.keys())
# print(warfarin_naive['percs_AE'])
print(warfarin_naive['percs_AE'].shape)
percs_AE_naive = np.mean(warfarin_naive['percs_AE'], axis=1)
print('percs_AE_naive:', percs_AE_naive)
percs_LE_naive = np.mean(warfarin_naive['percs_LE'], axis=1)
print('percs_LE_naive:', percs_LE_naive)
percs_GE_naive = np.mean(warfarin_naive['percs_GE'], axis=1)
print('percs_GE_naive:', percs_GE_naive)


print('-------------------------------------------------------------- warfarin_knn')
loadData_knn = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_knn.npy', allow_pickle=True)
warfarin_knn = loadData_knn.tolist()
# print(warfarin_knn.keys())
# print(warfarin_knn['percs_AE'])
print(warfarin_knn['percs_AE'].shape)
percs_AE_knn = np.mean(warfarin_knn['percs_AE'], axis=1)
print('percs_AE_knn:', percs_AE_knn)
percs_LE_knn = np.mean(warfarin_knn['percs_LE'], axis=1)
print('percs_LE_knn:', percs_LE_knn)
percs_GE_knn = np.mean(warfarin_knn['percs_GE'], axis=1)
print('percs_GE_knn:', percs_GE_knn)


print('-------------------------------------------------------------- warfarin_lr')
loadData_hybrid = np.load(os.path.dirname(os.getcwd()) + '/Result_data/warfarin_hybrid.npy', allow_pickle=True)
warfarin_hybrid = loadData_hybrid.tolist()
print(warfarin_hybrid.keys())
print(warfarin_hybrid['percs_lr'].shape)
percs_GE_lr = np.mean(warfarin_hybrid['percs_lr'], axis=1)
print('percs_GE_lr:', percs_GE_lr)


print('-------------------------------------------------------------- warfarin_hybrid')
print(warfarin_hybrid['percs_AE'].shape)
percs_AE_hybrid = np.mean(warfarin_hybrid['percs_AE'], axis=1)
print('percs_AE_hybrid:', percs_AE_hybrid)


# patients number requiring low, intermediate and high actual dose
num_lows = np.mean(warfarin_hybrid['num_lows'])
num_highs = np.mean(warfarin_hybrid['num_highs'])
num_intermediates = np.mean(warfarin_hybrid['num_intermediates'])
print('num_lows:', num_lows)
print('num_intermediates:', num_intermediates)
print('num_highs:', num_highs)
# num_lows: 368.2  -- 368
# num_intermediates: 632.3 -- 632
# num_highs: 139.5 -- 140
