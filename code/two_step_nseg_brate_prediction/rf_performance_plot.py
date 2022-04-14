"""
    @description:       File for plotting training performance metrics
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              10.07.2020
"""

from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
#  Import Section: LOAD DATA
# =============================================================================

# Python lib/std-pkgs imports
import os
import pickle
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Load Data Imports
import load_data as ld
from rawDataHandler import RawDataHandler, NodeDfStore

# =============================================================================
#  Load dataframe with metrics
# =============================================================================

src_folder = "/home/daniel/Documents/00_DNA/DASH/dash-repo/code/rf_output/francis-2020-07-17/"
read_path_mode = src_folder + "log_mode_xgb.csv"
read_path_nseg = src_folder + "log_nseg_xgb.csv"
read_path_gnb = src_folder + "log_gnb.csv"

df_mode = pd.read_csv(read_path_mode)
df_nseg = pd.read_csv(read_path_nseg)
df_gnb = pd.read_csv(read_path_gnb)

auroc_vmin = 85
auroc_vmax = 99
acc_vmin = 68
acc_vmax = 88

""" MODE """
# MODE TEST AUROC
font = 16
fig_1 = plt.figure(figsize=(8, 5))
ax_1 = Axes3D(fig_1)
Z = round(df_mode['mode_test_auc']*100, 1)
surf_1 = ax_1.plot_trisurf(df_mode['mode_tree_depth'], df_mode['mode_n_trees'], Z, cmap=cm.coolwarm, linewidth=0.2,
                       # vmin=auroc_vmin,
                       # vmax=auroc_vmax
                       )
ax_1.set_title('Bitrate Mode Predictor Mean AUROC Test set', fontsize=font+1)
ax_1.set_ylabel('No. of trees', fontsize=font-4)
ax_1.set_xlabel('Max. tree depth', fontsize=font-4)
ax_1.set_zlabel('AUROC [%]', fontsize=font - 4)
elev = 15
azim = -45
ax_1.view_init(elev, azim)
fig_1.colorbar(surf_1, shrink=0.5, aspect=10)
# plt.show()

# MODE TEST ACC
font = 16
fig_2 = plt.figure(figsize=(8, 5))
ax_2 = Axes3D(fig_2)
Z = round(df_mode['test_acc']*100, 1)
surf_2 = ax_2.plot_trisurf(df_mode['mode_tree_depth'], df_mode['mode_n_trees'], Z, cmap=cm.GnBu, linewidth=0.2,
                       # vmin=acc_vmin,
                       # vmax=acc_vmax
                       )
ax_2.set_title('Bitrate Mode Predictor Accuracy Test set', fontsize=font+1)
ax_2.set_ylabel('No. of trees', fontsize=font-4)
ax_2.set_xlabel('Max. tree depth', fontsize=font-4)
ax_2.set_zlabel('Accuracy [%]', fontsize=font - 4)
elev = 15
azim = -45
ax_2.view_init(elev, azim)
fig_2.colorbar(surf_2, shrink=0.5, aspect=10)
# plt.show()

""" NSEG """
# NSEG TEST ACC
font = 16
fig_3 = plt.figure(figsize=(8, 5))
ax_3 = Axes3D(fig_3)
Z = round(df_nseg['nseg_test_auc']*100, 1)
surf_3 = ax_3.plot_trisurf(df_nseg['nseg_tree_depth'], df_nseg['nseg_n_trees'], Z, cmap=cm.coolwarm, linewidth=0.2,
                       # vmin=auroc_vmin,
                       # vmax=auroc_vmax
                       )
ax_3.set_title('No. Segments Predictor Mean AUROC Test set', fontsize=font+1)
ax_3.set_ylabel('No. of trees', fontsize=font-4)
ax_3.set_xlabel('Max. tree depth', fontsize=font-4)
ax_3.set_zlabel('AUROC [%]', fontsize=font - 4)
elev = 15
azim = -45
ax_3.view_init(elev, azim)
cbar = fig_3.colorbar(surf_3, shrink=0.5, aspect=10)
# plt.show()

# NSEG TRAIN ACC
font = 16
fig_4 = plt.figure(figsize=(8, 5))
ax_4 = Axes3D(fig_4)
Z = round(df_nseg['test_acc']*100, 1)
surf_4 = ax_4.plot_trisurf(df_nseg['nseg_tree_depth'], df_nseg['nseg_n_trees'], Z, cmap=cm.GnBu, linewidth=0.2,
                       # vmin=acc_vmin,
                       # vmax=acc_vmax
                       )
ax_4.set_title('No. Segments Predictor Accuracy Test set', fontsize=font+1)
ax_4.set_ylabel('No. of trees', fontsize=font-4)
ax_4.set_xlabel('Max. tree depth', fontsize=font-4)
ax_4.set_zlabel('Accuracy [%]', fontsize=font - 4)
elev = 15
azim = -45
ax_4.view_init(elev, azim)
cbar = fig_4.colorbar(surf_4, shrink=0.5, aspect=10)


plt.show()

print("EOF")

# alt = 2
# # NOTE: alternative 1
# if alt == 1:
#     X,Y = np.meshgrid(df_mode['mode_n_trees'].values,df_mode['mode_tree_depth'].values)
#     print(X.shape)
#     print(Y.shape)
#     Z = df_mode[['mode_n_trees', 'mode_tree_depth', 'mode_test_auc']].values
#     print(Z.shape)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z)
#     plt.show(block=True)
# elif alt == 3:
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     # X = np.arange(-5, 5, 0.25)
#     # Y = np.arange(-5, 5, 0.25)
#     # X, Y = np.meshgrid(X, Y)
#     # R = np.sqrt(X ** 2 + Y ** 2)
#     # Z = np.sin(R)
#     # Make data.
#     X = range(df_mode.shape[0])
#     Y = range(df_mode.shape[0])
#     X, Y = np.meshgrid(X, Y)
#
#     # Z = np.zeros((X.shape[0], X.shape[0]))
#     # Z[np.ravel(X), np.ravel(Y)] = df_mode['mode_test_auc'].values
#     # Z = df_mode['mode_test_auc'].values
#     Z = df_mode[['mode_n_trees', 'mode_tree_depth', 'mode_test_auc']].values
#     # Plot the surface.
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#     # Customize the z axis.
#     # ax.set_zlim(-1.01, 1.01)
#     # ax.zaxis.set_major_locator(LinearLocator(10))
#     # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#     # Add a color bar which maps values to colors.
#     cax = fig.add_axes([0.15, .87, 0.35, 0.03])
#     fig.colorbar(surf, shrink=0.5, aspect=5, cax=cax)
#     plt.show()

