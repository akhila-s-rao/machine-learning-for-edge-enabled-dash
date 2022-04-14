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
import matplotlib.ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Load Data Imports
import load_data as ld
from rawDataHandler import RawDataHandler, NodeDfStore

# =============================================================================
#  Load dataframe with metrics
# =============================================================================

src_folder = "/home/daniel/Documents/00_DNA/DASH/dash-repo/code/rf_output/francis-dataset7-4swsize-4sAggsize-2020-07-21/"
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

font=16
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 2))

Z_mode = round(df_mode['test_acc']*100, 1)

axes[1].tricontour(df_mode['mode_n_trees'].values, df_mode['mode_tree_depth'].values,
                   Z_mode, levels=14, linewidths=0.5, colors='k')
cntr_mode = axes[1].tricontourf(df_mode['mode_n_trees'].values, df_mode['mode_tree_depth'].values,
                            Z_mode, levels=14, cmap="coolwarm"
                            )

axes[1].plot(df_mode['mode_n_trees'].values, df_mode['mode_tree_depth'].values, 'ko', ms=3)
axes[1].set_title('MODE Test accuracy', fontsize=font+2)
axes[1].set_xscale('log')
axes[1].set_yticks([5, 15, 25])
axes[1].set_xticks([5, 15, 50, 100, 200, 350])
axes[1].set_ylabel('Max. Tree Depths', fontsize=font)
axes[1].set_xlabel('No. of Trees', fontsize=font)
axes[1].tick_params(axis='x', labelsize=font-2)
axes[1].tick_params(axis='y', labelsize=font-2)
axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.colorbar(cntr_mode, ax=axes[0])

Z_nseg = round(df_nseg['test_acc']*100, 1)

axes[0].tricontour(df_nseg['nseg_n_trees'].values, df_nseg['nseg_tree_depth'].values,
              Z_nseg, levels=14, linewidths=0.5, colors='k')
cntr_nseg = axes[0].tricontourf(df_nseg['nseg_n_trees'].values, df_nseg['nseg_tree_depth'].values,
                       Z_nseg, levels=14, cmap="coolwarm")
axes[0].plot(df_nseg['nseg_n_trees'].values, df_nseg['nseg_tree_depth'].values, 'ko', ms=3)
axes[0].set_title('NSEG Test accuracy', fontsize=font+2)
axes[0].set_xscale('log')
axes[0].set_xticks([5, 15, 50, 100, 200, 350])
axes[0].set_yticks([5, 15, 25])
axes[0].set_ylabel('Max. Tree Depths', fontsize=font)
axes[0].set_xlabel('No. of Trees', fontsize=font)
axes[0].tick_params(axis='x', labelsize=font-2)
axes[0].tick_params(axis='y', labelsize=font-2)
axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

fig.colorbar(cntr_nseg, ax=axes[1])
# save_path = "/home/daniel/Documents/00_DNA/DASH/dash-repo/documentation/diagrams/imgs/pred_plots/best-mode-nseg-train-test-acc.pdf"
fig.savefig(save_path, format="pdf", bbox_inches = 'tight', pad_inches = 0)

plt.show()

# setup some generic data
N = 37
x, y = np.mgrid[:N, :N]
Z = (np.cos(x*0.2) + np.sin(y*0.3))

test_vals = df_mode[['mode_n_trees', 'mode_tree_depth', 'test_acc']].values

max_vals = np.max(test_vals[:,0:2], axis=0).astype(int)

Zgrid = np.zeros((max_vals[0]+1, max_vals[1]+1))

for i in range(test_vals.shape[0]):
    Zgrid[test_vals[i,0].astype(int), test_vals[i,1].astype(int)] = test_vals[i,2]

fig, ax = plt.subplots()
im = ax.imshow(Zgrid, interpolation='bilinear', cmap=cm.RdYlGn,
               origin='lower', # extent=[-3, 3, -3, 3],
               vmax=abs(Z).max(), vmin=abs(Z).min())

plt.show()

""" MODE """
Z = round(df_mode['mode_test_auc']*100, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
p = ax.scatter3D(df_mode['mode_tree_depth'], df_mode['mode_n_trees'], zs=Z, c=Z, cmap='coolwarm', s=100)
cbar = fig.colorbar(p, ax=ax, shrink=0.5, aspect=10)
cbar.set_ticks(np.arange(0.6, 1.0, 0.2))
plt.show()

# Z = round(df_mode['mode_test_auc']*100, 1)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# p = ax.scatter3D(df_mode['mode_tree_depth'], df_mode['mode_n_trees'], zs=Z, c=Z, cmap='coolwarm', s=100)
# fig.colorbar(p, ax=ax, shrink=0.5, aspect=10)
# plt.show()

# MODE TEST AUROC
# font = 16
# fig_1 = plt.figure(figsize=(8, 5))
# ax_1 = Axes3D(fig_1)
# Z = round(df_mode['mode_test_auc']*100, 1)
# surf_1 = ax_1.plot_trisurf(df_mode['mode_tree_depth'], df_mode['mode_n_trees'], Z, cmap=cm.coolwarm, linewidth=0.2,
#                        # vmin=auroc_vmin,
#                        # vmax=auroc_vmax
#                        )
# ax_1.set_title('Bitrate Mode Predictor Mean AUROC Test set', fontsize=font+1)
# ax_1.set_ylabel('No. of trees', fontsize=font-4)
# ax_1.set_xlabel('Max. tree depth', fontsize=font-4)
# ax_1.set_zlabel('AUROC [%]', fontsize=font - 4)
# elev = 15
# azim = -45
# ax_1.view_init(elev, azim)
# fig_1.colorbar(surf_1, shrink=0.5, aspect=10)


# # MODE TEST ACC
# font = 16
# fig_2 = plt.figure(figsize=(8, 5))
# ax_2 = Axes3D(fig_2)
# Z = round(df_mode['test_acc']*100, 1)
# surf_2 = ax_2.plot_trisurf(df_mode['mode_tree_depth'], df_mode['mode_n_trees'], Z, cmap=cm.GnBu, linewidth=0.2,
#                        # vmin=acc_vmin,
#                        # vmax=acc_vmax
#                        )
# ax_2.set_title('Bitrate Mode Predictor Accuracy Test set', fontsize=font+1)
# ax_2.set_ylabel('No. of trees', fontsize=font-4)
# ax_2.set_xlabel('Max. tree depth', fontsize=font-4)
# ax_2.set_zlabel('Accuracy [%]', fontsize=font - 4)
# elev = 15
# azim = -45
# ax_2.view_init(elev, azim)
# fig_2.colorbar(surf_2, shrink=0.5, aspect=10)
# # plt.show()
#
# """ NSEG """
# # NSEG TEST ACC
# font = 16
# fig_3 = plt.figure(figsize=(8, 5))
# ax_3 = Axes3D(fig_3)
# Z = round(df_nseg['nseg_test_auc']*100, 1)
# surf_3 = ax_3.plot_trisurf(df_nseg['nseg_tree_depth'], df_nseg['nseg_n_trees'], Z, cmap=cm.coolwarm, linewidth=0.2,
#                        # vmin=auroc_vmin,
#                        # vmax=auroc_vmax
#                        )
# ax_3.set_title('No. Segments Predictor Mean AUROC Test set', fontsize=font+1)
# ax_3.set_ylabel('No. of trees', fontsize=font-4)
# ax_3.set_xlabel('Max. tree depth', fontsize=font-4)
# ax_3.set_zlabel('AUROC [%]', fontsize=font - 4)
# elev = 15
# azim = -45
# ax_3.view_init(elev, azim)
# cbar = fig_3.colorbar(surf_3, shrink=0.5, aspect=10)
# # plt.show()
#
# # NSEG TRAIN ACC
# font = 16
# fig_4 = plt.figure(figsize=(8, 5))
# ax_4 = Axes3D(fig_4)
# Z = round(df_nseg['test_acc']*100, 1)
# surf_4 = ax_4.plot_trisurf(df_nseg['nseg_tree_depth'], df_nseg['nseg_n_trees'], Z, cmap=cm.GnBu, linewidth=0.2,
#                        # vmin=acc_vmin,
#                        # vmax=acc_vmax
#                        )
# ax_4.set_title('No. Segments Predictor Accuracy Test set', fontsize=font+1)
# ax_4.set_ylabel('No. of trees', fontsize=font-4)
# ax_4.set_xlabel('Max. tree depth', fontsize=font-4)
# ax_4.set_zlabel('Accuracy [%]', fontsize=font - 4)
# elev = 15
# azim = -45
# ax_4.view_init(elev, azim)
# cbar = fig_4.colorbar(surf_4, shrink=0.5, aspect=10)


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

