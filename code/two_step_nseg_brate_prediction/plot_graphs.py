"""
    @description:       file for plotting metrics for a list of nodes and saves into a png file
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              22.06.2020
"""
# =============================================================================
#  Import Section
# =============================================================================

# Python lib/std-pkgs imports
import sys
import os
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Load Data Imports
import load_data as ld


def main(src_dir, save_dir):

    # Flag to load Rsrrp/rsrq
    load_rsrp = True
    # nodes to plot
    nd_toplot = [2, 5, 7, 10, 13, 15, 17, 20, 23, 25, 30, 33, 37]
    # Time and ticks parameters
    T_lower = 0.0
    T_upper = 1000.0
    # Size of the figure
    in_fsize = (300, 200)

    dir_dash_log = src_dir + "dash_client_log.txt"
    dir_DlMacStats = src_dir + "DlMacStats.txt"
    dir_UlMacStats = src_dir + "UlMacStats.txt"
    dir_DlRsrpSinrStats = src_dir + "DlRsrpSinrStats.txt"
    dir_mobility = src_dir + "mobility_trace.txt"
    dir_UlSinrStats = src_dir + "UlSinrStats.txt"
    dir_MpegLog = src_dir + "mpeg_player_log.txt"

    # Load dataframes
    df_DlMacStats = ld.load_std_log(dir_DlMacStats)
    df_UlMacStats = ld.load_std_log(dir_UlMacStats)
    # NOTE: DlRsrpSinrStats takes a lot of memory space, so only load if necessary
    if load_rsrp:
        df_DlRsrpSinrStats = ld.load_std_log(dir_DlRsrpSinrStats)
    df_UlSinrStats = ld.load_std_log(dir_UlSinrStats)

    # Additionally, change time units (microseconds to seconds)
    df_mobility = ld.load_std_log(dir_mobility, timeHead="tstamp_us")
    df_mobility['Time'] = round(df_mobility['Time'] / 1e6, 3)

    # NOTE: remove the trailing '\t' from the column headers
    df_MpegLog = ld.load_std_log(dir_MpegLog, timeHead="tstamp_us")
    df_MpegLog['Time'] = round(df_MpegLog['Time'] / 1e6, 3)
    # df_MpegLog['Node'] = df_MpegLog['Node'] + 1

    df_Dashlog = ld.load_dash_client_log(dir_dash_log, verbose=False)
    df_Dashlog['Time'] = round(df_Dashlog['Time'] / 1e6, 3)

    # =============================================================================
    #  Extract columns of interest from each dataframe
    # =============================================================================
    if load_rsrp:
        df_DlRsrpSinrStats = df_DlRsrpSinrStats[['Time', 'IMSI', 'cellId', 'rsrp']]
        df_DlRsrpSinrStats = df_DlRsrpSinrStats.rename(columns={"rsrp": "Dl-RSRP", 'IMSI': "Node"})

    df_UlSinrStats = df_UlSinrStats[['Time', 'IMSI', 'cellId', 'sinrLinear']]
    df_UlSinrStats = df_UlSinrStats.rename(columns={"sinrLinear": "Ul-SINRlinear", 'IMSI': "Node"})

    df_UlMacStats = df_UlMacStats[['Time', 'IMSI', 'mcs', 'size']]
    df_UlMacStats = df_UlMacStats.rename(columns={"IMSI": "Node", "mcs": "Ul-mcs", "size": "Ul-size"})

    # Only Tb1 chosen, since Tb2 is empty (zeros)
    df_DlMacStats = df_DlMacStats[['Time', 'IMSI', 'mcsTb1', 'sizeTb1']]
    df_DlMacStats = df_DlMacStats.rename(columns={"IMSI": "Node", "mcsTb1": "Dl-mcs", "sizeTb1": "Dl-size"})

    df_mobility = df_mobility[['Time', 'IMSI', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']]
    df_mobility = df_mobility.rename(columns={'IMSI': 'Node'})

    df_Dashlog = df_Dashlog

    df_MpegLog = df_MpegLog  # [['Time', 'Node', 'playbackTime', 'frameQueueBytes', 'frameQueueSize']]

    # =============================================================================
    #  Get grouped data
    # =============================================================================
    gp_Dashlog = df_Dashlog.groupby(['Node'])
    gp_MpegLog = df_MpegLog.groupby(['Node'])
    if load_rsrp:
        gp_Rsrp = df_DlRsrpSinrStats.groupby(['Node'])
    gp_Sinr = df_UlSinrStats.groupby(['Node'])
    gp_DlMacTbs = df_DlMacStats.groupby(['Node'])
    gp_UlMac = df_UlMacStats.groupby(['Node'])
    gp_mob = df_mobility.groupby(['Node'])

    # =============================================================================
    #  Plot data
    # =============================================================================

    print("PLOTTING INFO FOR NODES: ", nd_toplot)

    n_nds = len(nd_toplot)
    print(len(nd_toplot))

    # Create axes for the plots
    fig, axes = plt.subplots(nrows=13, ncols=n_nds, figsize=in_fsize)

    fsize = (15, 5)
    step = round((T_upper - T_lower) / 20)
    xtics = np.arange(T_lower, T_upper, step)

    for i in range(n_nds):
        node = nd_toplot[i]

        # Get the dataframes for the node

        df_node_raw = pd.DataFrame(gp_Dashlog.get_group(node))
        df_node_mpeg = pd.DataFrame(gp_MpegLog.get_group(node))

        if load_rsrp:
            df_node_rsrp = pd.DataFrame(gp_Rsrp.get_group(node))
            # Convert to dB miliwatts
            df_node_rsrp['Dl-RSRP'] = 10 * np.log10(df_node_rsrp['Dl-RSRP'] * 1000)

        df_node_sinr = pd.DataFrame(gp_Sinr.get_group(node))
        # Convert to dB
        df_node_sinr['Ul-SINRlinear'] = 10 * np.log10(df_node_sinr['Ul-SINRlinear'])

        df_node_dltbs = pd.DataFrame(gp_DlMacTbs.get_group(node))

        df_node_ulmac = pd.DataFrame(gp_UlMac.get_group(node))

        df_node_mob = pd.DataFrame(gp_mob.get_group(node))

        # Plot

        df_node_sinr[(T_lower <= df_node_sinr.Time) & (df_node_sinr.Time <= T_upper)].plot(x='Time', y='cellId',
                                                                                           style='o',
                                                                                           # kind='bar',
                                                                                           title='Cell-Id association',
                                                                                           grid=True,
                                                                                           #    figsize=fsize,
                                                                                           xticks=xtics,
                                                                                           ax=axes[0, i]
                                                                                           )

        # Plot newBitRate from dash_client_log
        df_node_raw[(T_lower <= df_node_raw.Time) & (df_node_raw.Time <= T_upper)].plot(x='Time', y='newBitRate_bps',
                                                                                        style='o',
                                                                                        # kind='bar',
                                                                                        title='DASHLOG New Bitrate',
                                                                                        grid=True,
                                                                                        #    figsize=fsize,
                                                                                        xticks=xtics,
                                                                                        ax=axes[1, i]
                                                                                        )
        # Plot bitRate from mpeg_player_log
        df_node_mpeg[(T_lower <= df_node_mpeg.Time) & (df_node_mpeg.Time <= T_upper)].plot(x='Time', y='bitRate',
                                                                                           style='o',
                                                                                           # kind='bar',
                                                                                           title='MPEG bitRate',
                                                                                           grid=True,
                                                                                           #  figsize=fsize,
                                                                                           xticks=xtics,
                                                                                           ax=axes[2, i]
                                                                                           )

        df_node_raw[(T_lower <= df_node_raw.Time) & (df_node_raw.Time <= T_upper)].plot(x='Time',
                                                                                        y='thputOverLastSeg_bps',
                                                                                        title='DASHLOG Throughput over '
                                                                                              'last Segment',
                                                                                        style='o',
                                                                                        grid=True,
                                                                                        #  figsize=fsize,
                                                                                        xticks=xtics,
                                                                                        ax=axes[3, i]
                                                                                        )

        df_node_raw[(T_lower <= df_node_raw.Time) & (df_node_raw.Time <= T_upper)].plot(x='Time',
                                                                                        y='avgThputOverWindow_bps'
                                                                                          '(estBitRate)',
                                                                                        title='DASHLOG Average '
                                                                                              'Throughput',
                                                                                        grid=True,
                                                                                        # kind='scatter',
                                                                                        style='o',
                                                                                        # figsize=fsize,
                                                                                        xticks=xtics,
                                                                                        ax=axes[4, i]
                                                                                        )

        df_node_raw[(T_lower <= df_node_raw.Time) & (df_node_raw.Time <= T_upper)].plot(x='Time', y='frameQueueBytes',
                                                                                        style='o',
                                                                                        title='DASHLOG FrameQueue Bytes',
                                                                                        grid=True,
                                                                                        # figsize=fsize,
                                                                                        xticks=xtics,
                                                                                        ax=axes[5, i]
                                                                                        )

        df_node_mpeg[(T_lower <= df_node_mpeg.Time) & (df_node_mpeg.Time <= T_upper)].plot(x='Time',
                                                                                           y='frameQueueBytes',
                                                                                           style='o',
                                                                                           # kind='bar',
                                                                                           title='MPEGLOG frameQueueBytes',
                                                                                           grid=True,
                                                                                           #  figsize=fsize,
                                                                                           xticks=xtics,
                                                                                           ax=axes[6, i]
                                                                                           )

        df_node_raw[(T_lower <= df_node_raw.Time) & (df_node_raw.Time <= T_upper)].plot(x='Time', y='frameQueueSize',
                                                                                        style='o',
                                                                                        title='DASHLOG FrameQueue Size',
                                                                                        grid=True,
                                                                                        # figsize=fsize,
                                                                                        xticks=xtics,
                                                                                        ax=axes[7, i]
                                                                                        )

        df_node_mpeg[(T_lower <= df_node_mpeg.Time) & (df_node_mpeg.Time <= T_upper)].plot(x='Time', y='frameQueueSize',
                                                                                           style='o',
                                                                                           # kind='bar',
                                                                                           title='MPEGLOG frameQueueSize',
                                                                                           grid=True,
                                                                                           #  figsize=fsize,
                                                                                           xticks=xtics,
                                                                                           ax=axes[8, i]
                                                                                           )
        if load_rsrp:
            df_node_rsrp[(T_lower <= df_node_rsrp.Time) & (df_node_rsrp.Time <= T_upper)].plot(x='Time', y='Dl-RSRP',
                                                                                               style='o',
                                                                                               title='RSRP',
                                                                                               grid=True,
                                                                                               #  figsize=fsize,
                                                                                               xticks=xtics,
                                                                                               ax=axes[9, i]
                                                                                               )

        df_node_sinr[(T_lower <= df_node_sinr.Time) & (df_node_sinr.Time <= T_upper)].plot(x='Time', y='Ul-SINRlinear',
                                                                                           style='o',
                                                                                           title='SINR',
                                                                                           grid=True,
                                                                                           #  figsize=fsize,
                                                                                           xticks=xtics,
                                                                                           ax=axes[10, i]
                                                                                           )

        df_node_dltbs[(T_lower <= df_node_dltbs.Time) & (df_node_dltbs.Time <= T_upper)].plot(x='Time', y='Dl-size',
                                                                                              style='o',
                                                                                              title='DL MAC TBS',
                                                                                              grid=True,
                                                                                              #  figsize=fsize,
                                                                                              xticks=xtics,
                                                                                              ax=axes[11, i]
                                                                                              )

        df_node_dltbs[(T_lower <= df_node_dltbs.Time) & (df_node_dltbs.Time <= T_upper)].plot(x='Time', y='Dl-mcs',
                                                                                              style='o',
                                                                                              title='DL MAC mcs-Tb1',
                                                                                              grid=True,
                                                                                              # figsize=fsize,
                                                                                              xticks=xtics,
                                                                                              ax=axes[12, i]
                                                                                              )
    fig.savefig(save_dir, format="pdf")

    print("All plots saves under ", save_dir)


if __name__ == "__main__":
    # =============================================================================
    #  Load data
    # =============================================================================
    # Specify the paths
    # read path
    try:
        src_dir = sys.argv[1]
    except IndexError:
        src_dir = "/home/daniel/Documents/00_DNA/DASH/data/scenario4_varyNumUes/"

    # write path
    save_dir = "./data_preproc_log/nodes_plot_all.pdf"

    mp = Pool(processes=4)
    mp.starmap(main, [[src_dir, save_dir]])
