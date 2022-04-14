"""
    @description:       File for generating training data
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              24.06.2020
"""

# =============================================================================
#  This code takes the raw logs from ns3 which consist of RAN logs as well as
#  the dash logs from the djvergad dash code module and generates a structured
#  dataset by aggregating metrics into windows. The datset is saved for use 
#  by the learning algorithms
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

import load_data as ld
from rawDataHandler import RawDataHandler, NodeDfStore

def gen_train_data(in_path, out_dir, out_file, pred_wsize, agg_wsize, sim_time):
    dir_dash_log = in_path + "dash_client_log.txt"
    dir_DlMacStats = in_path + "DlMacStats.txt"
    dir_DlRsrpSinrCellIdStats = in_path + "DlRsrpSinrStats.txt"
    dir_mobility = in_path + "mobility_trace.txt"
    dir_MpegLog = in_path + "mpeg_player_log.txt"
    dir_ParamSettings = in_path + "parameter_settings.txt"
    print(in_path)
    #  Create data handler object to perform merging
    dataHandler = RawDataHandler(dir_DlMacStats, dir_DlRsrpSinrCellIdStats, dir_mobility,
                                 dir_dash_log, dir_MpegLog, dir_ParamSettings,
                                 in_wsize_metricaggreg=agg_wsize, in_wsize_predhorizon=pred_wsize,
                                 in_simTime=sim_time, verbose=False)
    dataHandler.create_and_save_windowed_data(out_dir, out_file, verbose=False)


if __name__ == "__main__":
    
    pred_wsize = 6
    agg_wsize = 24
    sim_time = 1000.0 # seconds
    read_dir = "/home/shared_data/dash/raw_data/dataset7_35Mbps_max_brate_withCa/"
    data_set_types = ['train', 'eval']
    eval_runs = [6, 12, 21, 27]
    train_runs = [i for i in range(1, 28) if i not in eval_runs]
    
    for data_set_type in data_set_types:
        print('START '+ data_set_type +' data generation')
        proc_data = []
        out_dir = '../data/data_'+data_set_type+'/modFeat3_dataset7-'+str(pred_wsize)+'sWsize-'+str(agg_wsize)+'aggsize/'
        subfolders = ['run' + str(i) for i in train_runs]
        print("Runs to generate dataset from: ", subfolders)

        for run in subfolders:
            read_path = read_dir + run + "/"
            out_file = run + "wsize" + str(int(pred_wsize)) + ".csv"
            proc_data.append([read_path, out_dir, out_file, pred_wsize, agg_wsize, sim_time])

        mp = Pool(processes=10)
        mp.starmap(gen_train_data, proc_data)
        print('END '+ data_set_type +' data generation')
