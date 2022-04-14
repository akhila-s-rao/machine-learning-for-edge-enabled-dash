"""
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              30.05.2020
"""

import pandas as pd
import numpy as np
import math
import load_data as ld
from data_columns import *
import os

"""
    Resulting columns for Dashlog after merging (2020.06.10)
"""

class NodeDfStore:
    """
        Helper class to store DataFrames per user
    """
    def __init__(self):
        self.df_merged = None
        self.DfWsizeAnalysis = None
        
        self.Tnode_start = None
        self.Tnode_end = None
        pass


class RawDataHandler:
    def __init__(self,
                 dir_DlMacStats, dir_DlRsrpSinrCellIdStats, dir_mobility,
                 dir_Dashlog, dir_MpegLog,
                 dir_ParamSettings,
                 in_wsize_metricaggreg = 6.0,
                 in_wsize_predhorizon = 3.0,
                 in_simTime = 1000.0,
                 verbose=False):
        """
            Get the paths to the .txt source files. 
            Files cannot be loaded all at once due to possible memory constraints
        """
        self.dir_DlMacStats = dir_DlMacStats
        self.dir_DlRsrpSinrStats = dir_DlRsrpSinrCellIdStats
        self.dir_mobility = dir_mobility
        self.dir_Dashlog = dir_Dashlog
        self.dir_Mpeglog = dir_MpegLog
        self.dir_ParamSettings = dir_ParamSettings

        # Simulation Time in seconds
        self.simTime = in_simTime
        self.wsize_metricsaggreg = in_wsize_metricaggreg
        self.wsize_predhorizon = in_wsize_predhorizon

        # Create dict to store node <-> NodeDfStore maps
        self.nodeDfsMap = dict()

        # Store nodes present in DashLog that compose the training data
        self.nodes = list()
        
    def _get_wsize_time_init_param(self):
        window_check = False

        if self.wsize_metricsaggreg and self.wsize_predhorizon:
            window_check = True

        assert window_check, "ERRROR: RawDataHandler:_get_wsize_time_init_param: at least one window size is empty!"

        # Set time parameters
        Tagg_end = self.wsize_predhorizon
        Tagg_start = Tagg_end - self.wsize_metricsaggreg

        Thor_start = Tagg_end
        Thor_end = Thor_start + self.wsize_predhorizon

        return Tagg_start, Tagg_end, Thor_start, Thor_end

    def create_and_save_windowed_data(self, out_dir, out_file, verbose=False, time_end_threshold=900.0):
        """
        Function performs individual sliding windows operations across all raw logs and stores one DataFrame
        for each node in the dict self.nodeDfsMap
        :param verbose: choose to print messages or not
        :param time_end_threshold: (double) nodes that their end time is lower than this value are discarted
        :return: none
        """        
        df_Dashlog = ld.load_dash_client_log(self.dir_Dashlog)
        df_Dashlog['Time'] = round(df_Dashlog['Time'] / 1e6, 3)
        df_Dashlog['Time'] = df_Dashlog['Time'] + df_Dashlog['delayToNxtReq_s']
        gp_Dashlog = df_Dashlog.groupby(['Node'])

        df_MpegLog = ld.load_std_log(self.dir_Mpeglog, timeHead="tstamp_us")
        df_MpegLog['Time'] = round(df_MpegLog['Time'] / 1e6, 3)
        gp_MpegBuffer = df_MpegLog.groupby(['Node'])

        df_DlRsrpSinrStats = ld.load_std_log(self.dir_DlRsrpSinrStats)
        df_DlRsrpSinrStats = df_DlRsrpSinrStats[['Time', 'IMSI', 'rsrp', 'sinr']]
        df_DlRsrpSinrStats = df_DlRsrpSinrStats.rename(columns={"rsrp": "Dl-RSRP", 'IMSI': "Node", "sinr": "Dl-SINR"})
        gp_DlRsrpSinrStats = df_DlRsrpSinrStats.groupby(['Node'])

        df_DlMacStats = ld.load_std_log(self.dir_DlMacStats)
        df_DlMacStats = df_DlMacStats[['Time', 'IMSI', 'mcsTb1', 'sizeTb1']]
        df_DlMacStats = df_DlMacStats.rename(columns={"IMSI": "Node", "mcsTb1": "Dl-mcs", "sizeTb1": "Dl-size"})
        gp_DlMacStats = df_DlMacStats.groupby(['Node'])
        
        thput_miniwindow = 1.0

        gnb_pos = ld.get_base_station_coordinates_dict(self.dir_ParamSettings)
        df_mobility = ld.load_std_log(self.dir_mobility, timeHead="tstamp_us")
        df_mobility['Time'] = round(df_mobility['Time'] / 1e6, 3)
        df_mobility = df_mobility[['Time', 'IMSI', 'cellID', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']]
        df_mobility = df_mobility.rename(columns={'IMSI': 'Node'})
        gp_mobility = df_mobility.groupby(['Node'])

        # Iterate over each node
        for nd in gp_Dashlog.groups.keys():
            #print('Node ' + str(nd))
            # Select corresponding dataframe for the node
            df_node_raw_dash = pd.DataFrame(gp_Dashlog.get_group(nd))
            df_node_raw_mpeg = pd.DataFrame(gp_MpegBuffer.get_group(nd))
            df_node_raw_dl_signal = pd.DataFrame(gp_DlRsrpSinrStats.get_group(nd))
            df_node_raw_dl_mac = pd.DataFrame(gp_DlMacStats.get_group(nd))
            df_node_raw_mob = pd.DataFrame(gp_mobility.get_group(nd))
            
            # Convert signal to dBm
            df_node_raw_dl_signal['Dl-RSRP'] = 10 * np.log10(df_node_raw_dl_signal['Dl-RSRP'] * 1000)
            df_node_raw_dl_signal['Dl-SINR'] = 10 * np.log10(df_node_raw_dl_signal['Dl-SINR'])
            
            df_node_raw_dash = df_node_raw_dash.sort_values(['Time'])
            Tnode_last = df_node_raw_dash['Time'].iloc[-1]
            print(self.dir_Dashlog, ' Tnode_last: ', Tnode_last)       
            # Add node to self
            if nd not in self.nodes:
                self.nodes.append(nd)

            # Create DataFrame to store pre-processed values of nd DataFrame
            # consists of the same values + mean and quantile for rsrp
            cols = ['Node', 'Tagg_start', 'Tagg_end', 'Thor_start', 'Thor_end',
                    'n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps',
                    'n-HorznBitRates', 'mode-HorznBitRates', # the future window (y)
                    'last-SegId', 'video-Id',
                    'mean-BufferBytes', 
                    'mean-BufferSize',
                    'mean-Dl-rsrp', 
                    #'25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp',
                    'mean-Dl-sinr', 
                    #'25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr',
                    'mean-Dl-mcs',
                    'mean-Dl-thput', 
                    #'25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput',
                    'predCellId']
            df_node_window = pd.DataFrame(columns=cols)
            
            # Set time parameters
            Tagg_start, Tagg_end, Thor_start, Thor_end = self._get_wsize_time_init_param()
            #print('Time windows')
            #print(Tagg_start, Tagg_end, Thor_start, Thor_end)
            waiting_to_start = True
            # Variable to keep track of cellId, in case in one prediction window there is no cellId value
            # initialize
            cellId_cached = 0            
            cached_buff_bytes = 0
            cached_buff_size = 0
            cached_rsrp_mean = 0
            #cached_rsrp_q = [0, 0, 0, 0]
            cached_sinr_mean = 0
            #cached_sinr_q = [0, 0, 0, 0]
            cached_mean_dl_mcs = 0
            
            # while loop over time windows
            while (Thor_end < Tnode_last):
                #print('Thor_end: ',Thor_end, 'Tnode_last: ',Tnode_last)                    
                if waiting_to_start:
                    dash_start_wind = df_node_raw_dash[(df_node_raw_dash.Time >= Tagg_start) & (df_node_raw_dash.Time < Tagg_end)]
                    # if this window is non empty and it has atleast the second segment     
                    if (len(dash_start_wind) > 0): # there is atleast one segment in the aggregation window
                        #and (tmp_dash_df['segmentId'].iloc[-1] > 1) :
                        # hurray we can start 
                        waiting_to_start = False
                        #print('Found first segemnt')
                        # I can now go ahead and execute the rest of the code 
                        # I will not enter this condition again 
                    else: # increment window and keep waiting 
                        Thor_start += self.wsize_predhorizon
                        Thor_end += self.wsize_predhorizon
                        Tagg_start += self.wsize_predhorizon
                        Tagg_end += self.wsize_predhorizon
                        continue

                # Get DataFrame for given time frame
                tmp_dash_df = df_node_raw_dash[(df_node_raw_dash.Time >= Tagg_start) & (df_node_raw_dash.Time < Tagg_end)]
                tmp_mpeg_df = df_node_raw_mpeg[(df_node_raw_mpeg.Time >= Tagg_start) & (df_node_raw_mpeg.Time < Tagg_end)].copy()
                tmp_mpeg_df.reset_index(inplace=True)
                tmp_dl_signal_df = df_node_raw_dl_signal[(df_node_raw_dl_signal.Time >= Tagg_start) & (df_node_raw_dl_signal.Time < Tagg_end)]
                tmp_dl_mac_df = df_node_raw_dl_mac[(df_node_raw_dl_mac.Time >= Tagg_start) & (df_node_raw_dl_mac.Time < Tagg_end)]
                # cellID NOTE: this cellId should be mode in the prediction window
                tmp_cellId_df = df_node_raw_mob[(df_node_raw_mob.Time >= Thor_start) & (df_node_raw_mob.Time < Thor_end)]
                if len(tmp_cellId_df):
                    cellId_cached = np.sort(tmp_cellId_df['cellID'].mode().to_numpy())[-1]
                
                # if atleast one dash log sample in this aggregation window
                # I only want to process windows that have a segment in the aggregation window 
                # because for the others I want to propagate the previous window value 
                if len(tmp_dash_df) > 0:
                    n_agg_bitrates = len(tmp_dash_df['newBitRate_bps'])
                    #mean_AggBitRates = ser_agg_bitrate.mean()
                    #mean_AggBitRates = np.sort(ser_agg_bitrate.mode().to_numpy())[-1]
                    # the last segment sent out 
                    previous_BitRate = tmp_dash_df['newBitRate_bps'].iloc[-1]
                    # GET Throughput over last segment mean
                    thPutLastSeg_mean = tmp_dash_df['thputOverLastSeg_bps'].mean()
                    # GET the last segment quality requested (for implementation of predictor to keep track of seg Id and vid id)
                    last_seg_id = tmp_dash_df['segmentId'].iloc[-1]
                    vid_id = tmp_dash_df['videoId'].iloc[-1]
                    # GET average throughput over window
                    avgThPutWindow_mean = tmp_dash_df['avgThputOverWindow_bps(estBitRate)'].mean()
                    #-----------------------------------
                    if len(tmp_mpeg_df) == 0:
                        print('EMPTY MPEG window')
                        latest_buff_bytes = cached_buff_bytes
                        latest_buff_size = cached_buff_size
                    else:
                        latest_buff_bytes = tmp_mpeg_df['frameQueueBytes'][tmp_mpeg_df.shape[0]-1]
                        latest_buff_size = tmp_mpeg_df['frameQueueSize'][tmp_mpeg_df.shape[0]-1]
                        cached_buff_bytes = latest_buff_bytes
                        cached_buff_size = latest_buff_size
                    #------------------------------------
                    if len(tmp_dl_signal_df) == 0:
                        print('EMPTY DL SIGNAL window')
                        rsrp_mean = cached_rsrp_mean
                        #rsrp_q = cached_rsrp_q
                        # SINR metrics
                        sinr_mean = cached_sinr_mean
                        #sinr_q = cached_sinr_q
                    else:
                        # RSRP metrics
                        rsrp_mean = tmp_dl_signal_df['Dl-RSRP'].mean()
                        cached_rsrp_mean = rsrp_mean
                        #rsrp_q = tmp_dl_signal_df['Dl-RSRP'].quantile([.25, .50, .75, .90])
                        #cached_rsrp_q = rsrp_q.copy()
                        # SINR metrics
                        sinr_mean = tmp_dl_signal_df['Dl-SINR'].mean()
                        cached_sinr_mean = sinr_mean
                        #sinr_q = tmp_dl_signal_df['Dl-SINR'].quantile([.25, .50, .75, .90])
                    #-----------------------------------
                    if len(tmp_dl_mac_df) == 0:
                        print('EMPTY DL MAC window')
                        mean_dl_mcs = cached_mean_dl_mcs
                        mean_dl_thput = 0
                        #q_dl_thput = [0, 0, 0, 0]
                    else: 
                        mean_dl_mcs = tmp_dl_mac_df['Dl-mcs'].mean()
                        cached_mean_dl_mcs = mean_dl_mcs
                        Tphys_start = Tagg_start
                        Tphys_end = Tphys_start + thput_miniwindow
                        # List to store temporal phys-lay thput over miniwindow jumps in [Tagg_start, Tagg_end)
                        l_phys_thputs = []
                        while Tphys_end <= Tagg_end:
                            if verbose: print("Tphys_start: ", Tphys_start, ", Tphys_end: ", Tphys_end)
                            # Get data transmitted in miniwindow
                            tmp_data = tmp_dl_mac_df['Dl-size'][(tmp_dl_mac_df.Time >= Tphys_start)
                                                              & (tmp_dl_mac_df.Time < Tphys_end)].sum()
                            if tmp_data > 0:
                                tmp_thput = tmp_data / thput_miniwindow
                                l_phys_thputs.append(tmp_thput)
                            # Update miniwindow ranges
                            Tphys_start += thput_miniwindow
                            Tphys_end += thput_miniwindow

                        np_dl_thput = np.asarray(l_phys_thputs)
                        mean_dl_thput = np.mean(np_dl_thput)
                        #q_dl_thput = [np.quantile(np_dl_thput, 0.25), np.quantile(np_dl_thput, 0.50), 
                        #              np.quantile(np_dl_thput, 0.75), np.quantile(np_dl_thput, 0.90)]
                    
                    # ADD ROW to DataFrame
                    k = len(df_node_window)
                    df_node_window.loc[k] = \
                                [nd, Tagg_start, Tagg_end, Thor_start, Thor_end,
                                 n_agg_bitrates, previous_BitRate, thPutLastSeg_mean, avgThPutWindow_mean,
                                 0, 0, # horzn values. fill these outside the loop 
                                 last_seg_id, vid_id,
                                 latest_buff_bytes, 
                                 latest_buff_size,  
                                 rsrp_mean, 
                                 #rsrp_q[0], rsrp_q[1], rsrp_q[2], rsrp_q[3], 
                                 sinr_mean, 
                                 #sinr_q[0], sinr_q[1], sinr_q[2], sinr_q[3], 
                                 mean_dl_mcs,
                                 mean_dl_thput, 
                                 #q_dl_thput[0], q_dl_thput[1], q_dl_thput[2], q_dl_thput[3], 
                                 cellId_cached]
                    
                # empty aggregation window in dash log
                else:
                    # ffill what we had for previous aggr window
                    k = len(df_node_window)
                    df_node_window.loc[k] = df_node_window.loc[k-1].copy()                    

                # Horizon window
                # even though I ffill for empty aggregation windows
                # I need to fill the latest values for the horizon window 
                ser_horzn_bitrates = df_node_raw_dash['newBitRate_bps'][(df_node_raw_dash.Time >= Thor_start) &
                                                                   (df_node_raw_dash.Time < Thor_end)]
                df_node_window.loc[k]['n-HorznBitRates'] = len(ser_horzn_bitrates)
                if len(ser_horzn_bitrates) > 0:
                    # keep only the first segment. Because we would have fetched only one segment of predicted brate 
                    df_node_window.loc[k]['mode-HorznBitRates'] = ser_horzn_bitrates.to_numpy()[0]
                else:
                    df_node_window.loc[k]['mode-HorznBitRates'] = 0
                                                           
                # Update times
                Tagg_start += self.wsize_predhorizon
                Tagg_end += self.wsize_predhorizon
                Thor_start += self.wsize_predhorizon
                Thor_end += self.wsize_predhorizon
            
            # Add created dataframe to node_data dict as an element
            try:
                self.nodeDfsMap[nd].df_merged = df_node_window.fillna(0)
                #self.nodeDfsMap[nd].Tnode_end = Tnode_last
            except KeyError:
                # Add a new element to the dict if not present already:
                self.nodeDfsMap[nd] = NodeDfStore()
                self.nodeDfsMap[nd].df_merged = df_node_window.fillna(0)
                #self.nodeDfsMap[nd].Tnode_end = Tnode_last  
                
        comb_df = pd.concat([self.nodeDfsMap[elem].df_merged for elem in self.nodeDfsMap.keys()])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file = open(out_dir+out_file, 'w')
        comb_df.to_csv(file, index=False)
        file.close()