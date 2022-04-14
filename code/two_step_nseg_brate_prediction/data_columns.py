"""
    @description:       file specifying list of strings containing columns of interest
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              22.06.2020
"""

BITRATE_LIST = [1000000., 2500000., 5000000., 8000000., 16000000., 35000000.]
#BITRATE_LIST = np.array([1000000., 2500000., 5000000., 8000000., 16000000., 35000000.])
BITRATE_INT_DICT = {k:v for k,v in zip(BITRATE_LIST,range(len(BITRATE_LIST)))}

FEAT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps',
             'mean-BufferBytes', 'mean-BufferSize', 
             'mean-Dl-rsrp', 'mean-Dl-sinr', 
             'mean-Dl-mcs', 'mean-Dl-thput']
             #'25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp',
             #'25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr',
             #'25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput',











INNERJOIN_COLS = ['Node', 'Tagg_start', 'Tagg_end', 'Thor_start', 'Thor_end']

RES_DASHLOG_COLS = ['n-AggBitRates',
                    'mean-AggBitRates',
                    '25q-AggBitRates', '50q-AggBitRates', '75q-AggBitRates', '90q-AggBitrates',
                    'mean-ThPutOverLastSeg_bps',
                    'mean-avgThPutOverWindow_bps',
                    '25q-avgThPutOverWindow', '50q-avgThPutOverWindow', '75q-avgThPutOverWindow',
                    '90q-avgThPutOverWindow',
                    'n-HorznBitRates',
                    'mode-HorznBitRates']
# Columns to remove from dashlog metrics:
DASHLOG_QUANTILES = ['25q-AggBitRates', '50q-AggBitRates', '75q-AggBitRates', '90q-AggBitrates',
                     '25q-avgThPutOverWindow', '50q-avgThPutOverWindow', '75q-avgThPutOverWindow',
                     '90q-avgThPutOverWindow']
DASHLOG_SEGID_COL = ['last-SegId']
DASHLOG_VIDID_COL = ['video-Id']

RES_MPEGLOG_COLS = ['mean-BufferBytes',
                    '25q-BufferBytes', '50q-BufferBytes', '75q-BufferBytes', '90q-BufferBytes',
                    'mean-BufferSize',
                    '25q-BufferSize', '50q-BufferSize', '75q-BufferSize', '90q-BufferSize']
RES_MPEGLOG_COLS_QUANT = ['25q-BufferBytes', '50q-BufferBytes', '75q-BufferBytes', '90q-BufferBytes',
                          '25q-BufferSize', '50q-BufferSize', '75q-BufferSize', '90q-BufferSize']

RES_MOB_COLS = ['predCellId', 'mean-vel_x', 'mean-vel_y', 'mean-vel_z']
GNB_COLS_FILTER = ['predCellId']
GNB_COLS = ['gNB-' + str(i) for i in range(1, 13)]


RES_DLMAC_COLS = ['mean-Dl-mcs', 'mode-Dl-mcs',
                  'mean-Dl-thput', '25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput']

RES_DLRSRPSINRCELLID_COLS = ['mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp',
                             'mean-Dl-sinr', '25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr']
RES_DLRSRPSINRCELLID_COLS_QUANT = ['25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp',
                                   '25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr']

RES_MERGED_COLS = INNERJOIN_COLS + RES_DASHLOG_COLS + DASHLOG_SEGID_COL + DASHLOG_VIDID_COL + RES_MPEGLOG_COLS + \
                  RES_MOB_COLS + RES_DLMAC_COLS + RES_DLRSRPSINRCELLID_COLS + GNB_COLS

TARGET_COLS = ['n-HorznBitRates', 'mode-HorznBitRates', 'predCellId']

FILTER_COLS_ALLBUTBUFF = INNERJOIN_COLS + TARGET_COLS + DASHLOG_SEGID_COL + DASHLOG_VIDID_COL + GNB_COLS + RES_MOB_COLS + RES_DLRSRPSINRCELLID_COLS + GNB_COLS + RES_DASHLOG_COLS # + RES_DLMAC_COLS

TARGET_SEGMODE = ['mode-HorznBitRates']
TARGET_SEGNUMBER = ['n-HorznBitRates']
TARGET_CELLIDASSOC = ['predCellId']

RAN_FEAT_COLS = ['mean-Dl-mcs', 'mode-Dl-mcs', 'mean-Dl-thput', '25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput', 'mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp', 'mean-Dl-sinr', '25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr']
DASH_FEAT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps', 'mean-BufferBytes', '25q-BufferBytes', '50q-BufferBytes', '75q-BufferBytes', '90q-BufferBytes', 'mean-BufferSize', '25q-BufferSize', '50q-BufferSize', '75q-BufferSize', '90q-BufferSize']

FILTER_COLS = INNERJOIN_COLS + TARGET_COLS + DASHLOG_SEGID_COL + DASHLOG_VIDID_COL + GNB_COLS + DASHLOG_QUANTILES

FILTER_COLS_VEL = INNERJOIN_COLS + TARGET_COLS + DASHLOG_SEGID_COL + DASHLOG_VIDID_COL + GNB_COLS + \
                  DASHLOG_QUANTILES + ['mean-vel_x', 'mean-vel_y', 'mean-vel_z']

FILTER_COLS_MACARONI = INNERJOIN_COLS + TARGET_COLS + DASHLOG_SEGID_COL + DASHLOG_VIDID_COL + GNB_COLS \
                       + ['mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp',
                          'mean-Dl-sinr', '25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr']

FILTER_COLS_ALLQ_BUT_MAC = INNERJOIN_COLS + TARGET_COLS + DASHLOG_SEGID_COL + DASHLOG_VIDID_COL + GNB_COLS + \
                           DASHLOG_QUANTILES + RES_MPEGLOG_COLS_QUANT + RES_MOB_COLS + RES_DLRSRPSINRCELLID_COLS_QUANT

FEATURE_COLS_GNB = RES_MOB_COLS + GNB_COLS + ['mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp',
                                              '90q-Dl-rsrp']

# RF_MODE_FEATCOLS = [x for x in RES_MERGED_COLS if not x in FILTER_COLS]
# RF_NSEG_FEATCOLS = [x for x in RES_MERGED_COLS if not x in FILTER_COLS]
#
# RES_DLRSRP_COLS = ['mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp']
#
# RES_ULMAC_COLS = ['mean-Ul-mcs', 'mode-Ul-mcs',
#                   'mean-Ul-thput', '25q-Ul-thput', '50q-Ul-thput', '75q-Ul-thput', '90q-Ul-thput']
#
# RES_ULSINR_COLS = ['mean-Ul-sinr', '25q-Ul-sinr', '50q-Ul-sinr', '75q-Ul-sinr', '90q-Ul-sinr']

# ##### FOR DEBUG PLOTTING
BUFF_BYTES_COLS = ['mean-BufferBytes',
                   '25q-BufferBytes', '50q-BufferBytes', '75q-BufferBytes', '90q-BufferBytes']
BUFF_SIZE_COLS = ['mean-BufferSize',
                  '25q-BufferSize', '50q-BufferSize', '75q-BufferSize', '90q-BufferSize']
RSRP_COLS = ['mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp']
SINR_COLS = ['mean-Dl-sinr', '25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr']
DL_MAC_THPUT_COLS = ['mean-Dl-thput', '25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput']
DASH_AVG_THPUT_COLS = ['mean-avgThPutOverWindow_bps',
                       '25q-avgThPutOverWindow', '50q-avgThPutOverWindow', '75q-avgThPutOverWindow',
                       '90q-avgThPutOverWindow']
COMBINED_COLS = ['mean-BufferBytes', 'mean-Dl-rsrp', 'mean-Dl-sinr', 'mean-Dl-thput']

FEATURE_PLOT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps',
       'mean-avgThPutOverWindow_bps', 'mean-BufferBytes',
       'mean-BufferSize', 'mean-Dl-mcs', 'mean-Dl-thput',
       'mean-Dl-rsrp', 'mean-Dl-sinr']
ALL_PLOT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps',
       'mean-avgThPutOverWindow_bps', 'mean-BufferBytes',
       'mean-BufferSize', 'mean-Dl-mcs', 'mean-Dl-thput',
       'mean-Dl-rsrp', 'mean-Dl-sinr', 'n-HorznBitRates', 'mode-HorznBitRates']

if __name__ == "__main__":
    print("RES_MERGED_COLS: ", RES_MERGED_COLS)
    print("TARGET_COLS: ", TARGET_COLS)
