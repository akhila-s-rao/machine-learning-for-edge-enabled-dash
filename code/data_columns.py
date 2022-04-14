"""
    @description:       file specifying list of strings containing columns of interest
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              22.06.2020
"""

BITRATE_LIST = [1000000., 2500000., 5000000., 8000000., 16000000., 35000000.]
BITRATE_INT_DICT = {k:v for k,v in zip(BITRATE_LIST,range(len(BITRATE_LIST)))}

FEAT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps',
             'mean-BufferBytes', 'mean-BufferSize', 
             'mean-Dl-rsrp', 'mean-Dl-sinr', 
             'mean-Dl-mcs', 'mean-Dl-thput']
             #'25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp',
             #'25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr',
             #'25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput',         
            
RAN_FEAT_COLS = ['mean-Dl-mcs', 'mode-Dl-mcs', 'mean-Dl-thput', '25q-Dl-thput', '50q-Dl-thput', '75q-Dl-thput', '90q-Dl-thput', 'mean-Dl-rsrp', '25q-Dl-rsrp', '50q-Dl-rsrp', '75q-Dl-rsrp', '90q-Dl-rsrp', 'mean-Dl-sinr', '25q-Dl-sinr', '50q-Dl-sinr', '75q-Dl-sinr', '90q-Dl-sinr']

DASH_FEAT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps', 'mean-BufferBytes', '25q-BufferBytes', '50q-BufferBytes', '75q-BufferBytes', '90q-BufferBytes', 'mean-BufferSize', '25q-BufferSize', '50q-BufferSize', '75q-BufferSize', '90q-BufferSize']

TARGET_SEGMODE = ['mode-HorznBitRates']
TARGET_SEGNUMBER = ['n-HorznBitRates']
TARGET_CELLIDASSOC = ['predCellId']