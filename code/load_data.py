"""
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              20.05.2020
"""

import pandas as pd
import re
import os
import pickle
import numpy as np
import random

def load_std_log(in_path, timeHead="% time"):
    """
    Loads a .txt without overhead, separated by tab, into a pandas DataFrame.
    :param in_path: (string) path to .txt
    :param timeHead: (string) header of the time column
    :return: (pd.DataFrame) DataFrame
    """
    df_tmp = pd.read_csv(in_path, sep='\t')
    df_tmp = df_tmp.rename(columns={timeHead: "Time"})
    # df_tmp['Time'] = df_tmp['Time'].astype(float)
    df_tmp['Time'] = round(df_tmp['Time'], 3)
    return df_tmp


def load_dash_client_log(in_path, verbose=False):
    """
    Loads a .txt with overhead, separated by tab, into a pandas DataFrame.
    The .txt overhead is ignored until the line starting with 'tstamp' is found, then it reads as a csv until EOF
    :param in_path: (string) path to .txt
    :return: (pd.DataFrame) DataFrame
    """
    n_lines = 0
    r_start = 0
    r_end = 0
    with open(in_path, 'r') as f:
        # Read file until table start
        for line in f:
            if 'tstamp' in line.strip():  # table column headers start with 'tstsamp' string
                r_start = n_lines
                n_lines += 1
            elif 'ns3::' in line.strip():
                r_end = n_lines
                break
            else:
                n_lines += 1
    if verbose:
        print("Line number for HEADER: ", r_start)
        print("Line number for ENDOFDATA: ", r_end)
    if r_end < 1:
        r_end = n_lines

    with open(in_path, 'r') as f:
        # Get whole content
        tmp_content = f.readlines()[r_start:r_end]

        # Get clean column names
        tmp_fieldnames = tmp_content.pop(0)
        tmp_fieldnames = re.sub(r'[\n %]+', '', tmp_fieldnames)
        tmp_fieldnames = re.split(r'[\t]+', tmp_fieldnames.strip('\t'))
        # clean table lines
        for i in range(len(tmp_content)):
            tmp_content[i] = re.sub(r'[\n]+', '', tmp_content[i])
            tmp_content[i] = re.split(r'[\t]+', tmp_content[i].strip('\t'))
        # Create DataFrame
        df_tmp = pd.DataFrame(tmp_content, columns=tmp_fieldnames)
        df_tmp = df_tmp.astype(float)
        df_tmp = df_tmp.rename(columns={"tstamp_us": "Time"})
        df_tmp['Time'] = df_tmp['Time'].astype(int)
        df_tmp['Node'] = df_tmp['Node'].astype(int)
        df_tmp['delayToNxtReq_s'] = df_tmp['delayToNxtReq_s'].astype(float)
    return df_tmp


def get_base_station_coordinates_dict(in_path, verbose=False):
    n_lines = 0
    r_start = 0
    r_end = 0
    with open(in_path, 'r') as f:
        # Read file until table start
        for line in f:
            if 'cellid' in line.strip() and r_end < 1:  # table column headers start with 'tstsamp' string
                r_start = n_lines
                n_lines += 1
            elif 'Locations' in line.strip() and r_start > 0:
                r_end = n_lines
                break
            else:
                n_lines += 1
    if verbose:
        print("Line number for HEADER: ", r_start)
        print("Line number for ENDOFDATA: ", r_end)

    # Dict to output base station coordinates
    gnb_pos = []

    with open(in_path, 'r') as f:
        # Get whole content
        tmp_content = f.readlines()[(r_start+1):r_end]

        # clean table lines
        for i in range(len(tmp_content)):
            tmp_content[i] = re.sub(r'[\n]+', '', tmp_content[i])
            tmp_content[i] = re.split(r'[\t]+', tmp_content[i].strip('\t'))
            gnb_pos.append(np.array([float(tmp_content[i][1]), float(tmp_content[i][2]), float(tmp_content[i][3])]))
    return np.asarray(gnb_pos)


def get_subdirectories(in_path):
    """
    Get all child folders of in_path
    :param in_path: (str) - path
    :return: (list of str) name of subdirectories
    """
    # TODO: modify
    dirs = [x[0] for x in os.walk(in_path)]
    return dirs


def get_files_in_subdirectory(in_dir, file_type='.csv', verbose=False):
    """
    Get all files in in_dir ending with file_type
    :param in_dir: (str) - path to search for files
    :param file_type: (str) - file ending, e.g. '.csv'
    :return: (list of str) list of files
    """
    files = os.listdir(in_dir)
    files = list(filter(lambda f: f.endswith(file_type), files))
    if verbose:
        print("The list of ", file_type, "-files in ", in_dir, " is: ", files)
    return files


def load_rf_trainer_data(src_path, verbose=False, slice=0.0):
    """
    Load the training data for use in training the RF models
    :param src_path: (str) - path where all the training .csvs are located
    :param verbose: (bool) - print info flag
    :return: (pd.DataFrame) merged DataFrame containing all the training data csvs
    """
    # load the files
    files = get_files_in_subdirectory(src_path, verbose=verbose)
    #print("Remove me later")
    #files = [files[1]]
    #print(files) if verbose

    assert files, "### ERROR: ld:load_rf_trainer_data: no csv files in that subdirectory!"

    dfs_list = []

    # Select a subset of the files
    if slice > 0.001:
        random.shuffle(files)
        files = files[0:round(len(files) * slice)]

    for file in files:
        tmp_dir = src_path + "/" + file
        tmp_df = pd.read_csv(tmp_dir)
        dfs_list.append(tmp_df)

    assert len(dfs_list) > 0, "### ERROR: ld:load_rf_trainer_data: dataframes could not be loaded!"

    ret_df = dfs_list.pop(0)

    if len(dfs_list) > 0:
        while len(dfs_list) > 0:
            tmp_df = dfs_list.pop(0)
            ret_df = ret_df.append(tmp_df)

    return ret_df