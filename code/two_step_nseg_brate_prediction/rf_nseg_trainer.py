"""
    @description:       File for finding the best performing RF for predicting segment mode
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              24.06.2020
"""

# =============================================================================
#  Import Section: RANDOM FOREST
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
# Load Data Imports
import load_data as ld
import train_eval_utils as utils
from data_columns import *
#RES_MERGED_COLS, TARGET_SEGNUMBER, FILTER_COLS, FILTER_COLS_MACARONI, FILTER_COLS_VEL, FILTER_COLS_ALLBUTBUFF, RAN_FEAT_COLS, DASH_FEAT_COLS
from csv import DictWriter

def run_rf_nseg(PARAMETERS):
    # ==================================================================
    # Specify paths
    # ==================================================================
    smote = False
    if smote:
        from imblearn.over_sampling import SMOTE

    # NOTE load data path
    train_data_dir = '../data/data_train/modFeat_dataset7-' \
                                + PARAMETERS['wind_size'] + 'sWsize-' \
                                + PARAMETERS['wind_size'] + 'aggsize/'
    eval_data_dir = '../data/data_eval/modFeat_dataset7-' \
                                + PARAMETERS['wind_size'] + 'sWsize-' \
                                + PARAMETERS['wind_size'] + 'aggsize/'  
    model_save_path = '../models/'+ PARAMETERS['model_type']+'/model' + str(PARAMETERS['model_save_num']) + '/'

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    print("Model save path: ", model_save_path)
    
    # Directory for saving logger and plots
    #out_path_log = model_save_path + "log/"
    #if not os.path.isdir(out_path_log):
    #    os.makedirs(out_path_log)

    # directory for saving csv log
    out_path_pdlogger = '../models/' + PARAMETERS['model_type']+ '/model_parameters.csv'
    print("CSV logger output: ", out_path_pdlogger)

    # ==================================================================
    # Load the data
    # ==================================================================
    print("Loading the data ...")
    df_train = ld.load_rf_trainer_data(train_data_dir, verbose=True)
    print("The data contains ", len(df_train), " samples.")
    # print("INF values in df_train: ", df_train.index[np.isinf(df_train).any(1)])
    df_train = df_train.astype('float32')
    df_train = df_train[~df_train.isin([np.nan, np.inf, -np.inf]).any(1)]
    # print("INF values in df_train: ", df_train.index[np.isinf(df_train).any(1)])
    print("The data contains ", len(df_train), " samples AFTER removing inf vals")
    print("Unique values in the dataset: ", df_train[TARGET_SEGNUMBER[0]].value_counts())
    # NOTE: modify according to the wize analysis plot
    # df_train['n-HorznBitRates'][df_train['n-HorznBitRates'] > 3] = 3.0
    df_train.loc[df_train[TARGET_SEGNUMBER[0]] > 3.0, TARGET_SEGNUMBER[0]] = 3.0
    print("n-HorznBitRates bigger than 3 removed since too few samples")
    # ==================================================================
    # Parameters for Random Forest exploration
    # ==================================================================
    # # Number of possible n_estimators to evaluate
    # n_n_estimators_step = 50
    # # Bounds for possible n_estimators values
    # n_estim_lowbound = 5
    # n_estim_upbound = 301
    # assert n_estim_upbound > n_estim_lowbound, "### ERROR: n_estim_upbound < n_estim_lowbound"
    # assert n_estim_lowbound < (n_estim_upbound - n_estim_lowbound), "### ERROR: n_estim_lowbound > (n_estim_upbound - " \
    #                                                                 "n_estim_lowbound)"
    # # Number of trees in the RF
    # n_estimators = np.arange(n_estim_lowbound, n_estim_upbound, step=n_n_estimators_step)
    #n_estimators = np.array([5, 15, 50, 100, 200, 350])
    n_estimators = np.array(PARAMETERS['n_trees_list'])
    print("Possible number of trees: ", n_estimators)

    # # Number of possible tree depths to evaluate
    # n_max_depth_step = 5
    # # Bounds for possible tree depths values
    # nmaxdepth_lowbound = 5
    # nmaxdepth_upbound = 60
    # assert nmaxdepth_upbound > nmaxdepth_lowbound, "### ERROR: nmaxdepth_upbound < nmaxdepth_lowbound"
    # assert n_max_depth_step < (nmaxdepth_upbound - nmaxdepth_lowbound), "### ERROR: n_max_depth > (nmaxdepth_upbound - " \
    #                                                                "nmaxdepth_lowbound)"
    # # Max depth of tree
    # max_depth = np.arange(nmaxdepth_lowbound, nmaxdepth_upbound, step=n_max_depth_step)
    #max_depth = np.array([5, 15, 25])
    max_depth = np.array(PARAMETERS['tree_depth_list'])
    print("Possible max depths of trees: ", max_depth)

    # Exploration grid
    rf_grid = [(i, j) for i in n_estimators.tolist() for j in max_depth.tolist()]

    # Number of parallel processes
    n_jobs_rf = 30

    # Test-train set split size
    split_size = 0.20

    # Cross Validation K folds
    n_Kfold_splits = 2

    n_tot_cases = n_estimators.shape[0] * max_depth.shape[0]
    print("A total of ", n_tot_cases, " cases will be evaluated.\n")

    # Number of min samples to split tree
    n_min_samples_split = 10

    # Dictionary for storing best performance tree in the test set
    rf_winner = {"logger_index": None,
                 "nseg_n_trees": 0,
                 "nseg_tree_depth": 0,
                 "nseg_auc_val": 0.0,
                 "nseg_auc_test": 0.0,
                 "nseg_acc_test": 0.0,
                 "nseg_acc_train": 0.0,
                 "nseg_rf_object": None,
                 "nseg_featcols": None,
                 "nseg_targetcol": None}

    mean_auc_val = 0.0

    max_acc_test = 0.0

    # ===============================================================
    # Load and split the Data
    # ===============================================================

    # Lets first shuffle the rows in the dataFrame
    df_train = df_train.sample(frac=1).copy()

    Y_tot = df_train[TARGET_SEGNUMBER].values
    unique, counts = np.unique(Y_tot, return_counts=True)
    print("TARGET UNIQUE VALUES: ", dict(zip(unique, counts)))
    n_classes_unique = np.unique(Y_tot)
    print("SEGMENTS nseg unique classes: ", n_classes_unique)
    n_classes = np.shape(n_classes_unique)[0]
    print("Number of unique classes: ", n_classes)

    FEAT_COLS = [x for x in RES_MERGED_COLS if not x in FILTER_COLS_VEL]
    print("FEATURE COLS before filtering: ", FEAT_COLS)
    if PARAMETERS['remove_dash_features']:
            # remove the DASH features
            FEAT_COLS = [x for x in FEAT_COLS if not x in DASH_FEAT_COLS]
    if PARAMETERS['remove_ran_features']:
            # remove the RAN features
            FEAT_COLS = [x for x in FEAT_COLS if not x in RAN_FEAT_COLS]
    if PARAMETERS['remove_prev_brate_val']:
            # remove aggr bitrate related features
                FEAT_COLS = [x for x in FEAT_COLS if not x in \
                             ['n-AggBitRates', 'mean-AggBitRates',  
                              '25q-AggBitRates', '50q-AggBitRates', 
                              '75q-AggBitRates', '90q-AggBitrates'] ]

    print("FEATURE COLS after filtering: ", FEAT_COLS) 

    Xnseg_train, Ynseg_train, Xnseg_test, Ynseg_test, stratified = utils.split_test_train(df_train, FEAT_COLS,
                                                                                          TARGET_SEGNUMBER, split_size)

    rf_winner["nseg_featcols"] = FEAT_COLS
    rf_winner["nseg_targetcol"] = TARGET_SEGNUMBER

    # if smote:
    #     print('BEFORE SMOTE dataset shape %s' % Counter(Ynseg_train[:,0]))
    #     # Oversample all but majority class
    #     sm = SMOTE(sampling_strategy='not majority', k_neighbors=10)
    #     Xnseg_train, Ynseg_train = sm.fit_resample(Xnseg_train, Ynseg_train)
    #     print('Resampled dataset shape %s' % Counter(Ynseg_train))
    # ===============================================================
    # Build-Train-Test
    # ===============================================================
    t_start = time.time()
    logger_index = 0
    for (estim, depth) in rf_grid:
        print("\nNumber of estim: ", estim, "\tMax tree depth: ", depth)

        if stratified:
            cv = StratifiedKFold(n_splits=n_Kfold_splits)
        else:
            cv = KFold(n_splits=n_Kfold_splits)

        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=estim, max_depth=depth,
                                                                # min_samples_split=n_min_samples_split,
                                                                n_jobs=n_jobs_rf))
        mean_auc_cv = []

        cv_tracker = 0

        for train, test in cv.split(Xnseg_train, Ynseg_train):

            print("Starting CV fold ", cv_tracker)

            classifier.fit(Xnseg_train[train], Ynseg_train[train])

            y_score = classifier.predict_proba(Xnseg_train[test])
            y_score[np.isnan(y_score)] = 0

            fpr_val = dict()
            tpr_val = dict()
            roc_auc_val = dict()

            roc_auc_list = []

            for i in range(n_classes):
                i_class = n_classes_unique[i]

                Y_nseg_class = np.asarray(Ynseg_train[test] == i_class, dtype=np.float)
                tmp_y_score = y_score[:, i]

                fpr_val[i_class], tpr_val[i_class], _ = roc_curve(Y_nseg_class, tmp_y_score)
                roc_auc_val[i_class] = auc(fpr_val[i_class], tpr_val[i_class])
                roc_auc_list.append(roc_auc_val[i_class])

            mean_auc_fold = np.mean(np.asarray(roc_auc_list))
            mean_auc_cv.append(mean_auc_fold)
            print("nseg mean ROC AUC on val set ", cv_tracker, " :", mean_auc_fold)
            cv_tracker += 1

        mean_auc_TRAIN = np.mean(np.asarray(mean_auc_cv))

        print("### nseg mean ROC AUC in CROSS validated set: ", mean_auc_TRAIN)
        # Calculate accuracy on train set
        y_train_tot_predclass = classifier.predict(Xnseg_train)
        acc_train = accuracy_score(Ynseg_train, y_train_tot_predclass)
        print("### NSEG ACC on TRAIN: ", acc_train, "\n")

        print("### nseg evaluating test set ... ")

        y_test_score = classifier.predict_proba(Xnseg_test)
        y_test_score[np.isnan(y_test_score)] = 0

        fpr_test = dict()
        tpr_test = dict()
        roc_auc_test = dict()

        roc_auc_list_test = []

        for i in range(n_classes):
            i_class = n_classes_unique[i]

            Y_test_class = np.asarray(Ynseg_test == i_class, dtype=np.float)
            tmp_y_test_score = y_test_score[:, i]

            fpr_test[i_class], tpr_test[i_class], _ = roc_curve(Y_test_class, tmp_y_test_score)
            roc_auc_test[i_class] = auc(fpr_test[i_class], tpr_test[i_class])
            roc_auc_list_test.append(roc_auc_test[i_class])

        tmp_test_auc = np.asarray(roc_auc_list_test)
        tmp_test_auc = np.nan_to_num(tmp_test_auc)

        mean_auc_fold_test = np.mean(tmp_test_auc)

        # Calculate accuracy on test set
        y_test_predclass = classifier.predict(Xnseg_test)
        acc_test = accuracy_score(Ynseg_test, y_test_predclass)

        conf_mat = confusion_matrix(Ynseg_test, y_test_predclass)

        print("CONFUSION MATRIX: ")
        print(conf_mat)

        print("NSEG mean AUC test set: ", mean_auc_fold_test)
        print("NSEG accuracy test set: ", acc_test)

        print("Class\t Prec \t Rec")
        for i in range(n_classes):
            i_class = n_classes_unique[i]
            prec, rec = utils.get_prec_rec(conf_mat, i)
            print(i_class, "\t", prec, "\t", rec)

        if acc_test > max_acc_test:

            max_acc_test = acc_test

            print("### nseg MAXIMUM ACCURACY TEST FOUND: ", acc_test)
            print("### TEST FOUND: ", mean_auc_fold_test)
            print("### nseg Number of trees: ", estim)
            print("### Max depth of trees: ", depth)

            rf_winner["logger_index"] = logger_index
            rf_winner["nseg_n_trees"] = estim
            rf_winner["nseg_tree_depth"] = depth
            rf_winner["nseg_auc_test"] = mean_auc_fold_test
            rf_winner["nseg_auc_val"] = mean_auc_fold
            rf_winner["nseg_rf_object"] = classifier
            rf_winner["nseg_acc_test"] = acc_test
            rf_winner["nseg_acc_train"] = acc_train

            # Save winner model
            pickle_out = model_save_path + "rf_nseg_winner.pkl"
            with open(pickle_out, 'wb') as file:
                pickle.dump(rf_winner, file, protocol=pickle.HIGHEST_PROTOCOL)

            # for i, color in zip(range(n_classes), colors):
            for i in range(n_classes):
                i_class = n_classes_unique[i]
                # color = colors[i]
                lab = 'ROC class {0} (area = {1:0.2f})'.format(i_class, roc_auc_test[i_class])
                plt.plot(fpr_test[i_class], tpr_test[i_class],
                         label=lab)
                print("# Best test AUC for class ", i_class, ": ", roc_auc_test[i_class])
            out_file = model_save_path + "rf_nseg_best_acc.pdf"
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC for RF multi-class', fontsize=16)
            plt.legend(loc='lower right')
            plt.savefig(out_file, format="pdf")
            # plt.show(block=True)
            plt.close()

        print("######################## nsegl iteration ", logger_index, " finished ######################### \n\n")
        logger_index += 1

    print("FINISHED TRAINING")
    print("TIME TO COMPLETE ", n_tot_cases, " CV CASES: ", (time.time() - t_start), " sec")
    print("Best performing RF combination:")
    print("logger_index: ", rf_winner["logger_index"])
    print("nseg_n_trees", rf_winner["nseg_n_trees"])
    print("nseg_tree_depth", rf_winner["nseg_tree_depth"])
    print("nseg_auc_test", rf_winner["nseg_auc_test"])
    print("nseg_auc_val", rf_winner["nseg_auc_val"])
    print("nseg_acc_test", rf_winner["nseg_acc_test"])
    print("nseg_acc_train", rf_winner["nseg_acc_train"])

    save_dict = {"model_save_num": PARAMETERS['model_save_num'],
                                       "wind_size": PARAMETERS['wind_size'],
                                       "pred_var": PARAMETERS['pred_var'],
                                       "model_type": PARAMETERS['model_type'],
                                       'remove_ran_features': PARAMETERS['remove_ran_features'],
                                       'remove_dash_features': PARAMETERS['remove_dash_features'],
                                       'remove_prev_brate_val': PARAMETERS['remove_prev_brate_val'],
                                       "n_trees": rf_winner["nseg_n_trees"],
                                       "tree_depth": rf_winner["nseg_tree_depth"],
                                       "auc_test": rf_winner["nseg_auc_test"],
                                       "auc_val": rf_winner["nseg_auc_val"],
                                       "acc_train": rf_winner["nseg_acc_train"],
                                       "acc_test": rf_winner["nseg_acc_test"]}
    #pd_logger = pd.DataFrame(columns=save_dict.keys())
    #pd_logger.loc[logger_index] = save_dict
    # Save log table
    #pd_logger.to_csv(out_path_pdlogger, mode='a+')
    with open(out_path_pdlogger, 'a+') as file:
        dictwriter_object = DictWriter(file, fieldnames=save_dict.keys())
        dictwriter_object.writerow(save_dict)
        file.close()