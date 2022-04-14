"""
    @description:       File for finding the best performing RF for predicting GNB association
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              07.07.2020
"""

# =============================================================================
#  Import Section:
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy import interpolate
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load Data Imports
import load_data as ld
import train_eval_utils as utils
from data_columns import RES_MERGED_COLS, TARGET_CELLIDASSOC, FILTER_COLS, FEATURE_COLS_GNB, TARGET_COLS, GNB_COLS

NON_SIG_GNB_FEAT_COLS =['predCellId', 'mean-vel_x', 'mean-vel_y', 'mean-vel_z'] + GNB_COLS
# ==================================================================
# Specify paths
# ==================================================================
francis = True
smote = False
only_sig_features = True

if smote:
    from imblearn.over_sampling import SMOTE

# NOTE load data path
load_path = "./data_train/parallel/"
# load_path = "../../data/"
#if francis: load_path = "/home/daniel/Documents/dash/data/data_train/dataset7-4sWsize-4aggsize/" # NOTE
if francis: load_path = "../data/data_train/dataset7-4sWsize-4aggsize/" # NOTE
print("Load path: ", load_path)

# Directory for saving winner
out_path_winner = "./model_output/" + time.strftime("%Y-%m-%d/")
if francis:
    out_path_winner = "./model_output/" + time.strftime("francis-dataset7-4swsize-4sAggsize-%Y-%m-%d/")
if francis and smote:
    out_path_winner = "./model_output/" + time.strftime("francis-smote-%Y-%m-%d/")
if not os.path.isdir(out_path_winner):
    os.makedirs(out_path_winner)
print("Out path parent: ", out_path_winner)

# Directory for saving logger and plots
out_path_log = out_path_winner + "log/"
if not os.path.isdir(out_path_log):
    os.makedirs(out_path_log)

# directory for saving csv log
out_path_pdlogger = out_path_winner + 'log_gnb.csv'
print("CSV logger output: ", out_path_pdlogger)

# ==================================================================
# Load the data
# ==================================================================
print("Loading the data ...")
df_train = ld.load_rf_trainer_data(load_path, verbose=True)
print("The data contains ", len(df_train), " samples.")
print("INF values in df_train: ", df_train.index[np.isinf(df_train).any(1)])
df_train = df_train.astype('float32')
df_train = df_train[~df_train.isin([np.nan, np.inf, -np.inf]).any(1)]
print("INF values in df_train: ", df_train.index[np.isinf(df_train).any(1)])
print("The data contains ", len(df_train), " samples AFTER removing inf vals")

print("Unique values in the dataset: ", df_train[TARGET_CELLIDASSOC[0]].value_counts())
# NOTE: modify according to the wize analysis plot
df_train = df_train.loc[df_train[TARGET_CELLIDASSOC[0]] > 0.0].copy()
print("predCellId bigger than 0 removed since no need to predict non-associated UEs")
# ==================================================================
# Parameters for Random Forest exploration
# ==================================================================
# Number of possible n_estimators to evaluate
n_n_estimators_step = 20
# Bounds for possible n_estimators values
n_estim_lowbound = 50
n_estim_upbound = 100
assert n_estim_upbound > n_estim_lowbound, "### ERROR: n_estim_upbound < n_estim_lowbound"
assert n_n_estimators_step < (n_estim_upbound - n_estim_lowbound), "### ERROR: n_estim_lowbound > (n_estim_upbound - " \
                                                                "n_estim_lowbound)"
# Number of trees in the RF
n_estimators = np.arange(n_estim_lowbound, n_estim_upbound, step=n_n_estimators_step)
print("Possible number of trees: ", n_estimators)

# Number of possible tree depths to evaluate
n_max_depth_step = 10
# Bounds for possible n_estimators values
nmaxdepth_lowbound = 15
nmaxdepth_upbound = 30
assert nmaxdepth_upbound > nmaxdepth_lowbound, "### ERROR: nmaxdepth_upbound < nmaxdepth_lowbound"
assert n_max_depth_step < (nmaxdepth_upbound - nmaxdepth_lowbound), "### ERROR: n_max_depth > (nmaxdepth_upbound - " \
                                                               "nmaxdepth_lowbound)"
# Max depth of tree
max_depth = np.arange(nmaxdepth_lowbound, nmaxdepth_upbound, step=n_max_depth_step)
print("Possible max depths of trees: ", max_depth)

# Exploration grid
rf_grid = [(i, j) for i in n_estimators.tolist() for j in max_depth.tolist()]

# Number of parallel processes
n_jobs_rf = 30

# Test-train set split size
split_size = 0.20

# Cross Validation K folds
n_Kfold_splits = 10

n_tot_cases = n_estimators.shape[0] * max_depth.shape[0]
print("A total of ", n_tot_cases, " cases will be evaluated.\n")

# Number of min samples to split tree
# n_min_samples_split = 1

# Dictionary for storing best performance tree in the test set
rf_winner = {"logger_index": None,
             "gnb_n_trees": 0,
             "gnb_tree_depth": 0,
             "gnb_auc_val": 0.0,
             "gnb_auc_test": 0.0,
             "gnb_acc_test": 0.0,
             "gnb_acc_train": 0.0,
             "gnb_rf_object": None,
             "gnb_featcols": None,
             "gnb_targetcol": None}

mean_auc_val = 0.0

max_acc_test = 0.0

# Pandas dataFrame for logging
out_cols = ["gnb_n_trees", "gnb_tree_depth", "gnb_cv_mean_auc", "train_acc", "gnb_test_auc", "test_acc"]
pd_logger = pd.DataFrame(columns=out_cols)
logger_index = 0


# ===============================================================
# Load and split the Data
# ===============================================================

# Shuffle the rows in the dataFrame
df_train = df_train.sample(frac=1).copy()

Y_tot = df_train[TARGET_CELLIDASSOC].values
unique, counts = np.unique(Y_tot, return_counts=True)
print("TARGET UNIQUE VALUES: ", dict(zip(unique, counts)))
n_classes_unique = np.unique(Y_tot)
print("GNB unique classes: ", n_classes_unique)
n_classes = np.shape(n_classes_unique)[0]
print("Number of GNB unique classes: ", n_classes)

FEAT_COLS = [x for x in FEATURE_COLS_GNB if not x in TARGET_COLS]
if only_sig_features:
	# Keep only the signal features
	FEAT_COLS = [x for x in FEAT_COLS if not x in NON_SIG_GNB_FEAT_COLS]

print(FEAT_COLS)

Xgnb_train, Ygnb_train, Xgnb_test, Ygnb_test, stratified = utils.split_test_train(df_train, FEAT_COLS,
                                                                                  TARGET_CELLIDASSOC, split_size)
rf_winner["gnb_featcols"] = FEAT_COLS
rf_winner["gnb_targetcol"] = TARGET_CELLIDASSOC

if smote:
    print('BEFORE SMOTE dataset shape %s' % Counter(Ygnb_train[:,0]))
    # Oversample all but majority class
    sm = SMOTE(sampling_strategy='not majority', random_state=42)
    Xgnb_train, Ygnb_train = sm.fit_resample(Xgnb_train, Ygnb_train)
    print('Resampled dataset shape %s' % Counter(Ygnb_train))
# ===============================================================
# Build-Train-Test
# ===============================================================
t_start = time.time()

for (estim, depth) in rf_grid:
    print("\nNumber of estim: ", estim, "\tMax tree depth: ", depth)
    if stratified:
        cv = StratifiedKFold(n_splits=n_Kfold_splits)
    else:
        cv = KFold(n_splits=n_Kfold_splits)

    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=estim, max_depth=depth,
                                                            # min_samples_split=n_min_samples_split,
                                                            n_jobs=n_jobs_rf,
                                                            # random_state=42
                                                            ))
    mean_auc_cv = []

    cv_tracker = 0

    for cv_train, cv_test in cv.split(Xgnb_train, Ygnb_train):

        print("Starting CV fold ", cv_tracker)

        classifier.fit(Xgnb_train[cv_train], Ygnb_train[cv_train])

        y_score_cv = classifier.predict_proba(Xgnb_train[cv_test])
        y_score_cv[np.isnan(y_score_cv)] = 0

        fpr_val = dict()
        tpr_val = dict()
        roc_auc_val = dict()

        roc_auc_list = []

        for i in range(n_classes):
            i_class = n_classes_unique[i]

            Y_gnb_class = np.asarray(Ygnb_train[cv_test] == i_class, dtype=np.float)
            tmp_y_score = y_score_cv[:, i]

            fpr_val[i_class], tpr_val[i_class], _ = roc_curve(Y_gnb_class, tmp_y_score)
            roc_auc_val[i_class] = auc(fpr_val[i_class], tpr_val[i_class])
            roc_auc_list.append(roc_auc_val[i_class])

        mean_auc_fold = np.mean(np.asarray(roc_auc_list))
        mean_auc_cv.append(mean_auc_fold)
        print("gnb mean ROC AUC on val set ", cv_tracker, " :", mean_auc_fold)
        cv_tracker += 1

    mean_auc_TRAIN = np.mean(np.asarray(mean_auc_cv))

    print("### gnb mean ROC AUC in CROSS validated set: ", mean_auc_TRAIN, "\n")
    # Calculate accuracy on train set
    y_train_tot_predclass = classifier.predict(Xgnb_train)
    acc_train = accuracy_score(Ygnb_train, y_train_tot_predclass)
    print("### GNB ACC on TRAIN: ", acc_train, "\n")

    print("### GNB evaluating test set ... ")

    y_test_score = classifier.predict_proba(Xgnb_test)
    y_test_score[np.isnan(y_test_score)] = 0

    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    roc_auc_list_test = []

    for i in range(n_classes):
        i_class = n_classes_unique[i]

        Y_test_class = np.asarray(Ygnb_test == i_class, dtype=np.float)
        tmp_y_test_score = y_test_score[:, i]

        fpr_test[i_class], tpr_test[i_class], _ = roc_curve(Y_test_class, tmp_y_test_score)
        roc_auc_test[i_class] = auc(fpr_test[i_class], tpr_test[i_class])
        roc_auc_list_test.append(roc_auc_test[i_class])

    tmp_test_auc = np.asarray(roc_auc_list_test)
    tmp_test_auc = np.nan_to_num(tmp_test_auc)

    mean_auc_fold_test = np.mean(tmp_test_auc)

    # Calculate accuracy on test set
    y_test_predclass = classifier.predict(Xgnb_test)
    acc_test = accuracy_score(Ygnb_test, y_test_predclass)

    conf_mat = confusion_matrix(Ygnb_test, y_test_predclass)

    print("CONFUSION MATRIX: ")
    print(conf_mat)

    print("gnb mean AUC test set: ", mean_auc_fold_test)
    print("GNB accuracy test set: ", acc_test)

    print("Class\t Prec \t Rec")
    for i in range(n_classes):
        i_class = n_classes_unique[i]
        prec, rec = utils.get_prec_rec(conf_mat, i)
        print(i_class, "\t", prec, "\t", rec)

    if acc_test > max_acc_test:

        max_acc_test = acc_test

        print("### GNB MAXIMUM AUC TEST FOUND: ", mean_auc_fold_test)
        print("### GNB Number of trees: ", estim)
        print("### Max depth of trees: ", depth)

        rf_winner["logger_index"] = logger_index
        rf_winner["gnb_n_trees"] = estim
        rf_winner["gnb_tree_depth"] = depth
        rf_winner["gnb_auc_test"] = mean_auc_fold_test
        rf_winner["gnb_auc_val"] = mean_auc_fold
        rf_winner["gnb_rf_object"] = classifier
        rf_winner["gnb_acc_test"] = acc_test
        rf_winner["gnb_acc_train"] = acc_train

        # Save winner model
        pickle_out = out_path_winner + "rf_gnb_winner.pkl"
        with open(pickle_out, 'wb') as file:
            pickle.dump(rf_winner, file, protocol=pickle.HIGHEST_PROTOCOL)

        # for i, color in zip(range(n_classes), colors):
        for i in range(n_classes):
            i_class = n_classes_unique[i]
            # color = colors[i]
            lab = 'ROC class {0} (area = {1:0.2f})'.format(i_class, roc_auc_test[i_class])
            print(lab)
            plt.plot(fpr_test[i_class], tpr_test[i_class],
                     label='ROC for {0} (area = {1:0.2f})'.format(i_class, roc_auc_test[i_class])
                     )
            print("# Best test AUC for class ", i_class, ": ", roc_auc_test[i_class])
        out_file = out_path_winner + "rf_gnb_best_auc.pdf"
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

    pd_logger.loc[logger_index] = {"gnb_n_trees": estim,
                                   "gnb_tree_depth": depth,
                                   "gnb_cv_mean_auc": mean_auc_TRAIN,
                                   "train_acc": acc_train,
                                   "gnb_test_auc": mean_auc_fold_test,
                                   "test_acc": acc_test}
    # Save log table
    pd_logger.to_csv(out_path_pdlogger)

    print("######################## Model Case ", logger_index,
          " finished ######################### \n\n")
    logger_index += 1

print("FINISHED TRAINING")
print("TIME TO COMPLETE ", n_tot_cases, " CV CASES: ", (time.time() - t_start), " sec")
print("Best performing RF combination:")
print("logger_index: ", rf_winner["logger_index"])
print("gnb_n_trees", rf_winner["gnb_n_trees"])
print("gnb_tree_depth", rf_winner["gnb_tree_depth"])
print("gnb_auc_test", rf_winner["gnb_auc_test"])
print("gnb_auc_val", rf_winner["gnb_auc_val"])
print("gnb_acc_test", rf_winner["gnb_acc_test"])
print("gnb_acc_train", rf_winner["gnb_acc_train"])

# ===============================================================
# DEBUG CODE
# ===============================================================
# df_train_vals = df_train.to_numpy(dtype=np.float32)
# print("Df train vals : ", np.any(np.isinf(df_train_vals)), np.max(np.abs(df_train_vals)))
# inf_index = df_train.index[np.isinf(df_train).any(1)]
# print("INF values in df_train: ", df_train.columns.to_series()[np.isinf(df_train).any()])
# Xmode_train = Xmode_train.astype(np.float32)
# print("Performin nan checks: ")
# print("Xmode_train : ", np.any(np.isnan(Xmode_train)), np.max(np.abs(Xmode_train)))
# print("Xmode_test : ", np.any(np.isnan(Xmode_test)), np.max(np.abs(Xmode_test)))
# print("Ymode_train : ", np.any(np.isnan(Ymode_train)), np.max(np.abs(Ymode_train)))
# print("Ymode_test : ", np.any(np.isnan(Ymode_test)), np.max(np.abs(Ymode_test)))
# print('BEFORE SMOTE dataset shape %s' % Counter(Ymode_train[:,0]))
# # Oversample all but majority class
# sm = SMOTE(sampling_strategy='not majority', random_state=42)
# Xmode_train, Ymode_train = sm.fit_resample(Xmode_train, Ymode_train)
# print('Resampled dataset shape %s' % Counter(Ymode_train[:,0]))
