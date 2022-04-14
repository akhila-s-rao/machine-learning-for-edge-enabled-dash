"""
    @description:       File for finding the best performing RF for predicting segment mode
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              17.06.2020
"""

# =============================================================================
#  Import Section:
# =============================================================================
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
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
import data_columns as dc
from csv import DictWriter

def run_rf_brate(PARAMETERS, train_data_dir, model_save_path):
    
    # path to save the model after training
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    print("Model save path: ", model_save_path)

    # path to save parameters and result metrics as an entry in a csv log
    out_path_pdlogger = '../models/' + PARAMETERS['model_type']+ '/modFeat3_model_parameters.csv'
    print("CSV logger output: ", out_path_pdlogger)

    # load the dataset
    df = ld.load_rf_trainer_data(train_data_dir, verbose=True)
    df = df.astype('float32')
    # keep only the real number values 
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # columns from the dataset that are input features 
    FEAT_COLS = dc.FEAT_COLS
    # column from the dataset that is the ground truth label
    TARGET_SEGMODE = dc.TARGET_SEGMODE
    print("The data contains ", len(df), " samples AFTER removing nan/inf vals")
    print("Unique values in the dataset: ") 
    print(df[TARGET_SEGMODE[0]].value_counts())
    # NOTE: modify according to the wize analysis plot
    # remove the samples for which we do not have ground truth 
    df = df.loc[df[TARGET_SEGMODE[0]] > 0.0].copy()

    # hyperparameter setting for RF
    n_estimators = PARAMETERS['n_trees']
    print("Number of trees: ", n_estimators)
    max_depth = np.array(PARAMETERS['tree_depth_list'])
    print("List of max depths of trees: ", max_depth)
    min_samples_split = np.array(PARAMETERS['min_samples_split_list'])
    print("List of min samples split of trees: ", min_samples_split)
    # Exploration grid
    rf_grid = [(i, j) for i in min_samples_split.tolist() for j in max_depth.tolist()]
    # Number of parallel processes
    n_jobs = 30
    # Test-train set split size
    split_size = 0.20
    # Cross Validation K folds
    n_Kfold_splits = 2
    n_tot_cases = min_samples_split.shape[0] * max_depth.shape[0]
    print("A total of ", n_tot_cases, " cases will be evaluated.\n")

    # Dictionary for storing best performance tree in the test set
    model = {'logger_index': None,
                 'n_trees': 0,
                 'tree_depth': 0,
                 'min_samples_split': 0,
                 'roc_auc_test': 0.0,
                 'acc_test': 0.0,
                 'f1score_test': 0.0,
                 'classifier': None,
                 'featcols': None,
                 'targetcol': None}

    # metrics used to select the best hyperparameter combination
    max_roc_auc_overHyperparam = 0.0
    max_acc_overHyperparam = 0.0
    max_f1score_overHyperparam = 0.0
    
    # Shuffle the rows in the dataFrame
    df = df.sample(frac=1).copy()
    classes_unique = np.unique(df[TARGET_SEGMODE].values)
    
    # remove subgroup of features if indicated to do so
    if PARAMETERS['remove_dash_features']:
            FEAT_COLS = [x for x in FEAT_COLS if not x in dc.DASH_FEAT_COLS]
    if PARAMETERS['remove_ran_features']:
            FEAT_COLS = [x for x in FEAT_COLS if not x in dc.RAN_FEAT_COLS]
    if PARAMETERS['remove_prev_brate_val']:
                FEAT_COLS = [x for x in FEAT_COLS if not x in \
                             ['n-AggBitRates', 'mean-AggBitRates'] ]
    #print("FEATURE COLS after filtering: ", FEAT_COLS) 
    
    X_train, Y_train, X_test, Y_test, stratified = utils.split_test_train(df, FEAT_COLS,
                                                                                          TARGET_SEGMODE, split_size)
    model['featcols'] = FEAT_COLS
    model['targetcol'] = TARGET_SEGMODE
    t_start = time.time()
    logger_index = 0
    winning_hyperParam = {'logger_index': None,
         'n_trees': 0,
         'tree_depth': 0,
         'min_samples_split': 0,
         'roc_auc_test': 0.0,
         'acc_test': 0.0,
         'f1score_test': 0.0,
         'classifier': None,
         'featcols': None,
         'targetcol': None}
    
    # iterate over hyperparameters 
    for (samp_split, depth) in rf_grid:
        print('-------------------------------------------')
        print("\nMin samples split: ", samp_split, "\tMax tree depth: ", depth)
        if stratified:
            cv = StratifiedKFold(n_splits=n_Kfold_splits)
        else:
            cv = KFold(n_splits=n_Kfold_splits)

        classifier = XGBClassifier(n_estimators=n_estimators, max_depth=depth,
                                   objective='multi:softmax',
                                   min_child_weight = 
                                   # learning_rate=0.05,
                                   n_jobs=n_jobs)
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, 
                                                                min_samples_split=samp_split, n_jobs=n_jobs))
        cv_tracker = 1
        acc_overKFolds = []
        roc_auc_overKFolds = []
        f1score_overKFolds = []

        # iterate over K CV folds 
        for train, test in cv.split(X_train, Y_train):
            print("Starting CV fold ", cv_tracker)
            # train
            classifier.fit(X_train[train], Y_train[train])
            # predict class probabilities 
            yhat_proba = classifier.predict_proba(X_train[test])
            yhat_class = classifier.predict(X_train[test])
            # based on https://stackoverflow.com/questions/49525618/sklearns-predict-proba-returns-infinite-probabilties
            # set the nans to 0 
            yhat_proba[np.isnan(yhat_proba)] = 0
            yhat_class[np.isnan(yhat_class)] = 0

            roc_auc_overKFolds.append(roc_auc_score(Y_train[test], yhat_proba, multi_class='ovr', average='macro'))
            # compute accuracy metric and append to list
            acc_overKFolds.append(accuracy_score(Y_train[test], yhat_class))
            # compute precision, recall, fscore using 'macro' averaging over the classes 
            prec, rec, f1score, support = precision_recall_fscore_support(Y_train[test], yhat_class, average='macro')
            # saving only the f1score
            f1score_overKFolds.append(f1score)
        
            cv_tracker += 1

        # average metrics over k folds
        mean_roc_auc_overKFolds = np.mean(np.asarray(roc_auc_overKFolds))
        mean_acc_overKFolds = np.mean(np.asarray(acc_overKFolds))
        mean_f1score_overKFolds = np.mean(np.asarray(f1score_overKFolds))
        
        print("mean (roc_auc, accuracy, f1score) over CV sets: ", 
              mean_roc_auc_overKFolds, mean_acc_overKFolds, mean_f1score_overKFolds)

        # Update the best performing hyperparameter combination 
        if mean_acc_overKFolds > max_acc_overHyperparam:
            max_acc_overHyperparam = mean_acc_overKFolds
            winning_hyperParam["logger_index"] = logger_index
            winning_hyperParam["n_trees"] = n_estimators
            winning_hyperParam["tree_depth"] = depth
            winning_hyperParam["min_samples_split"] = samp_split
            #winning_hyperParam["rf_object"] = classifier
            winning_hyperParam["roc_auc"] = mean_roc_auc_overKFolds
            winning_hyperParam["acc"] = mean_acc_overKFolds
            winning_hyperParam["f1score"] = mean_f1score_overKFolds
                
        logger_index += 1

    print('finished hyperparameter tuning')
    print('min_samples_split: ' + str(winning_hyperParam["min_samples_split"]))
    print('tree_depth: ' + str(winning_hyperParam["tree_depth"]))
    
    # Train over the entire dataset with this hyperparameter setting 
    tuned_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=winning_hyperParam["n_trees"], 
                                                                  max_depth=winning_hyperParam["tree_depth"], 
                                                                  min_samples_split=winning_hyperParam["min_samples_split"], n_jobs=n_jobs))
    
    tuned_classifier.fit(X_train, Y_train)
    # predict 
    yhat_proba = tuned_classifier.predict_proba(X_test)
    yhat_class = tuned_classifier.predict(X_test)
    # set the nans to 0 
    yhat_proba[np.isnan(yhat_proba)] = 0
    yhat_class[np.isnan(yhat_class)] = 0
    # evaluate and save 
    # average roc_auc metric over classes and append to list
    model["roc_auc"] = roc_auc_score(Y_test, yhat_proba, multi_class='ovr', average='macro')
    # compute accuracy metric and append to list
    model["acc"] = accuracy_score(Y_test, yhat_class)
    # compute precision, recall, fscore using 'macro' averaging over the classes 
    prec, rec, f1score, support = precision_recall_fscore_support(Y_test, yhat_class, average='macro')
    # saving only the f1score
    model["f1score"] = f1score

    model["n_trees"] = winning_hyperParam["n_trees"]
    model["tree_depth"] = winning_hyperParam["tree_depth"]
    model["min_samples_split"] = winning_hyperParam["min_samples_split"]
    model["classifier"] = tuned_classifier
            
    print('roc_auc, acc, f1score: ', model["roc_auc"], model["acc"], model["f1score"])    
    print("TIME TO COMPLETE ", (time.time() - t_start), " sec")
    
    # Save the model for best hyperparameter set
    pickle_out = model_save_path + "brate_rf_classifier.pkl"
    with open(pickle_out, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_dict = {"model_save_num": PARAMETERS['model_save_num'],
                                       "wind_size": PARAMETERS['wind_size'],
                                       "pred_var": PARAMETERS['pred_var'],
                                       "model_type": PARAMETERS['model_type'],
                                       'remove_ran_features': PARAMETERS['remove_ran_features'],
                                       'remove_dash_features': PARAMETERS['remove_dash_features'],
                                       'remove_prev_brate_val': PARAMETERS['remove_prev_brate_val'],
                                       "n_trees": model["n_trees"],
                                       "tree_depth": model["tree_depth"],
                                       "min_samples_split": model["min_samples_split"],
                                       "roc_auc_test": model["roc_auc"],
                                       "acc_test": model["acc"],
                                       "f1score_test": model["f1score"]}
    with open(out_path_pdlogger, 'a+') as file:
        dictwriter_object = DictWriter(file, fieldnames=save_dict.keys())
        dictwriter_object.writerow(save_dict)
        file.close()
