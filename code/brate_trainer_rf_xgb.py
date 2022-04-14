"""
    @description:       File for finding the best performing RF/XGB for predicting segment mode
    @author:            Daniel F. Perez-Ramirez
    @collaborators:     Akhila Rao, Rasoul Behrabesh, Rebecca Steinert
    @project:           DASH
    @date:              17.06.2020
"""

# =============================================================================
#  Import Section:
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load Data Imports
import load_data as ld
import data_columns as dc
from csv import DictWriter
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)

def split_test_train(in_df, feat_cols, ycol, in_test_size, rstate=42):
    X = in_df[feat_cols].values
    Y = in_df[ycol].values

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=in_test_size, stratify=Y,
                                                            random_state=rstate)
        stratified = True
    except ValueError:
        print("\n## NOTE: COULD NOT STRATIFY TARGET COL: too few samples \n")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=in_test_size, random_state=rstate)
        stratified = False
    return X_train, y_train, X_test, y_test, stratified

def hyperparameter_tune(X_train, Y_train, PARAMETERS):
    # hyperparameter setting for RF and XGB
    n_estimators = PARAMETERS['n_trees']
    print("Number of trees: ", n_estimators)
    max_depth = np.array(PARAMETERS['tree_depth_list'])
    print("List of max depths of trees: ", max_depth)
    min_samples_leaf = np.array(PARAMETERS['min_samples_leaf_list'])
    print("List of min samples per leaf node of trees: ", min_samples_leaf)
    # Exploration grid
    rf_grid = [(i, j) for i in min_samples_leaf.tolist() for j in max_depth.tolist()]
    # Number of parallel processes
    n_jobs = 30
    # Cross Validation K folds
    n_Kfold_splits = 5
    n_tot_cases = min_samples_leaf.shape[0] * max_depth.shape[0]
    print("A total of ", n_tot_cases, " cases will be evaluated.\n")

    # metrics used to select the best hyperparameter combination
    max_roc_auc_overHyperparam = 0.0
    max_acc_overHyperparam = 0.0
    max_f1score_overHyperparam = 0.0

    logger_index = 0
    winning_hyperParam = {'logger_index': None,
         'n_trees': 0,
         'tree_depth': 0,
         'min_samples_leaf': 0,
         'roc_auc_test': 0.0,
         'acc_test': 0.0,
         'f1score_test': 0.0,
         'classifier': None}
    
    # iterate over hyperparameters 
    for (samp_leaf, depth) in rf_grid:
        print('-------------------------------------------')
        print("\nMin samples leaf: ", samp_leaf, "\tMax tree depth: ", depth)
        if stratified:
            cv = StratifiedKFold(n_splits=n_Kfold_splits)
        else:
            cv = KFold(n_splits=n_Kfold_splits)

        #classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, 
        #                                                        min_samples_leaf=samp_leaf, n_jobs=n_jobs))
        if PARAMETERS['model_type'] == 'rf':
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, 
                                                                min_samples_leaf=samp_leaf, n_jobs=n_jobs)
        elif PARAMETERS['model_type'] == 'xgb':
            classifier = XGBClassifier(n_estimators=n_estimators, max_depth=depth,
                                   objective='multi:softmax',
                                   min_child_weight=samp_leaf, n_jobs=n_jobs, use_label_encoder=False)
                                   # learning_rate=0.05,
        else: print('ERROR: unknown model type')
                                   
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
            winning_hyperParam["min_samples_leaf"] = samp_leaf
            winning_hyperParam["roc_auc"] = mean_roc_auc_overKFolds
            winning_hyperParam["acc"] = mean_acc_overKFolds
            winning_hyperParam["f1score"] = mean_f1score_overKFolds
                
        logger_index += 1

    return winning_hyperParam["n_trees"], winning_hyperParam["tree_depth"], winning_hyperParam["min_samples_leaf"] 

def find_baseline(PARAMETERS, train_data_dir):
    print('Reading datset from: ', train_data_dir)
    # load the dataset
    df = ld.load_rf_trainer_data(train_data_dir, verbose=False)
    df = df.astype('float32')
    # keep only the real number values 
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # columns from the dataset that are input features 
    FEAT_COLS = ['mean-AggBitRates']
    # column from the dataset that is the ground truth label
    TARGET_SEGMODE = dc.TARGET_SEGMODE
    # remove the samples for which we do not have ground truth 
    df = df.loc[df[TARGET_SEGMODE[0]] > 0.0].copy()
    # Test-train set split size
    split_size = 0.20
    X_train, y_train, X_test, Y_test, stratified = split_test_train(df, FEAT_COLS,
                                                                                          TARGET_SEGMODE, split_size)
    yhat_test = X_test.copy()
    #print('yhat_class: ', yhat_class)
    # set the nans to 0 
    yhat_test[np.isnan(yhat_test)] = 0
    # compute accuracy metric and append to list
    acc = accuracy_score(Y_test, yhat_test)
    # compute precision, recall, fscore using 'macro' averaging over the classes 
    prec, rec, f1score, support = precision_recall_fscore_support(Y_test, yhat_test, average='macro')
    print('Baseline')
    print('accuracy', acc, 'f1score', f1score)

def create_train_test_sets(PARAMETERS, train_data_dir):
    #print('Reading datset from: ', train_data_dir)
    # load the dataset
    df = ld.load_rf_trainer_data(train_data_dir, verbose=False)
    df = df.astype('float32')
    # keep only the real number values 
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # columns from the dataset that are input features 
    FEAT_COLS = dc.FEAT_COLS
    # column from the dataset that is the ground truth label
    TARGET_SEGMODE = dc.TARGET_SEGMODE
    #print("The data contains ", len(df), " samples AFTER removing nan/inf vals")
    #print("Unique values in the dataset: ") 
    #print(df[TARGET_SEGMODE[0]].value_counts())
    # NOTE: modify according to the wize analysis plot
    # remove the samples for which we do not have ground truth 
    df = df.loc[df[TARGET_SEGMODE[0]] > 0.0].copy()
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
    #FEAT_COLS = ['n-AggBitRates', 'mean-AggBitRates', 'mean-Dl-thput', 
    #             'mean-Dl-sinr', 'mean-Dl-rsrp', 'mean-Dl-mcs']
    #FEAT_COLS = ['n-AggBitRates', 'mean-AggBitRates']
    #FEAT_COLS = ['mean-AggBitRates', 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps',
    #            'mean-Dl-thput', 'mean-BufferSize']        
    #FEAT_COLS = ['mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps', 'mean-Dl-thput',
    #            'mean-BufferSize', 'mean-BufferBytes', 'mean-Dl-sinr']
    #['n-AggBitRates', , 'mean-ThPutOverLastSeg_bps', 'mean-avgThPutOverWindow_bps',
    #         , 'mean-BufferSize', 
    #         'mean-Dl-rsrp', , 
    #         'mean-Dl-mcs', ]
    #print('WARNING: overriding feature cols selection')
    # Test-train set split size
    split_size = 0.20
    X_train, Y_train, X_test, Y_test, stratified = split_test_train(df, FEAT_COLS,
                                                                                          TARGET_SEGMODE, split_size)  
    return X_train, Y_train, X_test, Y_test

def train_rf_xgb(PARAMETERS, train_data_dir, model_save_path, tune_hyperparam):
    # path to save the model after training
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    #print("Model save path: ", model_save_path)

    # path to save parameters and result metrics as an entry in a csv log
    out_path_pdlogger = '../models/' + PARAMETERS['model_type']+ '/' + PARAMETERS['output_file_name']
    #print("CSV logger output: ", out_path_pdlogger)
    
    X_train, Y_train, X_test, Y_test = create_train_test_sets(PARAMETERS, train_data_dir)
    # Dictionary for storing best performance tree in the test set
    model = {'logger_index': None,
                 'n_trees': 0,
                 'tree_depth': 0,
                 'min_samples_leaf': 0,
                 'acc': 0.0,
                 'roc_auc': 0.0,
                 'f1score': 0.0,
                 'classifier': None}
    t_start = time.time()

    if tune_hyperparam:
        n_trees, tree_depth, min_samples_leaf = hyperparameter_tune(X_train, Y_train, PARAMETERS)
    else:
        n_trees, tree_depth, min_samples_leaf = PARAMETERS['n_trees'], PARAMETERS['tree_depth'], PARAMETERS['min_samples_leaf'] 

    #print('#trees: ', n_trees, 'tree_depth: ', tree_depth,' min_samples_leaf: ', min_samples_leaf)
    n_jobs = 30  
    # Train over the entire dataset with this hyperparameter setting 
    #tuned_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=winning_hyperParam["n_trees"], 
    #                                                              max_depth=winning_hyperParam["tree_depth"], 
    #                                                              min_samples_leaf=winning_hyperParam["min_samples_leaf"], 
    #                                                              n_jobs=n_jobs))
    if PARAMETERS['model_type'] == 'rf':
        tuned_classifier = RandomForestClassifier(n_estimators=n_trees, 
                                                                  max_depth=tree_depth, 
                                                                  min_samples_leaf=min_samples_leaf, 
                                                                  n_jobs=n_jobs)
    elif PARAMETERS['model_type'] == 'xgb':
        tuned_classifier = XGBClassifier(n_estimators=n_trees, max_depth=tree_depth,
                               objective='multi:softmax',
                               min_child_weight=min_samples_leaf, n_jobs=n_jobs,
                                         use_label_encoder=True)
                               # learning_rate=0.05,
    else: print('ERROR: unknown model type')

    #print('Y_train: ', Y_train)
    tuned_classifier.fit(X_train, Y_train.ravel())
    # predict 
    yhat_proba = tuned_classifier.predict_proba(X_test)
    #print('yhat_proba: ', yhat_proba)
    yhat_class = tuned_classifier.predict(X_test)
    #print('yhat_class: ', yhat_class)
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
    
    model["n_trees"] = n_trees
    model["tree_depth"] = tree_depth
    model["min_samples_leaf"] = min_samples_leaf
    model["classifier"] = tuned_classifier
            
    print('acc, roc_auc, f1score: ', model["acc"], model["roc_auc"], model["f1score"]) 
    np.set_printoptions(suppress=True)
    con_mat = confusion_matrix(Y_test, yhat_class, normalize='true')
    print(con_mat)
    runtime = time.time() - t_start
    labels = ['1', '2.5', '5', '8', '16', '35']
    plt.figure(figsize=(con_mat.shape[0],con_mat.shape[1]))
    hmap = sns.heatmap(con_mat, vmin=0, cmap='Greens', 
                       xticklabels = labels, yticklabels = labels, 
                       annot=True, vmax=1, annot_kws={"fontsize":12})
    fig = hmap.get_figure()
    plt.yticks(rotation=45)
    fig.autofmt_xdate(rotation=45)
    plt.show()
    
    # Save the model for best hyperparameter set
    pickle_out = model_save_path + "brate_classifier.pkl"
    with open(pickle_out, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_dict = {"model_save_num": PARAMETERS['model_save_num'],
                                       "aggr_wind_size": PARAMETERS['aggr_wind_size'],
                                       "horz_wind_size": PARAMETERS['horz_wind_size'],
                                       "pred_var": PARAMETERS['pred_var'],
                                       "model_type": PARAMETERS['model_type'],
                                       'remove_ran_features': PARAMETERS['remove_ran_features'],
                                       'remove_dash_features': PARAMETERS['remove_dash_features'],
                                       'remove_prev_brate_val': PARAMETERS['remove_prev_brate_val'],
                                       "n_trees": model["n_trees"],
                                       "tree_depth": model["tree_depth"],
                                       "min_samples_leaf": model["min_samples_leaf"],
                                       'runtime': runtime,
                                       "acc_test": model["acc"],
                                       "roc_auc_test": model["roc_auc"],                 
                                       "f1score_test": model["f1score"]}
    with open(out_path_pdlogger, 'a+') as file:
            dictwriter_object = DictWriter(file, fieldnames=save_dict.keys())
            dictwriter_object.writeheader()
            dictwriter_object.writerow(save_dict)
            file.close()
