# mlp for multi-label classification
import numpy as np
from numpy import mean
from numpy import std

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], #enable=True)
from tensorflow import keras
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import CuDNNLSTM as LSTM
from keras.utils import to_categorical
from keras.metrics import * 
from tensorflow.keras.callbacks import EarlyStopping
import json
#from numba import jit, cuda 
import pandas as pd
import matplotlib.pyplot as plt
import time
#import os
import subprocess
import load_data as ld
#from data_columns import *
import data_columns as dc
from csv import DictWriter
#from sklearn.externals import joblib
import joblib
import seaborn as sns

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

def ohe_to_class(ohe_x):
    ie_x = np.argmax(ohe_x, axis = 1)
    return ie_x

def class_to_ohe(ie_x):
    # TO DO 
    return ohe_x

# create the model
def get_mlp_model(n_inputs, n_outputs, PARAMETERS):
    model = Sequential()
    # input and first hidden layer
    model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    if PARAMETERS['num_layers'] > 1:
        # second hidden layer
        model.add(Dense(15, kernel_initializer='he_uniform', activation='relu'))
        if PARAMETERS['num_layers'] > 2:
            # Third hidden layer
            model.add(Dense(6, kernel_initializer='he_uniform', activation='relu'))
    if(PARAMETERS['classification']):
        # output layer
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss=PARAMETERS['loss'], optimizer='adam', 
                      metrics=PARAMETERS['metrics'])
    
    return model

def get_lstm_model(n_in_feat, n_in_time, n_outputs, PARAMETERS):
    # design network
    model = Sequential()
    if PARAMETERS['num_layers'] == 1: 
        model.add(LSTM(30,
               input_shape=(n_in_feat, n_in_time)))  
    # returns a sequence of vectors of dimension 32
    if PARAMETERS['num_layers'] == 2:
        model.add(LSTM(30, return_sequences=True,
               input_shape=(n_in_feat, n_in_time)))  
        # returns a sequence of vectors of dimension 32
        model.add(LSTM(15))  
        # returns a sequence of vectors of dimension 32
    
    model.add(Dense(n_outputs, activation='softmax'))

    #model.add(LSTM(50, input_shape=(n_in_feat, n_in_time)))
    #model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss=PARAMETERS['loss'], optimizer='adam',
                 metrics=PARAMETERS['metrics'])
    
    return model

def evaluate_NN_model(X_train, y_train, X_test, y_test, model_save_path, PARAMETERS):
    # y is already OHE
    # define model
    if PARAMETERS['model_type'] == 'lstm':
        # reshape input to be 3D [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        n_in_feat, n_in_time, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
        model = get_lstm_model(n_in_feat, n_in_time, n_outputs, PARAMETERS)       
    elif PARAMETERS['model_type'] == 'mlp': 
        n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
        model = get_mlp_model(n_inputs, n_outputs, PARAMETERS)       
    else: 
        print ("ERROR: " + str(PARAMETERS['model_type']) + " is undefined")

    # fit model
    #callback = EarlyStopping(monitor='val_'+PARAMETERS['metrics'][0], 
    #                         #min_delta=0.0000001,
    #                        patience=20)
    
    history = model.fit(X_train, y_train, verbose=0, epochs=PARAMETERS['epochs'], 
                        batch_size=PARAMETERS['batch_size'],
                      #  callbacks=[callback],
                     validation_data=(X_test, y_test), 
                     shuffle=False)
                     #callbacks=[early_stopping])
    # make a prediction on the train and test set
    # these predictions have shape (n_samples, n_classes) and contain probailities
    yhat_train_proba = model.predict(X_train)
    yhat_test_proba = model.predict(X_test)
    scores = model.evaluate(X_test, y_test, return_dict=True)

    # save the model
    model.save(model_save_path)
    if(PARAMETERS['classification']):
        # round probabilities to get OHE format
        yhat_train_ohe = yhat_train_proba.round()
        yhat_test_ohe = yhat_test_proba.round()
        # calculate accuracy
        test_acc = accuracy_score(y_test, yhat_test_ohe)
        train_acc = accuracy_score(y_train, yhat_train_ohe)
        # make sure this has probabilities and not labels 
        roc_auc = roc_auc_score(y_test, yhat_test_proba, multi_class='ovr', average='macro')
        # compute accuracy metric and append to list
        # compute precision, recall, fscore using 'macro' averaging over the classes 
        prec, rec, f1score, support = precision_recall_fscore_support(y_test, yhat_test_ohe, average='macro')
        # saving only the f1score
    else: # regression
        print('Not yet ready with regression')
    
    results = {
            'accuracy': test_acc,
            'roc_auc': roc_auc,
            'f1score': f1score
        }
    print(model.summary())
    print('acc, roc_auc, f1score: ', results["accuracy"], results["roc_auc"], results["f1score"])
    np.set_printoptions(suppress=True)
    con_mat = confusion_matrix(ohe_to_class(y_test), ohe_to_class(yhat_test_ohe), normalize='true')
    print(con_mat)
    labels = ['1', '2.5', '5', '8', '16', '35']
    plt.figure(figsize=(con_mat.shape[0],con_mat.shape[1]))
    hmap = sns.heatmap(con_mat, vmin=0, cmap='Greens', 
                       xticklabels = labels, yticklabels = labels, 
                       annot=True, vmax=1, annot_kws={"fontsize":12})
    fig = hmap.get_figure()
    plt.yticks(rotation=45)
    fig.autofmt_xdate(rotation=45)
    plt.show()    
    return results, history, scores
 
def get_naive_bayes_model():
    model = GaussianNB()
    return model

def evaluate_naive_bayes_model(X_tr,y_tr, X_ev, y_ev, model_save_path,cross_validate=False):
    results = list()
    history = list()
    K = 0
    split_size = 0.2
    # define evaluation procedure
    if cross_validate:
        K = 2
        cv = RepeatedKFold(n_splits=K, n_repeats=1, random_state=1)
    
    print(range(K+1))
    for k in range(K+1):
        print("Inside for loop")
        if crossValidate:
            train_ix, test_ix = cv.split(X_tr)
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            print("This K-fold validation has " + str(X_train.shape[0]) + "samples")
        else:
            X_train, X_test, y_train, y_test = X_tr, X_ev, y_tr, y_ev
        
        model = get_naive_bayes_model() 
        hist = model.fit(X_train, y_train.squeeze())
        history.append(hist)
        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)
        train_acc = 1 - ((y_train.squeeze() != yhat_train).sum()/X_train.shape[0])
        test_acc = 1 - ((y_test.squeeze() != yhat_test).sum()/X_test.shape[0])
        print("Train accuracy: %.3f" % train_acc)
        print("Test accuracy: %.3f" % test_acc)
        results.append(test_acc)
    return results, history

def load_model_and_predict ():
    # load the saved model and predict again 
    loaded_model = load_model(model_save_path) 
    loaded_model.summary()
    #custom_objects={'kullback_leibler_divergence': keras.metrics.KLDivergence})
    if PARAMETERS['model_type'] == 'lstm':
        # reshape input to be 3D [samples, timesteps, features]
        X_t = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_e = X_eval.reshape((X_eval.shape[0], 1, X_eval.shape[1]))
    elif PARAMETERS['model_type'] == 'mlp':
        X_t = X_train
        X_e = X_eval
    # make a prediction on the train set
    yhat_train = loaded_model.predict(X_t)
    yhat_test = loaded_model.predict(X_e)
    if(PARAMETERS['classification']):
        # round probabilities to class labels
        yhat_test = yhat_test.round()
        yhat_train = yhat_train.round()
        # calculate accuracy
        test_acc = accuracy_score(y_eval, yhat_test)
        train_acc = accuracy_score(y_train, yhat_train)
        print("After reloading model Train accuracy: %.3f" % train_acc)
        print("After reloading model Test accuracy: %.3f" % test_acc)

def tell_human_im_done ():
    process = subprocess.Popen(['ssh', 'akhila@193.10.65.7', 
                                'sh', '/home/akhila/telegram_send_msg_to_akhila.sh'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr

def train_mlp_lstm(PARAMETERS, train_data_dir, model_save_path, tune_hyperparam):
    append_model_parameters = True
    change_pred = False
    
    # path to save the model after training
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # path to save parameters and result metrics as an entry in a csv log
    out_path_pdlogger = '../models/' + PARAMETERS['model_type']+ '/' + PARAMETERS['output_file_name']
    print("CSV logger output: ", out_path_pdlogger)

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
    #df = df.sample(frac=1).copy()
    classes_unique = np.unique(df[TARGET_SEGMODE].values)
    n_unique_classes = len(classes_unique)
    
    # remove subgroup of features if indicated to do so
    if PARAMETERS['remove_dash_features']:
            FEAT_COLS = [x for x in FEAT_COLS if not x in dc.DASH_FEAT_COLS]
    if PARAMETERS['remove_ran_features']:
            FEAT_COLS = [x for x in FEAT_COLS if not x in dc.RAN_FEAT_COLS]
    if PARAMETERS['remove_prev_brate_val']:
                FEAT_COLS = [x for x in FEAT_COLS if not x in \
                             ['n-AggBitRates', 'mean-AggBitRates'] ]
    # Test-train set split size
    split_size = 0.20
    n_jobs = 30  
    X_train, y_train, X_test, y_test, stratified = split_test_train(df, FEAT_COLS,
                                                                                          TARGET_SEGMODE, split_size)
    #t_start = time.time()

    #if tune_hyperparam:
        #n_trees, tree_depth, min_samples_leaf = hyperparameter_tune(X_train, Y_train, PARAMETERS)
    #else:
        #n_trees, tree_depth, min_samples_leaf = PARAMETERS['n_trees'], PARAMETERS['tree_depth'], PARAMETERS['min_samples_leaf'] 

  
    #print('Finished hyperparameter tuning')
    if PARAMETERS['pred_var'] == 'brate':
        y_int_train = [dc.BITRATE_INT_DICT[i] for i in y_train.squeeze()]
        y_int_test = [dc.BITRATE_INT_DICT[i] for i in y_test.squeeze()]
    else: print('Dont know what you want me to do')
    #elif PARAMETERS['pred_var'] == 'nseg':
    #    y_int_train = y_train.copy()
    #    y_int_eval = y_eval.copy()

    # one hot encode output variable        
    y_train = to_categorical(y_int_train, num_classes=n_unique_classes)
    y_test = to_categorical(y_int_test, num_classes=n_unique_classes)   
    
    # Normalize the data 
    # create scaler
    X_scaler = MinMaxScaler()
    # fit scaler on train data
    X_scaler.fit(X_train)
    # apply transform to both train and test data 
    X_train = X_scaler.transform(X_train).copy()
    X_test = X_scaler.transform(X_test).copy()
    # Save the scalee to use when using eval set 
    joblib.dump(X_scaler, model_save_path+'X_scaler.pkl')

    #configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    #session = tf.compat.v1.Session(config=configuration)
    ## evaluate model
    start_time = time.time()
    PARAMETERS['train_size'] = X_train.shape[0]
    PARAMETERS['test_size'] = X_test.shape[0]
    if PARAMETERS['model_type'] == 'lstm' or 'mlp':
        with tf.device(PARAMETERS['device']):
            results, history, scores = evaluate_NN_model(X_train, y_train, X_test, y_test, 
                                      model_save_path, PARAMETERS)
    elif PARAMETERS['model_type'] == 'naive_bayes':
        results, history = evaluate_naive_bayes_model(X_train, y_train, X_test, y_test, 
                                      model_save_path, 
                                      PARAMETERS['cross_validate'])
    else: 
        print('ERROR: Not one of the specified models')
    end_time = time.time()
    print("Runtime: " + str(end_time-start_time) + " seconds")
    PARAMETERS['runtime'] = end_time-start_time  
    PARAMETERS['accuracy'] = mean(results['accuracy'])
    PARAMETERS['roc_auc'] = mean(results['roc_auc'])
    PARAMETERS['f1score'] = mean(results['f1score'])
    
    tell_human_im_done()

    # save parameters of the model and training in respective folders
    with open(model_save_path+'learning_history.txt', 'w') as file:
        file.write(json.dumps(history.history))

    # also append parameters to a file to keep track of what has been run  
    if append_model_parameters:
        with open(out_path_pdlogger, 'a+') as file:
            dictwriter_object = DictWriter(file, fieldnames=PARAMETERS.keys())
            dictwriter_object.writeheader()
            dictwriter_object.writerow(PARAMETERS)
            file.close()

    # summarize history for loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.figure(2)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # To get predictions from scaled data
    # inverse transform
    #inverse = scaler.inverse_transform(normalized) 
