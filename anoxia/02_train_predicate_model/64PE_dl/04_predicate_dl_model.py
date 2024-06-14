# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

Load model and predict using load_model

# 6 parameters to set: surr_type=str, simples=int, ID_train=str, ID_test=str, DL_model=str, repeats=int

@author: Zhiqin Ma
"""

import os
import sys
import pandas as pd
import numpy as np
from numpy import array
from pandas import read_csv
from keras.models import load_model
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")


print('------------------------------ 04 Start predicate ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                        ## (Set it up to your needs)
# simples = 1000                            ## (Set it up to your needs)
# ID_train = 'S9_S8_S7'                     ## (Set it up to your needs)
# ID_test = 'S3'                            ## (Set it up to your needs)
# DL_model = 'CNN_model'                    ## (Set it up to your needs)
# # DL_model = 'three_head_CNN_model'       ## (Set it up to your needs)
# # DL_model = 'LSTM_model'                 ## (Set it up to your needs)
# # DL_model = 'CNN_LSTM_model'             ## (Set it up to your needs)
# # DL_model = 'ConvLSTM_model'             ## (Set it up to your needs)
# repeats = 10                              ## (Set it up to your needs)


# Setting parameters
surr_type = str(sys.argv[1])                ## (Set it up to your needs)
simples = int(sys.argv[2])                  ## (Set it up to your needs)
ID_train = str(sys.argv[3])                 ## (Set it up to your needs)
ID_test = str(sys.argv[4])                  ## (Set it up to your needs)
DL_model = str(sys.argv[5])                 ## (Set it up to your needs)
repeats = int(sys.argv[6])                  ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('ID_test:', ID_test)
print('DL_model:', DL_model)
print('repeats:', repeats)
print()


try:
    os.mkdir('04_prediction_data')
except:
    print('04_prediction_data directory already exists!')



# define the location of the dataset
full_path_test_X = '03_sliding_window_data/sliding_window_features_surrogate_{}_simples_{}_test_{}.csv'.format(surr_type, simples, ID_test)
# load data
X_test = read_csv(full_path_test_X, header=None)
# review shape
print('X_test.shape', X_test.shape)

# convert dataframe to np.array
X_test_array = array(X_test)
# reshape (samples, timesteps, features) =  (样本数, 时间步长, 特征)
X_test_array = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1], 1)
# X_text_array = expand_dims(X_text_array, axis=-1)
# review shape
print('X_test_array.shape: ', X_test_array.shape)

# prediction
print('--------------- prediction ... ---------------')

# Setting parameters
batch_size, verbose = 64, 0
# select DL model
if DL_model == 'CNN_model':
    y_pred_list = []
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
        # load the model from disk
        best_model = load_model(filepath)
        # prediction results = 预测结果
        y_pred = best_model.predict(X_test_array, batch_size=batch_size, verbose=verbose)
        # print('y_pred: ', y_pred)
        # append predicted results to y_pred_list
        y_pred_list.append(y_pred)
    # merge array via column
    y_pred_list = np.column_stack(y_pred_list)
    print('y_pred_list.shape', y_pred_list.shape)

    # Compute mean
    mean_y_pred = np.mean(y_pred_list, axis=1)
    # Calculate the length of the error bar
    std_y_pred = np.std(y_pred_list, axis=1)
    error = 1.96 * std_y_pred / np.sqrt(mean_y_pred.shape[0])
    # reversion from numpy into dataframe
    predict_probability_df_mean = pd.DataFrame(mean_y_pred)
    predict_probability_df_error = pd.DataFrame(error)
    # save results
    predict_probability_df_mean.to_csv(
        "04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)
    predict_probability_df_error.to_csv(
        "04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)

    # # Plot the error bar
    # plt.errorbar(range(len(mean_y_pred)), mean_y_pred, yerr=error, fmt='-o')
    # plt.show()
elif DL_model == 'three_head_CNN_model':
    y_pred_list = []
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
        # load the model from disk
        best_model = load_model(filepath)
        # prediction results = 预测结果
        y_pred = best_model.predict([X_test_array, X_test_array, X_test_array], batch_size=64, verbose=verbose)
        # print('y_pred: ', y_pred)
        # append predicted results to y_pred_list
        y_pred_list.append(y_pred)
    # merge array via column
    y_pred_list = np.column_stack(y_pred_list)
    print('y_pred_list.shape', y_pred_list.shape)

    # Compute mean
    mean_y_pred = np.mean(y_pred_list, axis=1)
    # Calculate the length of the error bar
    std_y_pred = np.std(y_pred_list, axis=1)
    error = 1.96 * std_y_pred / np.sqrt(mean_y_pred.shape[0])
    # reversion from numpy into dataframe
    predict_probability_df_mean = pd.DataFrame(mean_y_pred)
    predict_probability_df_error = pd.DataFrame(error)
    # save results
    predict_probability_df_mean.to_csv(
        "04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)
    predict_probability_df_error.to_csv(
        "04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)

    # # Plot the error bar
    # plt.errorbar(range(len(mean_y_pred)), mean_y_pred, yerr=error, fmt='-o')
    # plt.show()
elif DL_model == 'LSTM_model':
    y_pred_list = []
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
        # load the model from disk
        best_model = load_model(filepath)
        # prediction results = 预测结果
        y_pred = best_model.predict(X_test_array, batch_size=batch_size, verbose=verbose)
        # print('y_pred: ', y_pred)
        # append predicted results to y_pred_list
        y_pred_list.append(y_pred)
    # merge array via column
    y_pred_list = np.column_stack(y_pred_list)
    print('y_pred_list.shape', y_pred_list.shape)

    # Compute mean
    mean_y_pred = np.mean(y_pred_list, axis=1)
    # Calculate the length of the error bar
    std_y_pred = np.std(y_pred_list, axis=1)
    error = 1.96 * std_y_pred / np.sqrt(mean_y_pred.shape[0])
    # reversion from numpy into dataframe
    predict_probability_df_mean = pd.DataFrame(mean_y_pred)
    predict_probability_df_error = pd.DataFrame(error)
    # save results
    predict_probability_df_mean.to_csv(
        "04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)
    predict_probability_df_error.to_csv(
        "04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)

    # # Plot the error bar
    # plt.errorbar(range(len(mean_y_pred)), mean_y_pred, yerr=error, fmt='-o')
    # plt.show()
elif DL_model == 'CNN_LSTM_model':
    y_pred_list = []
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
        # load the model from disk
        best_model = load_model(filepath)
        # reshape into subsequences (samples, steps, length, channels)
        n_steps = 10
        n_length = int(X_test_array.shape[1] / n_steps)
        textX = X_test_array.reshape((X_test_array.shape[0], n_steps, n_length, 1))
        print('textX=(samples, steps, length, channels):', textX.shape)
        # prediction results = 预测结果
        y_pred = best_model.predict(textX, batch_size=batch_size, verbose=verbose)
        # print('y_pred: ', y_pred)
        # append predicted results to y_pred_list
        y_pred_list.append(y_pred)
    # merge array via column
    y_pred_list = np.column_stack(y_pred_list)
    print('y_pred_list.shape', y_pred_list.shape)

    # Compute mean
    mean_y_pred = np.mean(y_pred_list, axis=1)
    # Calculate the length of the error bar
    std_y_pred = np.std(y_pred_list, axis=1)
    error = 1.96 * std_y_pred / np.sqrt(mean_y_pred.shape[0])
    # reversion from numpy into dataframe
    predict_probability_df_mean = pd.DataFrame(mean_y_pred)
    predict_probability_df_error = pd.DataFrame(error)
    # save results
    predict_probability_df_mean.to_csv(
        "04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)
    predict_probability_df_error.to_csv(
        "04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)

    # # Plot the error bar
    # plt.errorbar(range(len(mean_y_pred)), mean_y_pred, yerr=error, fmt='-o')
    # plt.show()
elif DL_model == 'ConvLSTM_model':
    y_pred_list = []
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
        # load the model from disk
        best_model = load_model(filepath)
        # reshape into subsequences (samples, time steps, rows, cols, channels)
        n_steps = 10
        n_length = int(X_test_array.shape[1] / n_steps)
        textX = X_test_array.reshape((X_test_array.shape[0], n_steps, 1, n_length, 1))
        print('textX=(samples, time steps, rows, cols, channels):', textX.shape)
        # prediction results = 预测结果
        y_pred = best_model.predict(textX, batch_size=batch_size, verbose=verbose)
        # print('y_pred: ', y_pred)
        # append predicted results to y_pred_list
        y_pred_list.append(y_pred)
    # merge array via column
    y_pred_list = np.column_stack(y_pred_list)
    print('y_pred_list.shape', y_pred_list.shape)

    # Compute mean
    mean_y_pred = np.mean(y_pred_list, axis=1)
    # Calculate the length of the error bar
    std_y_pred = np.std(y_pred_list, axis=1)
    error = 1.96 * std_y_pred / np.sqrt(mean_y_pred.shape[0])
    # reversion from numpy into dataframe
    predict_probability_df_mean = pd.DataFrame(mean_y_pred)
    predict_probability_df_error = pd.DataFrame(error)
    # save results
    predict_probability_df_mean.to_csv(
        "04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)
    predict_probability_df_error.to_csv(
        "04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model),
        sep=',', header=False, index=False)

    # # Plot the error bar
    # plt.errorbar(range(len(mean_y_pred)), mean_y_pred, yerr=error, fmt='-o')
    # plt.show()
else:
    print('\n''-------------------- Abnormal exit 1, try again input DL_model --------------------')
    exit(1)

print('------------------------------ 04 Completed predicate ------------------------------')
print()
