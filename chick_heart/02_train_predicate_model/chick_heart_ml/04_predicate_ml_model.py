# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

Load model and predict using load_model

# 6 parameters to set: surr_type=str, simples=int, ID_train=str, ID_test=str, ML_model=str, repeats=int

@author: Zhiqin Ma
"""

import os
import sys
import pandas as pd
import numpy as np
from numpy import array
from pandas import read_csv
from joblib import load
# import warnings
# warnings.filterwarnings("ignore")

print('------------------------------ 04 Start predicate ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)
# ID_train = '8'                    ## (Set it up to your needs)
# ID_test = '14'                    ## (Set it up to your needs)
# ML_model = 'SVM_model'            ## (Set it up to your needs)
# # ML_model = 'Bagging_model'      ## (Set it up to your needs)
# # ML_model = 'RF_model'           ## (Set it up to your needs)
# # ML_model = 'GBM_model'          ## (Set it up to your needs)
# # ML_model = 'Xgboost_model'      ## (Set it up to your needs)
# # ML_model = 'LGBM_model'         ## (Set it up to your needs)
# repeats = 10                      ## (Set it up to your needs)

# Setting parameters
surr_type = str(sys.argv[1])        ## (Set it up to your needs)
simples = int(sys.argv[2])          ## (Set it up to your needs)
ID_train = str(sys.argv[3])         ## (Set it up to your needs)
ID_test = str(sys.argv[4])          ## (Set it up to your needs)
ML_model = str(sys.argv[5])         ## (Set it up to your needs)
repeats = int(sys.argv[6])          ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('ID_test:', ID_test)
print('ML_model:', ML_model)
print('repeats:', repeats)
print()


try:
    os.mkdir('04_prediction_data')
except:
    print('04_prediction_data directory already exists!')


# define the location of the dataset
full_path_test_X = '03_sliding_window_data/sliding_window_features_surrogate_{}_simples_{}_test_{}.csv'.format(surr_type, simples, ID_test)
# load data
df_X_test = read_csv(full_path_test_X, header=None)
# review shape
print('df_X_test.shape', df_X_test.shape)

# # 1. Transform DataFrame objects to array objects = 1. 将df对象转换为array对象
# array_X_test = array(df_X_test)
# # 2. Transform array to list = 2. 将array转换为list
# list_X_test = array_X_test.tolist()

# prediction
print('----------------------- prediction ... -----------------------')

df_all_predict_probability = pd.DataFrame()
for r in range(repeats):
    # Define the filepath for the best model
    filepath = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, r)
    # load the model from disk
    loaded_model = load(filepath)

    # prediction probabilities
    y_pred_proba = loaded_model.predict_proba(df_X_test)
    # print(y_pred_proba[:, 1])

    # Save prediction probabilities
    df_all_predict_probability['repeat_{}'.format(r)] = y_pred_proba[:, 1]
    # print('y_pred_list:', df_all_predict_probability)
    print('y_pred_list.shape:', df_all_predict_probability.shape)

# Compute mean
mean_y_pred = np.mean(df_all_predict_probability, axis=1)
# Calculate the length of the error bar
std_y_pred = np.std(df_all_predict_probability, axis=1)
error = 1.96 * std_y_pred / np.sqrt(mean_y_pred.shape[0])

# reversion from numpy into dataframe
predict_probability_df_mean = pd.DataFrame(mean_y_pred)
predict_probability_df_error = pd.DataFrame(error)

# save results
predict_probability_df_mean.to_csv("04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type,
                                                                                                  simples, ID_test,
                                                                                                  ML_model), sep=',', header=False, index=False)
predict_probability_df_error.to_csv("04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type,
                                                                                                                                      simples, ID_test,
                                                                                                                                      ML_model), sep=',', header=False, index=False)

print('------------------------------ 04 Completed predicate ------------------------------')
print()
