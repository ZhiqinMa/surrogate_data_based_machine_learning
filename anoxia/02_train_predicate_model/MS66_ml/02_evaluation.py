# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

Evaluation F1 score, sensitivity (precision), specificity (recall) on model

# 5 parameters to set: surr_type=str, simples=int, ID_train=str, ML_model=str, repeats=int

@author: Zhiqin Ma
"""

# Start timer to record execution time
import time
start_time = time.time()

import os
import sys
import ewstools
import statistics
from numpy import mean, std, expand_dims
from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from joblib import load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import warnings
# warnings.filterwarnings("ignore")

print('------------------------------ 02 Start Evaluation ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                  ## (Set it up to your needs)
# simples = 1000                      ## (Set it up to your needs)
# ID_train = 'S5_S4'                  ## (Set it up to your needs)
# ML_model = 'SVM_model'              ## (Set it up to your needs)
# # ML_model = 'Bagging_model'        ## (Set it up to your needs)
# # ML_model = 'RF_model'             ## (Set it up to your needs)
# # ML_model = 'GBM_model'            ## (Set it up to your needs)
# # ML_model = 'Xgboost_model'        ## (Set it up to your needs)
# # ML_model = 'LGBM_model'           ## (Set it up to your needs)
# repeats = 10                        ## (Set it up to your needs)

# Setting parameters
surr_type = str(sys.argv[1])          ## (Set it up to your needs)
simples = int(sys.argv[2])            ## (Set it up to your needs)
ID_train = str(sys.argv[3])           ## (Set it up to your needs)
ML_model = str(sys.argv[4])           ## (Set it up to your needs)
repeats = int(sys.argv[5])            ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('ML_model:', ML_model)
print('repeats:', repeats)
print()


try:
    os.mkdir('02_evaluation_performance')
except:
    print('02_evaluation_performance directory already exists!')


# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    # label encode the target variable to have the classes 0 and 1
    # y = LabelEncoder().fit_transform(y)
    return X, y

def evaluation_model(X_test, y_test):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    predict_probability_df = pd.DataFrame()
    scores = list()
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, r)
        # load the model from disk
        loaded_model = load(filepath)
        # # evaluate model
        # Define cross validation method = 定义交叉验证方法
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        # Compute cross-validation scores = 计算交叉验证分数
        score = cross_val_score(loaded_model, X_test, y_test, scoring='roc_auc', cv=cv, n_jobs=-1)
        score = score * 100
        print('# {} >> score:{}:'.format(r + 1, score))
        scores.append(score)
    # summarize results
    print('------------------ summarize results -------------------------')
    print("ML_model:", ML_model)
    print('scores:', scores)
    print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))



### ----------------------------- start code -----------------------------
if __name__ == '__main__':
    # define the location of the dataset
    full_path_test = '01_evaluation_data/merge_test_surrogate_{}_simples_{}_train_{}.csv'.format(surr_type, simples, ID_train)
    # load the dataset
    X_test, y_test = load_dataset(full_path_test)

    # # Evaluation F1 score, sensitivity ( precision), specificity (recall) on model
    accuracys, f1s, precisions, recalls = [], [], [], []
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, r)
        # load the model from disk
        loaded_model = load(filepath)

        # Predict probabilities for the data
        y_pred = loaded_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        accuracys.append(accuracy)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    accuracys_mean = mean(accuracys)
    accuracys_std = std(accuracys)
    f1s_mean = mean(f1s)
    f1s_std = std(f1s)
    precisions_mean = mean(precisions)
    precisions_std = std(precisions)
    recalls_mean = mean(recalls)
    recalls_std = std(recalls)

    file_path = '02_evaluation_performance/evaluation_performance_surrogate_{}_simples_{}_{}_train_{}.txt'.format(surr_type, simples, ML_model, ID_train)
    evaluation_results = '------------------ Evaluation results -------------------------\n'
    evaluation_results += f"Accuracy: Mean = {accuracys_mean}, Std = {accuracys_std}\n" \
                          f"F1 Score: Mean = {f1s_mean}, Std = {f1s_std}\n" \
                          f"Precision: Mean = {precisions_mean}, Std = {precisions_std}\n" \
                          f"Recall: Mean = {recalls_mean}, Std = {recalls_std}\n"
    # Open the file for writing = 打开文件准备写入
    with open(file_path, 'w') as file:
        file.write(evaluation_results)
    print("The results have been saved to a txt file:", file_path)


    # # Evaluation = 评估性能
    # evaluation_model(X_test, y_test)

    # Stop timer
    end_time = time.time()
    print('\n''Running time {:.1f} seconds'.format(end_time - start_time))

    print('------------------------------ 02 Completed Evaluation ------------------------------')
    print()