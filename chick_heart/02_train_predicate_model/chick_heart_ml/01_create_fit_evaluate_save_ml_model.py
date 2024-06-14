# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

Using ML model, repeat 10 times and save models

# 5 parameters to set:  surr_type=str, simples=int,
                        ID_train=str, ML_model=str, repeats=int

@author: Zhiqin Ma
"""

# Start timer to record execution time
import time
start_time = time.time()

import os
import sys
import numpy as np
import pandas as pd
from numpy import mean, std
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump
from joblib import load
# import warnings
# warnings.filterwarnings("ignore")

print('------------------------------ 01 Start run ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)
# ID_train = '8'                    ## (Set it up to your needs)
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
ML_model = str(sys.argv[4])         ## (Set it up to your needs)
repeats = int(sys.argv[5])          ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('ML_model:', ML_model)
print('repeats:', repeats)
print()


try:
    os.mkdir('01_evaluation_data')
except:
    print('01_evaluation_data directory already exists!')

try:
    os.mkdir('models')
except:
    print('models directory already exists!\n')

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


# create, fit and evaluate ML model
def SVM_model(trainX, trainy, number):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    # define model to evaluate: SVC
    model = SVC(probability=True)
    # MinMaxScaler then fit model
    # pipeline = Pipeline(steps=[('m', model)])
    pipeline = Pipeline(steps=[('s', MinMaxScaler()), ('m', model)])
    # fit the model
    pipeline.fit(trainX, trainy)
    # define path + filename
    filename = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, number)
    # save the model to disk # # 将模型保存到磁盘
    dump(pipeline, filename)

    # evaluate model
    # Define cross validation method = 定义交叉验证方法
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Compute cross-validation scores = 计算交叉验证分数
    score = cross_val_score(pipeline, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1)
    return score

# create, fit and evaluate ML model
def Bagging_model(trainX, trainy, number):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    # define model to evaluate: Bagging
    model = BaggingClassifier(n_estimators=100)
    # MinMaxScaler then fit model
    # pipeline = Pipeline(steps=[('m', model)])
    pipeline = Pipeline(steps=[('s', MinMaxScaler()), ('m', model)])
    # fit the model
    pipeline.fit(trainX, trainy)
    # define path + filename
    filename = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, number)
    # save the model to disk # # 将模型保存到磁盘
    dump(pipeline, filename)

    # evaluate model
    # Define cross validation method = 定义交叉验证方法
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Compute cross-validation scores = 计算交叉验证分数
    score = cross_val_score(pipeline, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1)
    return score


def RF_model(trainX, trainy, number):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    # define model to evaluate: RF
    model = RandomForestClassifier(n_estimators=100)
    # MinMaxScaler then fit model
    # pipeline = Pipeline(steps=[('m', model)])  # LGBM 1.000 (0.000)
    pipeline = Pipeline(steps=[('s', MinMaxScaler()), ('m', model)])
    # fit the model
    pipeline.fit(trainX, trainy)
    # define path + filename
    filename = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, number)
    # save the model to disk # # 将模型保存到磁盘
    dump(pipeline, filename)

    # evaluate model
    # Define cross validation method = 定义交叉验证方法
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Compute cross-validation scores = 计算交叉验证分数
    score = cross_val_score(pipeline, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1)
    return score


def GBM_model(trainX, trainy, number):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    # define model to evaluate: GBM
    model = GradientBoostingClassifier(n_estimators=100)
    # MinMaxScaler then fit model
    # pipeline = Pipeline(steps=[('m', model)])
    pipeline = Pipeline(steps=[('s', MinMaxScaler()), ('m', model)])
    # fit the model
    pipeline.fit(trainX, trainy)
    # define path + filename
    filename = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, number)
    # save the model to disk # # 将模型保存到磁盘
    dump(pipeline, filename)

    # evaluate model
    # Define cross validation method = 定义交叉验证方法
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Compute cross-validation scores = 计算交叉验证分数
    score = cross_val_score(pipeline, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1)
    return score


def Xgboost_model(trainX, trainy, number):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    # define model to evaluate: Xgboost
    model = XGBClassifier()
    # MinMaxScaler then fit model
    # pipeline = Pipeline(steps=[('m', model)])
    pipeline = Pipeline(steps=[('s', MinMaxScaler()), ('m', model)])
    # fit the model
    pipeline.fit(trainX, trainy)
    # define path + filename
    filename = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, number)
    # save the model to disk # # 将模型保存到磁盘
    dump(pipeline, filename)

    # evaluate model
    # Define cross validation method = 定义交叉验证方法
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Compute cross-validation scores = 计算交叉验证分数
    score = cross_val_score(pipeline, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1)
    return score


def LGBM_model(trainX, trainy, number):
    # Setting parameters
    n_splits, n_repeats, random_state = 5, 3, 1
    # define model to evaluate: LGBM
    model = LGBMClassifier()
    # MinMaxScaler then fit model
    # pipeline = Pipeline(steps=[('m', model)])
    pipeline = Pipeline(steps=[('s', MinMaxScaler()), ('m', model)])
    # fit the model
    pipeline.fit(trainX, trainy)
    # define path + filename
    filename = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, number)
    # save the model to disk # # 将模型保存到磁盘
    dump(pipeline, filename)

    # evaluate model
    # Define cross validation method = 定义交叉验证方法
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    # Compute cross-validation scores = 计算交叉验证分数
    score = cross_val_score(pipeline, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1)
    return score

if __name__ == '__main__':
    # define the location of the dataset
    full_path = 'G:/surrogate_data_based_machine_learning/chick_heart/01_data_preprocessing/02_extracte_and_generate_surrogate_dataset/data/merge_surrogate_{}_simples_{}_train_{}.csv'.format(surr_type, simples, ID_train)
    # full_path = '../../01_data_preprocessing/02_extracte_and_generate_surrogate_dataset/data/merge_surrogate_{}_simples_{}_train_{}.csv'.format(surr_type, simples, ID_train)
    print('full_path=', full_path)

    # load the dataset
    X, y = load_dataset(full_path)
    # Split the training and test datasets = 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    # Transforms a NumPy array into a DataFrame = 将 NumPy 数组转换为 DataFrame
    df_X_test = pd.DataFrame(X_test)
    df_y_test = pd.DataFrame(y_test)
    # Save test data to compute ROC and AUC
    df_test = pd.concat([df_X_test, df_y_test], axis=1, ignore_index=True)
    # save dataframe file
    df_test.to_csv("01_evaluation_data/merge_test_surrogate_{}_simples_{}_train_{}.csv".format(surr_type, simples, ID_train),
                   sep=',', header=False, index=False)

    # select ML model
    if ML_model == 'SVM_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = SVM_model(X_train, y_train, str(r))
            score = score * 100.0
            print('>#{}: {}'.format(r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("ML_model:", ML_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif ML_model == 'Bagging_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = Bagging_model(X_train, y_train, str(r))
            score = score * 100.0
            print('>#{}: {}'.format(r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("ML_model:", ML_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif ML_model == 'RF_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = RF_model(X_train, y_train, str(r))
            score = score * 100.0
            print('>#{}: {}'.format(r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("ML_model:", ML_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif ML_model == 'GBM_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = GBM_model(X_train, y_train, str(r))
            score = score * 100.0
            print('>#{}: {}'.format(r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("ML_model:", ML_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif ML_model == 'Xgboost_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = Xgboost_model(X_train, y_train, str(r))
            score = score * 100.0
            print('>#{}: {}'.format(r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("ML_model:", ML_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif ML_model == 'LGBM_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = LGBM_model(X_train, y_train, str(r))
            score = score * 100.0
            print('>#{}: {}'.format(r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("ML_model:", ML_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    else:
        print('\n''-------------------- Abnormal exit 1, try again input DL_model --------------------')
        exit(1)

    # Stop timer
    end_time = time.time()
    print('\n''Running time {:.1f} seconds'.format(end_time - start_time))

    print('------------------------------ 01 Completed ------------------------------')
    print()
