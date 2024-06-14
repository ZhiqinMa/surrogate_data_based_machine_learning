# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

Compute ROC、AUC

# 5 parameters to set: surr_type=str, simples=int, ID_train=str, DL_model=str, repeats=int

@author: Zhiqin Ma
"""

# Start timer to record execution time
import time
start_time = time.time()

import os
import sys
import statistics
from numpy import mean, std, expand_dims
from pandas import read_csv
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import warnings
# warnings.filterwarnings("ignore")

print('------------------------------ 02 Start Evaluation ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                      ## (Set it up to your needs)
# simples = 1000                          ## (Set it up to your needs)
# ID_train = 'BMIII_PI'                   ## (Set it up to your needs)
# DL_model = 'CNN_model'                  ## (Set it up to your needs)
# # DL_model = 'three_head_CNN_model'     ## (Set it up to your needs)
# # DL_model = 'LSTM_model'               ## (Set it up to your needs)
# # DL_model = 'CNN_LSTM_model'           ## (Set it up to your needs)
# # DL_model = 'ConvLSTM_model'           ## (Set it up to your needs)
# repeats = 10                            ## (Set it up to your needs)

# Setting parameters
surr_type = str(sys.argv[1])              ## (Set it up to your needs)
simples = int(sys.argv[2])                ## (Set it up to your needs)
ID_train = str(sys.argv[3])               ## (Set it up to your needs)
DL_model = str(sys.argv[4])               ## (Set it up to your needs)
repeats = int(sys.argv[5])                ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('DL_model:', DL_model)
print('repeats:', repeats)
print()


try:
    os.mkdir('02_evaluation_performance')
except:
    print('02_evaluation_performance directory already exists!\n')


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


### ----------------------------- start code -----------------------------
if __name__ == '__main__':
    # define the location of the dataset
    full_path_test = '01_evaluation_data/merge_test_surrogate_{}_simples_{}_train_{}.csv'.format(surr_type, simples, ID_train)
    # load the dataset
    X_test, y_test = load_dataset(full_path_test)

    # # review shape
    # print('X_test1.shape', X_test.shape)
    # print('y_test1.shape', y_test.shape)
    # print('X_test1.dtype', X_test.dtype)
    # print('y_test1.dtype', y_test.dtype)

    # expand_dims features
    X_test = expand_dims(X_test, axis=-1)
    y_test = expand_dims(y_test, axis=-1)
    # Define shape
    X_test_shape0 = X_test.shape[0]
    X_test_shape1 = X_test.shape[1]
    X_test_shape2 = X_test.shape[2]

    # Setting parameters
    batch_size, verbose = 64, 0
    # select DL model
    accuracys, f1s, precisions, recalls = [], [], [], []
    if DL_model == 'CNN_model':
        for r in range(repeats):
            # Define the filepath for the best model
            filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
            # Load the best model
            best_model = load_model(filepath)

            # Evaluate the best model on the validation  data
            _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
            print('# {} >> accuracy:'.format(r), accuracy)

            # Predict probabilities for the testX data
            test_predicts = best_model.predict(X_test, verbose=verbose)
            # print('y_test=', y_test)
            # print('test_predicts=', test_predicts)
            #
            # # Review y_test and predict_proba_list
            # print('y_test: \n', y_test[:, 0])
            # print('test_predicts: \n', test_predicts[:, 0])
            # print('----------------------------')
            # print('y_test.shape=', y_test.shape)
            # print('test_predicts.shape=', test_predicts.shape)

            # # Convert probabilities to class labels = 将概率转换为类别标签
            # y_pred_labels = np.array([1 if prob > 0.5 else 0 for prob in test_predicts[:, 0]])
            y_pred_labels = np.array([1 if prob[0] > 0.5 else 0 for prob in test_predicts])
            # print('y_pred_labels=', y_pred_labels)

            f1 = f1_score(y_test, y_pred_labels)
            precision = precision_score(y_test, y_pred_labels)
            recall = recall_score(y_test, y_pred_labels)

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

        file_path = '02_evaluation_performance/evaluation_performance_surrogate_{}_simples_{}_{}_train_{}.txt'.format(
            surr_type, simples, DL_model, ID_train)
        evaluation_results = '------------------ Evaluation results -------------------------\n'
        evaluation_results += f"Accuracy: Mean = {accuracys_mean}, Std = {accuracys_std}\n" \
                              f"F1 Score: Mean = {f1s_mean}, Std = {f1s_std}\n" \
                              f"Precision: Mean = {precisions_mean}, Std = {precisions_std}\n" \
                              f"Recall: Mean = {recalls_mean}, Std = {recalls_std}\n"
        # Open the file for writing = 打开文件准备写入
        with open(file_path, 'w') as file:
            file.write(evaluation_results)
        print("The results have been saved to a txt file:", file_path)

    elif DL_model == 'three_head_CNN_model':
        for r in range(repeats):
            # Define the filepath for the best model
            filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
            # Load the best model
            best_model = load_model(filepath)

            # Evaluate the best model on the validation  data
            _, accuracy = best_model.evaluate([X_test, X_test, X_test], y_test, batch_size=batch_size, verbose=verbose)
            print('# {} >> accuracy:'.format(r), accuracy)

            # Predict probabilities for the testX data
            test_predicts = best_model.predict([X_test, X_test, X_test], verbose=verbose)
            # print('y_test=', y_test)
            # print('test_predicts=', test_predicts)

            # # Convert probabilities to class labels = 将概率转换为类别标签
            # y_pred_labels = np.array([1 if prob > 0.5 else 0 for prob in test_predicts[:, 0]])
            y_pred_labels = np.array([1 if prob[0] > 0.5 else 0 for prob in test_predicts])
            # print('y_pred_labels=', y_pred_labels)

            f1 = f1_score(y_test, y_pred_labels)
            precision = precision_score(y_test, y_pred_labels)
            recall = recall_score(y_test, y_pred_labels)

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

        file_path = '02_evaluation_performance/evaluation_performance_surrogate_{}_simples_{}_{}_train_{}.txt'.format(
            surr_type, simples, DL_model, ID_train)
        evaluation_results = '------------------ Evaluation results -------------------------\n'
        evaluation_results += f"Accuracy: Mean = {accuracys_mean}, Std = {accuracys_std}\n" \
                              f"F1 Score: Mean = {f1s_mean}, Std = {f1s_std}\n" \
                              f"Precision: Mean = {precisions_mean}, Std = {precisions_std}\n" \
                              f"Recall: Mean = {recalls_mean}, Std = {recalls_std}\n"
        # Open the file for writing = 打开文件准备写入
        with open(file_path, 'w') as file:
            file.write(evaluation_results)
        print("The results have been saved to a txt file:", file_path)

    elif DL_model == 'LSTM_model':
        for r in range(repeats):
            # Define the filepath for the best model
            filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
            # Load the best model
            best_model = load_model(filepath)

            # Evaluate the best model on the validation  data
            _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
            print('# {} >> accuracy:'.format(r), accuracy)

            # Predict probabilities for the testX data
            test_predicts = best_model.predict(X_test, verbose=verbose)
            # print('y_test=', y_test)
            # print('test_predicts=', test_predicts)

            # # Convert probabilities to class labels = 将概率转换为类别标签
            # y_pred_labels = np.array([1 if prob > 0.5 else 0 for prob in test_predicts[:, 0]])
            y_pred_labels = np.array([1 if prob[0] > 0.5 else 0 for prob in test_predicts])
            # print('y_pred_labels=', y_pred_labels)

            f1 = f1_score(y_test, y_pred_labels)
            precision = precision_score(y_test, y_pred_labels)
            recall = recall_score(y_test, y_pred_labels)

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

        file_path = '02_evaluation_performance/evaluation_performance_surrogate_{}_simples_{}_{}_train_{}.txt'.format(
            surr_type, simples, DL_model, ID_train)
        evaluation_results = '------------------ Evaluation results -------------------------\n'
        evaluation_results += f"Accuracy: Mean = {accuracys_mean}, Std = {accuracys_std}\n" \
                              f"F1 Score: Mean = {f1s_mean}, Std = {f1s_std}\n" \
                              f"Precision: Mean = {precisions_mean}, Std = {precisions_std}\n" \
                              f"Recall: Mean = {recalls_mean}, Std = {recalls_std}\n"
        # Open the file for writing = 打开文件准备写入
        with open(file_path, 'w') as file:
            file.write(evaluation_results)
        print("The results have been saved to a txt file:", file_path)

    elif DL_model == 'CNN_LSTM_model':
        for r in range(repeats):
            # Define the filepath for the best model
            filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
            # Load the best model
            best_model = load_model(filepath)

            # reshape into subsequences (samples, steps, length, channels)
            n_steps = 10
            n_length = int(X_test_shape1 / n_steps)
            n_features = X_test_shape2
            X_test = X_test.reshape((X_test_shape0, n_steps, n_length, 1))

            # Evaluate the best model on the validation  data
            _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
            print('# {} >> accuracy:'.format(r), accuracy)

            # Predict probabilities for the testX data
            test_predicts = best_model.predict(X_test, verbose=verbose)
            # print('y_test=', y_test)
            # print('test_predicts=', test_predicts)

            # # Convert probabilities to class labels = 将概率转换为类别标签
            # y_pred_labels = np.array([1 if prob > 0.5 else 0 for prob in test_predicts[:, 0]])
            y_pred_labels = np.array([1 if prob[0] > 0.5 else 0 for prob in test_predicts])
            # print('y_pred_labels=', y_pred_labels)

            f1 = f1_score(y_test, y_pred_labels)
            precision = precision_score(y_test, y_pred_labels)
            recall = recall_score(y_test, y_pred_labels)

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

        file_path = '02_evaluation_performance/evaluation_performance_surrogate_{}_simples_{}_{}_train_{}.txt'.format(
            surr_type, simples, DL_model, ID_train)
        evaluation_results = '------------------ Evaluation results -------------------------\n'
        evaluation_results += f"Accuracy: Mean = {accuracys_mean}, Std = {accuracys_std}\n" \
                              f"F1 Score: Mean = {f1s_mean}, Std = {f1s_std}\n" \
                              f"Precision: Mean = {precisions_mean}, Std = {precisions_std}\n" \
                              f"Recall: Mean = {recalls_mean}, Std = {recalls_std}\n"
        # Open the file for writing = 打开文件准备写入
        with open(file_path, 'w') as file:
            file.write(evaluation_results)
        print("The results have been saved to a txt file:", file_path)

    elif DL_model == 'ConvLSTM_model':
        for r in range(repeats):
            # Define the filepath for the best model
            filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, r)
            # Load the best model
            best_model = load_model(filepath)

            # reshape into subsequences (samples, time_steps, rows, cols, channels)
            n_steps = 10
            n_length = int(X_test_shape1 / n_steps)
            n_features = X_test_shape2
            X_test = X_test.reshape((X_test_shape0, n_steps, 1, n_length, n_features))

            # Evaluate the best model on the validation  data
            _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
            print('# {} >> accuracy:'.format(r), accuracy)

            # Predict probabilities for the testX data
            test_predicts = best_model.predict(X_test, verbose=verbose)
            # print('y_test=', y_test)
            # print('test_predicts=', test_predicts)

            # # Convert probabilities to class labels = 将概率转换为类别标签
            # y_pred_labels = np.array([1 if prob > 0.5 else 0 for prob in test_predicts[:, 0]])
            y_pred_labels = np.array([1 if prob[0] > 0.5 else 0 for prob in test_predicts])
            # print('y_pred_labels=', y_pred_labels)

            f1 = f1_score(y_test, y_pred_labels)
            precision = precision_score(y_test, y_pred_labels)
            recall = recall_score(y_test, y_pred_labels)

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

        file_path = '02_evaluation_performance/evaluation_performance_surrogate_{}_simples_{}_{}_train_{}.txt'.format(
            surr_type, simples, DL_model, ID_train)
        evaluation_results = '------------------ Evaluation results -------------------------\n'
        evaluation_results += f"Accuracy: Mean = {accuracys_mean}, Std = {accuracys_std}\n" \
                              f"F1 Score: Mean = {f1s_mean}, Std = {f1s_std}\n" \
                              f"Precision: Mean = {precisions_mean}, Std = {precisions_std}\n" \
                              f"Recall: Mean = {recalls_mean}, Std = {recalls_std}\n"
        # Open the file for writing = 打开文件准备写入
        with open(file_path, 'w') as file:
            file.write(evaluation_results)
        print("The results have been saved to a txt file:", file_path)

    else:
        print('\n''-------------------- Abnormal exit 1, try again input DL_model --------------------')
        exit(1)


    # Stop timer
    end_time = time.time()
    print('\n''Running time {:.1f} seconds'.format(end_time - start_time))

    print('------------------------------ 02 Completed Evaluation ------------------------------')
    print()
