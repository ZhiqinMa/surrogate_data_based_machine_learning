# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

Using DL model, repeat 10 times and save the best model

# 6 parameters to set:  Data_folder=str, surr_type=str, simples=int,
                        ID_train=str, DL_model=str, repeats=int

@author: Zhiqin Ma
"""

# Start timer to record execution time of notebook
import time
start_time = time.time()
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'         # used to suppress the OpenMP library duplication error
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # used to enable dynamic GPU memory allocation
import sys
import numpy as np
import pandas as pd
from numpy import mean, std
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import LSTM, TimeDistributed, ConvLSTM2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# import warnings
# warnings.filterwarnings("ignore")

# 设置TensorFlow使用GPU
import tensorflow as tf
# Check whether the GPU is available = 检查 GPU 是否可用
if tf.config.experimental.list_physical_devices('GPU'):
    print("Using GPU for TensorFlow.")
else:
    print("GPU not available, using CPU for TensorFlow.")


print('------------------------------ 01 Start run ------------------------------')

# # Setting parameters
# Data_folder = '64PE'                    ## (Set it up to your needs)
# surr_type = 'AAFT'                      ## (Set it up to your needs)
# simples = 1000                          ## (Set it up to your needs)
# ID_train = 'S9_S8_S7'                   ## (Set it up to your needs)
# DL_model = 'CNN_model'                  ## (Set it up to your needs)
# # DL_model = 'three_head_CNN_model'     ## (Set it up to your needs)
# # DL_model = 'LSTM_model'               ## (Set it up to your needs)
# # DL_model = 'CNN_LSTM_model'           ## (Set it up to your needs)
# # DL_model = 'ConvLSTM_model'           ## (Set it up to your needs)
# repeats = 10                            ## (Set it up to your needs)

# Setting parameters
Data_folder = str(sys.argv[1])            ## (Set it up to your needs)
surr_type = str(sys.argv[2])              ## (Set it up to your needs)
simples = int(sys.argv[3])                ## (Set it up to your needs)
ID_train = str(sys.argv[4])               ## (Set it up to your needs)
DL_model = str(sys.argv[5])               ## (Set it up to your needs)
repeats = int(sys.argv[6])                ## (Set it up to your needs)

print('Data_folder:', Data_folder)
print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('DL_model:', DL_model)
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


# create, fit and evaluate DL model
def CNN_model(trainX, valX, trainy, valy, number):
    epochs, patience, batch_size, verbose = 100, 5, 64, 0
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='Graphviz_{}.png'.format(DL_model))

    # Define the filepath for saving the best model
    filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, number)

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    # Create an EarlyStopping callback that defines the stopping condition = 创建一个 EarlyStopping 回调函数，定义停止训练的条件
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max')

    # Fit the model with the training data and labels, and use the validation data for evaluation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping],
              validation_data=(valX, valy), verbose=verbose)

    # Load the best model
    best_model = load_model(filepath)

    # Evaluate the best model on the validation  data
    _, accuracy = best_model.evaluate(valX, valy, batch_size=batch_size, verbose=verbose)

    # save all models = 保存所有模型
    # best_model.save('models/surrogate_{}_simples_{}_{}_{}_2.h5'.format(surr_type, simples, DL_model, number))  # method 1 xx.h5
    return accuracy


# create, fit and evaluate DL model
def three_head_CNN_model(trainX, valX, trainy, valy, number):
    epochs, patience, batch_size, verbose = 100, 5, 64, 0
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # head 1
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv1 = Conv1D(filters=32, kernel_size=3, strides=1)(inputs1)
    # conv1 = Conv1D(filters=64, kernel_size=5, strides=1)(inputs1)
    # conv1 = Conv1D(filters=32, kernel_size=5, strides=1)(inputs1)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # head 2
    inputs2 = Input(shape=(n_timesteps, n_features))
    conv2 = Conv1D(filters=32, kernel_size=5, strides=1)(inputs2)
    # conv2 = Conv1D(filters=64, kernel_size=7, strides=1)(inputs2)
    # conv2 = Conv1D(filters=32, kernel_size=7, strides=1)(inputs2)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # head 3
    inputs3 = Input(shape=(n_timesteps, n_features))
    conv3 = Conv1D(filters=64, kernel_size=11, strides=1)(inputs3)
    # conv3 = Conv1D(filters=64, kernel_size=11, strides=1)(inputs3)
    # conv3 = Conv1D(filters=32, kernel_size=11, strides=1)(inputs3)
    bn3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(128, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='Graphviz_{}.png'.format(DL_model))

    # Define the filepath for saving the best model
    filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, number)

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    # Create an EarlyStopping callback that defines the stopping condition = 创建一个 EarlyStopping 回调函数，定义停止训练的条件
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max')

    # Fit the model with the training data and labels, and use the validation data for evaluation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit([trainX, trainX, trainX], trainy, epochs=epochs, batch_size=batch_size,
              callbacks=[checkpoint, early_stopping],
              validation_data=([valX, valX, valX], valy), verbose=verbose)

    # Load the best model
    best_model = load_model(filepath)

    # Evaluate the best model on the validation  data
    _, accuracy = best_model.evaluate([valX, valX, valX], valy, batch_size=batch_size, verbose=verbose)

    # save all models = 保存所有模型
    # model.save('models/{}_{}.h5'.format(DL_model, number))  # method 1 xx.h5
    return accuracy


# create, fit and evaluate DL model
def LSTM_model(trainX, valX, trainy, valy, number):
    epochs, patience, batch_size, verbose = 100, 5, 64, 0
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = Sequential()
    model.add(LSTM(128, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='Graphviz_{}.png'.format(DL_model))

    # Define the filepath for saving the best model
    filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, number)

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    # Create an EarlyStopping callback that defines the stopping condition = 创建一个 EarlyStopping 回调函数，定义停止训练的条件
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max')

    # Fit the model with the training data and labels, and use the validation data for evaluation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping],
              validation_data=(valX, valy), verbose=verbose)

    # Load the best model
    best_model = load_model(filepath)

    # Evaluate the best model on the validation  data
    _, accuracy = best_model.evaluate(valX, valy, batch_size=batch_size, verbose=verbose)

    # save all models = 保存所有模型
    # model.save('models/{}_{}.h5'.format(DL_model, number))  # method 1 xx.h5
    return accuracy


# create, fit and evaluate DL model
def CNN_LSTM_model(trainX, valX, trainy, valy, number):
    epochs, patience, batch_size, verbose = 100, 5, 64, 0
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps = 10
    n_length = int(trainX.shape[1] / n_steps)
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    valX = valX.reshape((valX.shape[0], n_steps, n_length, n_features))

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='Graphviz_{}.png'.format(DL_model))

    # Define the filepath for saving the best model
    filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, number)

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    # Create an EarlyStopping callback that defines the stopping condition = 创建一个 EarlyStopping 回调函数，定义停止训练的条件
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max')

    # Fit the model with the training data and labels, and use the validation data for evaluation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping],
              validation_data=(valX, valy), verbose=verbose)

    # Load the best model
    best_model = load_model(filepath)

    # Evaluate the best model on the validation  data
    _, accuracy = best_model.evaluate(valX, valy, batch_size=batch_size, verbose=verbose)

    # save all models = 保存所有模型
    # model.save('models/{}_{}.h5'.format(DL_model, number))  # method 1 xx.h5
    return accuracy


# create, fit and evaluate DL model
def ConvLSTM_model(trainX, valX, trainy, valy, number):
    epochs, patience, batch_size, verbose = 100, 5, 64, 0
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps = 10
    n_length = int(trainX.shape[1] / n_steps)
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    valX = valX.reshape((valX.shape[0], n_steps, 1, n_length, n_features))

    # define model
    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 3), activation='relu', input_shape=(n_steps, 1, n_length,
                                                                     n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='Graphviz_{}.png'.format(DL_model))

    # Define the filepath for saving the best model
    filepath = 'models/surrogate_{}_simples_{}_{}_{}.h5'.format(surr_type, simples, DL_model, number)

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    # Create an EarlyStopping callback that defines the stopping condition = 创建一个 EarlyStopping 回调函数，定义停止训练的条件
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max')

    # Fit the model with the training data and labels, and use the validation data for evaluation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping],
              validation_data=(valX, valy), verbose=verbose)

    # Load the best model
    best_model = load_model(filepath)

    # Evaluate the best model on the validation  data
    _, accuracy = best_model.evaluate(valX, valy, batch_size=batch_size, verbose=verbose)

    # save all models = 保存所有模型
    # model.save('models/{}_{}.h5'.format(DL_model, number))  # method 1 xx.h5
    return accuracy



if __name__ == '__main__':
    # define the location of the dataset
    full_path = 'G:/surrogate_data_based_machine_learning/anoxia/01_data_preprocessing/02_extracte_and_generate_surrogate_dataset/{}/data/merge_surrogate_{}_simples_{}_train_{}.csv'.format(Data_folder, surr_type, simples, ID_train)
    # full_path = '../../01_data_preprocessing/02_extracte_and_generate_surrogate_dataset/{}/data/merge_surrogate_{}_simples_{}_train_{}.csv'.format(Data_folder, surr_type, simples, ID_train)
    print('full_path=', full_path)

    # load the dataset
    X, y = load_dataset(full_path)
    # Split the training and test datasets = 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2023)

    # Transforms a NumPy array into a DataFrame = 将 NumPy 数组转换为 DataFrame
    df_X_test = pd.DataFrame(X_test)
    df_y_test = pd.DataFrame(y_test)
    # Save test data to compute ROC and AUC
    df_test = pd.concat([df_X_test, df_y_test], axis=1, ignore_index=True)
    # save dataframe file
    df_test.to_csv("01_evaluation_data/merge_test_surrogate_{}_simples_{}_train_{}.csv".format(surr_type, simples, ID_train), sep=',',
                   header=False, index=False)

    # np.expand_dims features
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # np.expand_dims outputs
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    # select DL model
    if DL_model == 'CNN_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = CNN_model(X_train, X_val, y_train, y_val, str(r))
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("DL_model:", DL_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif DL_model == 'three_head_CNN_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = three_head_CNN_model(X_train, X_val, y_train, y_val, str(r))
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("DL_model:", DL_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif DL_model == 'LSTM_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = LSTM_model(X_train, X_val, y_train, y_val, str(r))
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("DL_model:", DL_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif DL_model == 'CNN_LSTM_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = CNN_LSTM_model(X_train, X_val, y_train, y_val, str(r))
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("DL_model:", DL_model)
        print('scores:', scores)
        print('Accuracy mean (std): %.3f%% (%.3f)' % (mean(scores), std(scores)))
    elif DL_model == 'ConvLSTM_model':
        scores = list()
        # run the experiment
        for r in range(repeats):
            # evaluate and save model
            score = ConvLSTM_model(X_train, X_val, y_train, y_val, str(r))
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        print('------------------ summarize results -------------------------')
        print("DL_model:", DL_model)
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

    # results record
    ## try 1
    # [99.50000047683716, 99.62499737739563, 99.87499713897705, 100.0, 100.0, 99.87499713897705, 100.0, 100.0, 99.87499713897705, 100.0]
    # Accuracy mena (std): 99.875% (0.168)

    ## try 2
    # [95.24999856948853, 95.6250011920929, 97.75000214576721, 99.75000023841858, 100.0, 99.25000071525574, 100.0, 99.87499713897705, 100.0, 100.0]
    # Accuracy mena (std): 98.750% (1.783)

    # ------------------ summarize results -------------------------
    # [99.62499737739563, 99.75000023841858, 99.62499737739563, 99.00000095367432, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # Accuracy mena (std): 99.800% (0.307)
