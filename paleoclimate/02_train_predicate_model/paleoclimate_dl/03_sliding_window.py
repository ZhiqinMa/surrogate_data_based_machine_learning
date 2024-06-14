# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

Spilt sliding window

8 parameters to set:    surr_type=str, simples=int, ID_train=str, ID_test=str,
                        Variable_label=str, tsid=int, features=int, sw_width=int

@author: Zhiqin Ma
"""

import os
import sys
import numpy as np
import pandas as pd
from pandas import concat
from pandas import read_csv
from matplotlib import pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

print('------------------------------ 03 Start sliding windows ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)
# ID_train = 'IV_III'               ## (Set it up to your needs)
# ID_test = 'II'                    ## (Set it up to your needs)
# Variable_label = 'paleoclimate'   ## (Set it up to your needs)
# tsid = 2                          ## (Set it up to your needs)
# features = 300                    ## (Set it up to your needs)
# sw_width = 446                    ## (Set it up to your needs)

# Setting parameters
surr_type = str(sys.argv[1])        ## (Set it up to your needs)
simples = int(sys.argv[2])          ## (Set it up to your needs)
ID_train = str(sys.argv[3])         ## (Set it up to your needs)
ID_test = str(sys.argv[4])          ## (Set it up to your needs)
Variable_label = str(sys.argv[5])   ## (Set it up to your needs)
tsid = int(sys.argv[6])             ## (Set it up to your needs)
features = int(sys.argv[7])         ## (Set it up to your needs)
sw_width = int(sys.argv[8])         ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('ID_test:', ID_test)
print('Variable_label:', Variable_label)
print('tsid:', tsid)
print('features:', features)
print('sw_width:', sw_width)
print()


try:
    os.mkdir('03_sliding_window_data')
except:
    print('03_sliding_window_data directory already exists!')



def split_sequence(sequence, sw_width, n_features):
    X = []
    for i in range(len(sequence)):
        # 找到最后一次滑动所截取数据中最后一个元素的索引，
        # 如果这个索引超过原序列中元素的索引则不截取；
        # Note: 如果特征数量不够，需要在每一行左边添加0，直到特征数量和分类器特征一样。
        end_element_index = i + sw_width
        if end_element_index > len(sequence):
            break

        # if 'end_element_index < n_features', neet to add o to lest side of list
        if end_element_index > n_features:
            sequence_x = sequence[end_element_index - n_features:end_element_index]
        else:
            sequence_x = sequence[0:end_element_index]

        # Add 0 to the left side of list
        add_0_num = n_features - len(sequence_x)
        # method 1
        sequence_x = ([0] * add_0_num) + sequence_x
        # # or method 2
        # for j in range(add_0_num):
        #     sequence_x.insert(0, 0)

        # 2D list = matrix
        X.append(sequence_x)

    return np.array(X)


if __name__ == '__main__':
    # define the dataset location
    filename = '../../01_data_preprocessing/01_organise__transition__interpolate__ews/data/03_ews/paleoclimate_df_ews_interpolate.csv'

    # load the csv file as a data frame
    dataframe = read_csv(filename, header=0)

    # get 0 and 1 column names
    df_ews_column_Age = dataframe.columns[0]
    df_ews_column_state = dataframe.columns[1]

    # select columns
    dataframe = dataframe[[df_ews_column_Age, df_ews_column_state]][(dataframe['Period'] == ID_test) & (dataframe['tsid'] == tsid)]
    # review shape
    print('{}.shape:'.format(ID_test), dataframe.shape)

    # # plot
    # plt.plot(dataframe.loc[:, 'Age'], dataframe.loc[:, 'tree_felling'])
    # plt.show()

    # # reversion numpy array
    # dataframe = dataframe.values
    # dataframe = dataframe[::-1]
    # # review data
    # print(dataframe.shape)

    # # numpy to dataframe
    # dataframe_df = pd.DataFrame(dataframe)
    # # save reversion data
    # dataframe_df.to_csv("data/transition_data_{}_{}.csv".format(surr_type, ID_test), sep=',', header=False, index=False)

    # seq_test_X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # example
    seq_test_X = dataframe['state']
    # print(seq_test_X)
    # convert pandas and numpy to a list
    if not type(seq_test_X) is list:
        seq_test_X = seq_test_X.tolist()
    # split sliding window
    test_X = split_sequence(seq_test_X, sw_width, features)
    # from numpy.ndarray to dataframe
    test_X_features = pd.DataFrame(test_X)
    # review
    # print(test_X_features)

    # save sliding window data
    test_X_features.to_csv("03_sliding_window_data/sliding_window_features_surrogate_{}_simples_{}_test_{}.csv".format(surr_type, simples, ID_test),
        sep=',', header=False, index=False)

    print("success-> sliding_window_features_shape (real=({}-{}+1, {})): ".format(dataframe.shape[0], sw_width, features), test_X_features.shape)
    print('------------------------------ 03 Completed sliding windows ------------------------------')
    print()
