"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# 5 parameters to set: tsid=int, Period=str, surr_type=str, simples=int, Variable_label=str

@author: Zhiqin Ma
"""

import time
start_time = time.time()

import os
import sys
import numpy as np
import pandas as pd
from pandas import read_csv

# # Setting parameters chick_heart
# tsid = 8  ## (Set it up to your needs)  # n=184
# simples = 1000  ## (Set it up to your needs)
# surr_type = 'AAFT'  ## (Set it up to your needs)
#
# # # passing parameters chick_heart
# # tsid = int(sys.argv[1])  ## (Set it up to your needs)
# # simples = int(sys.argv[2])  ## (Set it up to your needs)
# # surr_type = str(sys.argv[3])  ## (Set it up to your needs)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tsid', type=int,
                    help='Labels for the data',
                    default=8)
parser.add_argument('--simples', type=int,
                    help='Use the number of samples for generating a training dataset with surrogate data',
                    default=1000)
parser.add_argument('--surr_type', type=str,
                    help='Surrogate data. e.g., "RP", "FT", "AAFT", "IAAFT1", "IAAFT2"',
                    default='AAFT')
args = parser.parse_args()

tsid = args.tsid
simples = args.simples
surr_type = args.surr_type
print('tsid:', tsid)
print('simples:', simples)
print('surr_type:', surr_type)
print()

# Make export directory if doens't exist
try:
    os.mkdir('../data')
except:
    print('../data directory already exists!\n')


# Load in trajectory data
df_ews = pd.read_csv('../../01_organise__transition__ews/output_data/01_ews/df_ews_chick_heart.csv')
# print(df_ews)

# Load in transition times
df_transition = pd.read_csv('../../01_organise__transition__ews/raw_data/df_transitions.csv')
# print(df_transition)


# get 0 and 1 column names
column_Age = df_ews.columns[0]
column_state = df_ews.columns[1]
# print(column_Age, column_state)

# select 'Time', 'State variable', 'Smoothing' and 'Residuals' columns;
# and filter  'tsid' = 1 - 9 and 'Variable label' = 'Mo' columns
transition = df_transition[df_transition['tsid'] == tsid]['transition'].iloc[0]
# print(transition)
df = df_ews[[column_Age, column_state]][(df_ews['tsid'] == tsid) & (df_ews[column_Age] <= transition)].reset_index(drop=True)
# print(df)
# # review shape
print('df_tsid_{}.shape:'.format(tsid), df.shape)

# seting parameter
n = df.shape[0]  # Setting parameters
sliding_window = int(n / 2)  # 50% sliding window
df0 = df.iloc[0:sliding_window, :]  # 0 class
df1 = df.iloc[-sliding_window:, :]  # 1 class

# # review dataset
# print(df0)
# print(df1)

# Resetting the index
df0 = df0.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

# review dataset
# print(df0)

# save files
# df0.to_csv("V_MS21-Mo_S1_0.csv", header=True, index=False)
# df1.to_csv("V_MS21-Mo_S1_1.csv", header=True, index=False)
print("extract successfully.")

print("------------------------------------")

print("Load library of generate surrogate datasets...")

# # Load library of generate surrogate datasets = 加载生成替代数据集的库
# Select a surrogate data method
if surr_type == "WIAAFT":
    from julia.api import Julia  # 加载Julia库
    jl = Julia(compiled_modules=False)  # 禁用预编译缓存
    from julia import TimeseriesSurrogates, CairoMakie  # 加载julia函数
else:
    # 启动一个新的MATLAB进程，并返回Python的一个变量，它是一个MatlabEngine对象，用于与MATLAB过程进行通信。
    # Starts a new MATLAB process and returns a Python variable, which is a MatlabEngine object used to communicate with the MATLAB process.
    import matlab.engine  # import matlab引擎
    eng = matlab.engine.start_matlab()  # matlab's built-in functions can be called = 可以调用matlab的内置函数。

# dataframe to numpy
x_df0 = df0.loc[:, column_state].values
x_df1 = df1.loc[:, column_state].values
# array to list
x0 = x_df0.tolist()
x1 = x_df1.tolist()

# df0-------------------------------
# set surrogate parameter
# simples = 1000  ## (Set it up to your needs)
sliding_window = df0.shape[0]
t_start = df0.loc[0, column_Age]
t_end = df0.loc[sliding_window - 1, column_Age]
t_interval = (t_end - t_start) / sliding_window  # time is (kyr BP)
tf = 1.0 / round(t_interval, 8)
tf = float(tf)
# print('t_start, t_end=', t_start, t_end)
# print(tf)

# Initialize an array to store surrogate time series
surr_jl_0 = np.zeros((simples, sliding_window))
# Select a surrogate data method
if surr_type == "WIAAFT":
    # Rescale surrogate back to original values
    method = TimeseriesSurrogates.WLS(TimeseriesSurrogates.IAAFT(), rescale=True)
    for i in range(simples):
        surr_jl_0[i, :] = TimeseriesSurrogates.surrogate(x0, method)
    # numpy to dataframe
    surr_df0 = pd.DataFrame(surr_jl_0)
else:
    # Calling the .m function of Matlab
    surr0 = eng.surrogate(matlab.double(x0), simples, surr_type, 0, tf)  # 要
    # numpy to dataframe
    surr_df0 = pd.DataFrame(surr0)

print("surr_df0.shape: (simples, sliding_window) = ", surr_df0.shape)
# surr_df0.to_csv("results_surrogate_S1-MS21-Mo_0.csv", header=False, index=False)
# print("saved file successfully.")
print('surr0 successfully.')
# add y_label
surr_df0.loc[:, "y"] = 0

# df1-------------------------------
# set surrogate parameter
# simples = 1000  ## (Set it up to your needs)
sliding_window = df1.shape[0]
t_start = df1.loc[0, column_Age]
t_end = df1.loc[sliding_window - 1, column_Age]
t_interval = (t_end - t_start) / sliding_window
tf = 1.0 / round(t_interval, 8)
tf = float(tf)
# print('t_start, t_end=', t_start, t_end)
# print(tf)

# Initialize an array to store surrogate time series
surr_jl_1 = np.zeros((simples, sliding_window))
# Select a surrogate data method
if surr_type == "WIAAFT":
    # Rescale surrogate back to original values
    method = TimeseriesSurrogates.WLS(TimeseriesSurrogates.IAAFT(), rescale=True)
    for i in range(simples):
        surr_jl_1[i, :] = TimeseriesSurrogates.surrogate(x0, method)
    # numpy to dataframe
    surr_df1 = pd.DataFrame(surr_jl_1)
else:
    # Calling the .m function of Matlab
    surr1 = eng.surrogate(matlab.double(x1), simples, surr_type, 0, tf)  # 要
    # numpy to dataframe
    surr_df1 = pd.DataFrame(surr1)

print("surr_df1.shape: (simples, sliding_window): ", surr_df1.shape)
print('surr1 successfully.')
# add y_label
surr_df1.loc[:, "y"] = 1

# Merge
surr_df = pd.concat([surr_df0, surr_df1], axis=0, ignore_index=True)
# shuffle
surr_df = surr_df.sample(frac=1).reset_index(drop=True)
# review shape
print("surr_df.shape: ", surr_df.shape)
# save file
surr_df.to_csv("../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid, surr_type, simples, tsid),
               header=False,
               index=False)

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))

print('\n'"------------------- saved file successfully -------------------")
