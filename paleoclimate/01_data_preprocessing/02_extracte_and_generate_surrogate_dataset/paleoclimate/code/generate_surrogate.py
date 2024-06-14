"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# 5 parameters to set: tsid=int, Period=str, surr_type=str, simples=int, n_features=int

@author: Zhiqin Ma
"""
import os
import sys
import numpy as np
import pandas as pd
from pandas import read_csv

# # Setting parameters
# tsid = 1                      ## (Set it up to your needs)
# Period = "I"                  ## (Set it up to your needs)
# simples = 1000                ## (Set it up to your needs)
# surr_type = 'AAFT'            ## (Set it up to your needs)
# n_features = 300              ##  sliding window

# passing parameters
tsid = int(sys.argv[1])         ## (Set it up to your needs)
Period = str(sys.argv[2])       ## (Set it up to your needs)
simples = int(sys.argv[3])      ## (Set it up to your needs)
surr_type = str(sys.argv[4])    ## (Set it up to your needs)
n_features = int(sys.argv[5])   ## (Set it up to your needs)

print('tsid:', tsid)
print('Period:', Period)
print('simples:', simples)
print('surr_type:', surr_type)
print()

# Make export directory if doens't exist
try:
    os.mkdir('../data')
except:
    print('../data directory already exists!')

# path + name
file_path = r'../../../01_organise__transition__interpolate__ews/data/03_ews/paleoclimate_df_ews_interpolate.csv'
print('file_path=', file_path)

# load data
df = read_csv(file_path, header=0)
# review shape
print('df.shape:', df.shape)
# print(df)

# select 'Time', 'State variable', 'Smoothing' and 'Residuals' columns;
# and filter  'tsid' = 1 - 9 and 'Variable label' = 'Mo' columns
df = df[['Age', 'state']][(df['tsid'] == tsid) & (df['Period'] == Period)]
# review shape
print('df_{}.shape:'.format(Period), df.shape)

# seting parameter
n = df.shape[0]  # Setting parameters
if n > n_features * 2:
    sliding_window = n_features  # 50% sliding window
    df0 = df.iloc[0:sliding_window, :]  # 0 class
    df1 = df.iloc[-sliding_window:, :]  # 1 class
else:
    sliding_window = int(n / 2)  # 50% sliding window
    df0 = df.iloc[0:sliding_window, :]  # 0 class
    df1 = df.iloc[-sliding_window:, :]  # 1 class

# review dataset
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
x_df0 = df0.loc[:, "state"].values
x_df1 = df1.loc[:, "state"].values
# array to list
x0 = x_df0.tolist()
x1 = x_df1.tolist()

# df0-------------------------------
# set surrogate parameter
# simples = 1000  ## (Set it up to your needs)
sliding_window = df0.shape[0]
t_start = df0.loc[0, 'Age']
t_end = df0.loc[sliding_window - 1, 'Age']
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
t_start = df1.loc[0, 'Age']
t_end = df1.loc[sliding_window - 1, 'Age']
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
surr_df.to_csv("../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid, surr_type, simples, Period), header=False,
               index=False)

print('\n'"------------------- saved file successfully -------------------")
