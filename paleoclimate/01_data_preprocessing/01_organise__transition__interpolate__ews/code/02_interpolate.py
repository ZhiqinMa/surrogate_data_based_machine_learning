# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

interpolate in transition data

@author: Zhiqin Ma
"""

import os
from pandas import read_csv
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Make export directory if doens't exist
try:
    os.mkdir('../data/02_transitions_interpolate')
except:
    print('data/02_transitions_interpolate directory already exists!')

# path name
path_name = '../data/01_transitions/paleoclimate_transitions.csv'
# load dataset
df = read_csv(path_name, header=0)
# print(df)

# ----------------
# interpolate transitions data
# ----------------

# ----- End_of_glaciation_I -----

# Setting parameters
Period = 'I'    ## (Set it up to your needs)
tsid = 1        ## (Set it up to your needs)

# select columns
data = df[['Age', 'Proxy']][(df['Period'] == Period) & (df['tsid'] == tsid)].values
print('data.shape:', data.shape)

# select x, y columns
x = data[:, 0]
y = data[:, 1]

# line interpolate
func = interp1d(x, y)
# numbers point
numbers = int((np.max(x) - np.min(x)) / 100)
x_new = np.linspace(np.max(x), np.min(x), numbers)
y_new = func(x_new)
# review data shape
print('np.min(x)=', np.min(x), 'np.max(x)=', np.max(x), 'numbers=', numbers)

# create dataframe
columns = ['Age', 'Proxy', 'Climate_proxy', 'Record', 'Transition_start', 'Transition_end', 'Period', 'tsid']
new_df1 = pd.DataFrame(columns=columns)

# add columns
new_df1['Age'] = x_new
new_df1['Proxy'] = y_new
new_df1['Climate_proxy'] = df['Climate_proxy'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df1['Record'] = df['Record'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df1['Transition_start'] = df['Transition_start'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df1['Transition_end'] = df['Transition_end'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df1['Period'] = Period
new_df1['tsid'] = tsid
print('--------------------------------''\n')


# ----- End_of_glaciation_II -----

# Setting parameters
Period = 'II'   ## (Set it up to your needs)
tsid = 2        ## (Set it up to your needs)

# select columns
data = df[['Age', 'Proxy']][(df['Period'] == Period) & (df['tsid'] == tsid)].values
print('data.shape:', data.shape)

# select x, y columns
x = data[:, 0]
y = data[:, 1]

# line interpolate
func = interp1d(x, y)
# numbers point
numbers = int((np.max(x) - np.min(x)) / 100)
x_new = np.linspace(np.max(x), np.min(x), numbers)
y_new = func(x_new)
# review data shape
print('np.min(x)=', np.min(x), 'np.max(x)=', np.max(x), 'numbers=', numbers)

# create dataframe
columns = ['Age', 'Proxy', 'Climate_proxy', 'Record', 'Transition_start', 'Transition_end', 'Period', 'tsid']
new_df2 = pd.DataFrame(columns=columns)

# add columns
new_df2['Age'] = x_new
new_df2['Proxy'] = y_new
new_df2['Climate_proxy'] = df['Climate_proxy'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df2['Record'] = df['Record'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df2['Transition_start'] = df['Transition_start'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df2['Transition_end'] = df['Transition_end'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df2['Period'] = Period
new_df2['tsid'] = tsid
print('--------------------------------''\n')


# ----- End_of_glaciation_III -----

# Setting parameters
Period = 'III'  ## (Set it up to your needs)
tsid = 3        ## (Set it up to your needs)

# select columns
data = df[['Age', 'Proxy']][(df['Period'] == Period) & (df['tsid'] == tsid)].values
print('data.shape:', data.shape)

# select x, y columns
x = data[:, 0]
y = data[:, 1]

# line interpolate
func = interp1d(x, y)
# numbers point
numbers = int((np.max(x) - np.min(x)) / 100)
x_new = np.linspace(np.max(x), np.min(x), numbers)
y_new = func(x_new)
# review data shape
print('np.min(x)=', np.min(x), 'np.max(x)=', np.max(x), 'numbers=', numbers)

# create dataframe
columns = ['Age', 'Proxy', 'Climate_proxy', 'Record', 'Transition_start', 'Transition_end', 'Period', 'tsid']
new_df3 = pd.DataFrame(columns=columns)

# add columns
new_df3['Age'] = x_new
new_df3['Proxy'] = y_new
new_df3['Climate_proxy'] = df['Climate_proxy'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df3['Record'] = df['Record'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df3['Transition_start'] = df['Transition_start'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df3['Transition_end'] = df['Transition_end'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df3['Period'] = Period
new_df3['tsid'] = tsid
print('--------------------------------''\n')


# ----- End_of_glaciation_IV -----

# Setting parameters
Period = 'IV'   ## (Set it up to your needs)
tsid = 4        ## (Set it up to your needs)

# select columns
data = df[['Age', 'Proxy']][(df['Period'] == Period) & (df['tsid'] == tsid)].values
print('data.shape:', data.shape)

# select x, y columns
x = data[:, 0]
y = data[:, 1]

# line interpolate
func = interp1d(x, y)
# numbers point
numbers = int((np.max(x) - np.min(x)) / 100)
x_new = np.linspace(np.max(x), np.min(x), numbers)
y_new = func(x_new)
# review data shape
print('np.min(x)=', np.min(x), 'np.max(x)=', np.max(x), 'numbers=', numbers)

# create dataframe
columns = ['Age', 'Proxy', 'Climate_proxy', 'Record', 'Transition_start', 'Transition_end', 'Period', 'tsid']
new_df4 = pd.DataFrame(columns=columns)

# add columns
new_df4['Age'] = x_new
new_df4['Proxy'] = y_new
new_df4['Climate_proxy'] = df['Climate_proxy'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df4['Record'] = df['Record'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df4['Transition_start'] = df['Transition_start'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df4['Transition_end'] = df['Transition_end'][(df['Period'] == Period) & (df['tsid'] == tsid)].iloc[0]
new_df4['Period'] = Period
new_df4['tsid'] = tsid
print('--------------------------------''\n')


# ------------
# Concatenate dataframes
# --------------
dataframe_lists = [new_df1, new_df2, new_df3, new_df4]
# concat DataFrame
df_merged = pd.concat(dataframe_lists, axis=0, ignore_index=True)
# save dataframe
df_merged.to_csv('../data/02_transitions_interpolate/paleoclimate_transitions_interpolate.csv', index=False, header=True, sep=",")
