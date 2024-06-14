# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

- Prepare ROC  curve data in chick heart data

@author: Zhiqin Ma
"""

import os
import time
start_time = time.time()

import pandas as pd


# Make export directory if doens't exist
try:
    os.mkdir('../output_data')
except:
    print('../output_data directory already exists!')

try:
    os.mkdir('../output_data/02_prepare_roc_data')
except:
    print('../output_data/02_prepare_roc_data directory already exists!')


# Load in trajectory data
df = pd.read_csv('../output_data/01_ews/df_ews_chick_heart.csv', header=0)

# get 0, 1 and last column names
column_Age = df.columns[0]
column_state = df.columns[1]
column_tsid = df.columns[-1]
# print('column_Age=', column_Age)
# print('column_state=', column_state)
# print('column_tsid=', column_tsid)

# Load in transition times
df_transition = pd.read_csv('../raw_data/df_transitions.csv', header=0)
# print(df_transition[df_transition['tsid'] == 1]['transition'].iloc[0])
# df_transition.set_index('tsid', inplace=True)
# print(df_transition.loc[2, 'transition'])

# Select tsid
list_tsid = df['tsid'].unique()     # Select all tsid
# list_tsid = [1, 2]                # or Select specific tsid
# print(list_tsid)

list_df_pd = []
list_df_null = []
for tsid in list_tsid:
    df_before_transition = df[[column_Age, column_state, column_tsid]][(df['tsid'] == tsid) & (df[column_Age] <= df_transition[df_transition['tsid'] == tsid]['transition'].iloc[0])].reset_index(drop=True)
    # print(df_before_transition)
    # # review shape
    print('df_tsid_{}.shape:'.format(tsid), df_before_transition.shape)

    n = df_before_transition.shape[0]                                       # Setting parameters
    sliding_window = int(n / 2)                                             # 50% sliding window
    df_null = df_before_transition.iloc[0:sliding_window, :].copy()         # 0 class
    df_pd = df_before_transition.iloc[-sliding_window:, :].copy()           # 1 class
    df_pd[column_Age] = df_pd[column_Age] - df_pd[column_Age].iloc[0]       # 'Beat number' starts at 0
    df_null.loc[:, 'type'] = 'neutral'
    df_pd.loc[:, 'type'] = 'pd'
    list_df_pd.append(df_pd)
    list_df_null.append(df_null)

# save DataFrame
prepare_roc_data = pd.concat(list_df_pd + list_df_null, ignore_index=True)
# Export transition data
prepare_roc_data.to_csv('../output_data/02_prepare_roc_data/prepare_roc_data.csv', index=False)

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('\nRan in {:.2f}s'.format(time_taken))

print('\n'"---------------------------- 02 Completed ----------------------------")