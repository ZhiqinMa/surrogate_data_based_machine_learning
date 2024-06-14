# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

-Compute EWS and DL predictions rolling over chick heart data

@author: tbury
@author: Zhiqin Ma

Note：ewstools packages require python=3.8 above
"""

import time
start_time = time.time()

import os
import pandas as pd
import ewstools
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
# import warnings
# warnings.filterwarnings('ignore')


# Make export directory if doens't exist
try:
    os.mkdir('../output_data')
except:
    print('../output_data directory already exists!')

try:
    os.mkdir('../output_data/01_ews')
except:
    print('../output_data/01_ews directory already exists!')

np.random.seed(0)
project_name = 'chick_heart'

# Load in trajectory data
df_traj = pd.read_csv('../raw_data/df_chick.csv')
df_traj_pd = df_traj[df_traj['type'] == 'pd']
# print(df_traj_pd)

# Load in transition times
df_transition = pd.read_csv('../raw_data/df_transitions.csv')
# print(df_transition)

# -------------
# Compute EWS for period-doubling trajectories
# --------------
# EWS parameters
rw = 0.5        # rolling window
bw = 20         # Gaussian band width (# beats)
lag_times = 1   # lag-1 ac

list_ews = []

list_tsid = df_traj_pd['tsid'].unique()
# Loop through each record
for tsid in list_tsid:
    df_spec = df_traj_pd[df_traj_pd['tsid']==tsid].set_index('Beat number')
    transition = df_transition[df_transition['tsid']==tsid]['transition'].iloc[0]
    series = df_spec['IBI (s)'].iloc[:]

    # Create TimeSeries object (new)
    ts = ewstools.TimeSeries(data=series, transition=transition)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=bw)

    # Compute EWS
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=lag_times)

    # merge dataframe by columns
    df_ews = pd.concat([ts.state, ts.ews], axis=1)

    # add new columns
    df_ews['tsid'] = tsid

    # Add to list
    list_ews.append(df_ews)

# Concatenate dataframes
df_ews_pd = pd.concat(list_ews)

# Export ews dataframe
df_ews_pd.to_csv('../output_data/01_ews/df_ews_{}.csv'.format(project_name))

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('\nRan in {:.2f}s'.format(time_taken))

print('\n'"---------------------------- 01 Completed compute ews ----------------------------")



