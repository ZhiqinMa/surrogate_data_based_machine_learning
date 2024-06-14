# -*- coding: utf-8 -*-
"""
Created on Sep  26 10:52:43 2020
Created on Wed Mar 20 23:04:16 2021

Compute residauls and EWS for anoxia data

@author: Thomas M. Bury
@author: Zhiqin Ma

Note：ewstools packages require python=3.8 above
"""

import os
import pandas as pd
import ewstools
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')


# Make export directory if doens't exist
try:
    os.mkdir('../data/02_ews')
except:
    print('data/02_ews directory already exists!')

# Import transition data
df = pd.read_csv('../data/01_transitions/anoxia_transitions.csv')

# EWS computation parameters
# span = 100 # span for Lowess filtering
rw = 0.5  # half the length of the data
lag_times = 1  # lag times for autocorrelation computation (lag of 10 to show decreasing AC where tau=T/2)
span = 0.2
bandwidth = 0.09  # BW used in paper = 900yr = 0.09% of pre-transiiton data (10,000kyr)
smooth = 'Gaussian'

# -------------
# Compute EWS for transition data
# --------------

# Record names

# Loop through each record
list_df = []
list_tsid = df['tsid'].unique()
for tsid in list_tsid:
    # Get record specific data up to the transition point
    df_temp = df[(df['tsid'] == tsid)]
    df_select = df_temp[df_temp['Age [ka BP]'] >= df_temp['t_transition_start'].iloc[0]].copy()
    # review N
    print('N=', len(df_select['Age [ka BP]']))

    # Make time negative so it increaes up to transition
    df_select['Age [ka BP]'] = -df_select['Age [ka BP]']
    # Reverse order of dataframe so transition occurs at the end of the series
    df_select = df_select[::-1]

    # ------------
    # Compute EWS for Mo
    # ------------
    series = df_select.set_index('Age [ka BP]')['Mo [ppm]']
    # Create TimeSeries object (new)
    ts = ewstools.TimeSeries(data=series, transition=None)
    # Compute detrend
    ts.detrend(method='Gaussian', span=span)
    # Compute EWS
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=lag_times)
    # merge dataframe by columns
    df_ews = pd.concat([ts.state, ts.ews], axis=1)

    # # Compute EWS (old)
    # ews_dic = ewstools.core.ews_compute(series,
    #                                     roll_window=rw,
    #                                     smooth=smooth,
    #                                     upto='Full',
    #                                     span=span,
    #                                     ews=ews,
    #                                     lag_times=lag_times,
    #                                     )
    # df_ews = ews_dic['EWS metrics']

    # add new columns
    df_ews['tsid'] = tsid
    df_ews['Variable_label'] = 'Mo'

    # Export residuals for ML
    # df_ews[['Residuals']].reset_index().round(6).to_csv('data/resids/resids_anoxia_forced_mo_{}.csv'.format(tsid), index=False)

    # Add to list
    list_df.append(df_ews)

    # ------------
    # Compute EWS for U
    # ------------
    series = df_select.set_index('Age [ka BP]')['U [ppm]']
    # Create TimeSeries object (new)
    ts = ewstools.TimeSeries(data=series, transition=None)
    # Compute detrend
    ts.detrend(method='Gaussian', span=span)
    # Compute EWS
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=lag_times)
    # merge dataframe by columns
    df_ews = pd.concat([ts.state, ts.ews], axis=1)

    # # Compute EWS (old)
    # ews_dic = ewstools.core.ews_compute(series,
    #                                     roll_window=rw,
    #                                     smooth=smooth,
    #                                     upto='Full',
    #                                     span=span,
    #                                     ews=ews,
    #                                     lag_times=lag_times,
    #                                     )
    # df_ews = ews_dic['EWS metrics']

    # add new columns
    df_ews['tsid'] = tsid
    df_ews['Variable_label'] = 'U'

    # # Export residuals for ML
    # df_ews[['Residuals']].reset_index().to_csv('data/resids_gaussian/resids_anoxia_gaussian_forced_u_{}.csv'.format(tsid),index=False)

    # Add to list
    list_df.append(df_ews)

    print('EWS computed for tsid {}'.format(tsid))
    print('\n')

# Concatenate dataframes
df_ews = pd.concat(list_df)

# Export ews dataframe
df_ews.to_csv('../data/02_ews/anoxia_df_ews.csv')


print('\n'"---------------------------- Completed 02 compute ews ----------------------------")



