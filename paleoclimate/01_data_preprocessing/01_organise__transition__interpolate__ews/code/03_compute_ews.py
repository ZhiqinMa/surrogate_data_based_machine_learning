# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:52:43 2020
Created on Wed Mar 20 23:04:16 2023

Compute residauls and EWS in Dakos climate data

@author: Thomas M. Bury
@author: Zhiqin Ma

Note：ewstools packages require python=3.8 above
"""

import os
import numpy as np
import pandas as pd
import ewstools

# Make export directory if doens't exist
try:
    os.mkdir('../data/03_ews')
except:
    print('data/03_ews directory already exists!')

# Import transition data
df = pd.read_csv('../data/02_transitions_interpolate/paleoclimate_transitions_interpolate.csv')

# Record names
list_records = df['Record'].unique()
# print(list_records)

# Period names
list_periods = df['Period'].unique()
# print(list_periods)

# Bandwidth sizes for Gaussian kernel (used in Dakos (2008) Table S3)
dic_bandwidth = {
    'End_of_glaciation_I': 25,
    'End_of_glaciation_II': 25,
    'End_of_glaciation_III': 10,
    'End_of_glaciation_IV': 50
}


# span = 100 # span for Lowess filtering
rw = 0.5  # half the length of the data
lag_times = 1  # lag times for autocorrelation computation (lag of 10 to show decreasing AC where tau=T/2)

# Loop through each record
list_df = []
i = 1
for record in list_records:
    # Get record specific data
    df_temp = df[df['Record'] == record]
    df_select = df_temp[df_temp['Age'] >= df_temp['Transition_start'].iloc[0]].copy()
    print('N=', len(df_select['Age']))

    # Make time negative so it increaes up to transition
    df_select['Age'] = -df_select['Age']
    # Reverse order of dataframe so transition occurs at the end of the series
    # df_select = df_select[::-1]

    # ------------
    # Compute EWS
    # ------------
    # Series for computing EWS
    series = df_select.set_index('Age')['Proxy']
    # Create TimeSeries object (new)
    ts = ewstools.TimeSeries(data=series, transition=None)
    # Compute detrend
    ts.detrend(method='Gaussian', bandwidth=dic_bandwidth[record])
    # Compute EWS
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=lag_times)
    # merge dataframe by columns
    df_ews = pd.concat([ts.state, ts.ews], axis=1)

    # # Compute EWS (old)
    # ews_dic = ewstools.core.ews_compute(series,
    #                                     roll_window=rw,
    #                                     smooth='Gaussian',
    #                                     upto='Full',
    #                                     band_width=dic_bandwidth[record],
    #                                     ews=ews,
    #                                     lag_times=lag_times
    #                                     )
    # df_ews = ews_dic['EWS metrics']

    # add new columns
    df_ews['Record'] = record
    df_ews['Period'] = list_periods[i - 1]
    df_ews['tsid'] = i
    list_df.append(df_ews)

    # screen Print
    print('EWS computed for tsid {}'.format(i))
    print('\n')

    # Export residuals for ML
    # df_ews[['Residuals']].reset_index().to_csv('data/resids/resids_ar1_dakos_{}_forced.csv'.format(i), index=False)
    i += 1

# Concatenate dataframes
df_ews = pd.concat(list_df)
# print(df_ews.shape)

# # Export
df_ews.to_csv('../data/03_ews/paleoclimate_df_ews_interpolate.csv')

# printing tips
print('\n'"---------------------------- Completed 02 compute ews ----------------------------")
