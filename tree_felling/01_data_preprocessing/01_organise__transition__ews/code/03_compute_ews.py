# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:04:16 2023

Compute residauls and EWS in in Scheffer construction activity data

@author: Zhiqin Ma

Note：ewstools packages require python=3.8 above
"""

import math
import os
import pandas as pd
import ewstools
import warnings

warnings.filterwarnings('ignore')

# Make export directory if doens't exist
try:
    os.mkdir('../data')
except:
    print('data directory already exists!')

try:
    os.mkdir('../data/03_ews')
except:
    print('data/03_ews directory already exists!')
print("\n")

# Import transition data
df = pd.read_csv('../data/02_transitions/tree_felling_transitions.csv')
# print(df)

# Record names
list_records = df['Record'].unique()
# print(list_records)

# Period names
list_periods = df['Period'].unique()
# print(list_periods)

# Bandwidth sizes for Gaussian kernel (used in Marten Scheffer (2021) Fig. 3, Table S1 and Fig. S5)
dict_bandwidth = {
    'construction_activity_BMIII': 30,
    'construction_activity_PI': 30,
    'construction_activity_PII': 30,
    'construction_activity_EPIII': 15,
    'construction_activity_LPIII': 15,
}

# EWS computation parameters
# span = 100 # span for Lowess filtering
# rw = 0.5  # half the length of the data
lag_times = 1  # lag times for autocorrelation computation (lag of 10 to show decreasing AC where tau=T/2)

# counts the number of rows in the DataFrame that satisfy the condition
dict_rw = {
    'construction_activity_BMIII': 60.0,
    'construction_activity_PI': 60.0,
    'construction_activity_PII': 60.0,
    'construction_activity_EPIII': 20.0,
    'construction_activity_LPIII': 20.0,
}

# Loop through each record
list_df = []
i = 0
for record in list_records:
    # Get record specific data up to the transition point
    df_temp = df[df['Record'] == record]
    df_select = df_temp[df_temp['Age'] <= df_temp['Transition_start'].iloc[0]].copy()
    print('N=', len(df_select['Age']))

    # Series for computing EWS
    series = df_select.set_index('Age')['tree_felling']

    # Create TimeSeries object (new)
    ts = ewstools.TimeSeries(data=series, transition=None)
    # Compute detrend
    ts.detrend(method='Gaussian', bandwidth=dict_bandwidth[record])
    # # Compute EWS
    ts.compute_var(rolling_window=dict_rw[record]/len(df_select['Age']))
    ts.compute_auto(rolling_window=dict_rw[record]/len(df_select['Age']), lag=lag_times)

    # merge dataframe by columns
    df_ews = pd.concat([ts.state, ts.ews], axis=1)

    # Compute EWS (old)
    # ews_dic = ewstools.core.ews_compute(series,
    #                                     roll_window=rw[record],
    #                                     smooth='Lowess',
    #                                     upto='Full',
    #                                     band_width=dic_bandwidth[record],
    #                                     ews=ews,
    #                                     lag_times=lag_times
    #                                     )
    # df_ews = ews_dic['EWS metrics']

    # add new columns
    df_ews['standard_deviation'] = df_ews['variance'].apply(math.sqrt)  # apply the sqrt function to each value in the pandas
    df_ews['Record'] = record
    df_ews['Period'] = list_periods[i]
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

# # Export
df_ews.to_csv('../data/03_ews/tree_felling_df_ews.csv')

# printing tips
print('\n'"---------------------------- Completed ----------------------------")
