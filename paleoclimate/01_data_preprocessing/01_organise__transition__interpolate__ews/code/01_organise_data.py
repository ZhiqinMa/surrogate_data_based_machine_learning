# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:46:55 2020
Created on Wed Mar 20 23:04:16 2023

Organise climate data in same format as in Dakos 2008
Use time ranges and transition times as given in Table S1 Dakos 2008

@author: Thomas M. Bury
@author: Zhiqin Ma
"""

import os
import pandas as pd

# Make export directory if doens't exist
try:
    os.mkdir('../data/01_transitions')
except:
    print('data/01_transitions directory already exists!')

# ----------------
# Import and organise data
# ----------------

list_df = []

# path + name
path_name = '../data/raw_deutnat/deutnat.txt'

# End of glaciation 1 (EG1)
df = pd.read_csv(path_name,
                 header=0,
                 names=['Depth', 'Age', 'Proxy', 'deltaTS'],
                 delim_whitespace=True,
                 )
df = df[['Age', 'Proxy']]
# df = df[(df['Age'] <= 58800) & (df['Age'] >= 12000)].sort_values('Age', ascending=False).reset_index(drop=True)
df = df[(df['Age'] <= 108025) & (df['Age'] >= 12000)].sort_values('Age', ascending=False).reset_index(drop=True)

df['Climate_proxy'] = 'd2H (%)'
df['Record'] = 'End_of_glaciation_I'
df['Transition_start'] = 17000
df['Transition_end'] = 13.828e3
df['Period'] = 'I'
df['tsid'] = 1
list_df.append(df)

# End of glaciation II
df = pd.read_csv(path_name,
                 header=0,
                 names=['Depth', 'Age', 'Proxy', 'deltaTS'],
                 delim_whitespace=True,
                 )
df = df[['Age', 'Proxy']]
# df = df[(df['Age'] <= 151000) & (df['Age'] >= 128000)].sort_values('Age', ascending=False).reset_index(drop=True)
df = df[(df['Age'] <= 224351) & (df['Age'] >= 128000)].sort_values('Age', ascending=False).reset_index(drop=True)

df['Climate_proxy'] = 'd2H (%)'
df['Record'] = 'End_of_glaciation_II'
df['Transition_start'] = 135000
df['Transition_end'] = 129.705e3
df['Period'] = 'II'
df['tsid'] = 2
list_df.append(df)

# End of glaciation III
df = pd.read_csv(path_name,
                 header=0,
                 names=['Depth', 'Age', 'Proxy', 'deltaTS'],
                 delim_whitespace=True,
                 )
df = df[['Age', 'Proxy']]
# df = df[(df['Age'] <= 270000) & (df['Age'] >= 238000)].sort_values('Age', ascending=False).reset_index(drop=True)
df = df[(df['Age'] <= 306386) & (df['Age'] >= 238000)].sort_values('Age', ascending=False).reset_index(drop=True)

df['Climate_proxy'] = 'd2H (%)'
df['Record'] = 'End_of_glaciation_III'
df['Transition_start'] = 242000
df['Transition_end'] = 238.084e3
df['Period'] = 'III'
df['tsid'] = 3
list_df.append(df)

# End of glaciation IV
df = pd.read_csv(path_name,
                 header=0,
                 names=['Depth', 'Age', 'Proxy', 'deltaTS'],
                 delim_whitespace=True,
                 )
df = df[['Age', 'Proxy']]
# df = df[(df['Age'] <= 385300) & (df['Age'] >= 324600)].sort_values('Age', ascending=False).reset_index(drop=True)
df = df[(df['Age'] <= 386528) & (df['Age'] >= 324600)].sort_values('Age', ascending=False).reset_index(drop=True)

df['Climate_proxy'] = 'd2H (%)'
df['Record'] = 'End_of_glaciation_IV'
df['Transition_start'] = 334100
df['Transition_end'] = 325.039e3
df['Period'] = 'IV'
df['tsid'] = 4
list_df.append(df)

# ------------
# Concatenate dataframes
# --------------

df_full = pd.concat(list_df)
df_full.to_csv('../data/01_transitions/paleoclimate_transitions.csv', index=False)

# printing tips
print('\n'"---------------------------- Completed 01 organise data ----------------------------")
