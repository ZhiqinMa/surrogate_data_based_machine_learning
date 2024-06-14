# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:04:16 2023

Organise construction activity data in same format as in Scheffer (2021)
Use time ranges and transition times as given in Figs. 2 and 3 of Scheffer (2021)

@author: Zhiqin Ma
"""

import os
import pandas as pd

# Make export directory if doens't exist
try:
    os.mkdir('../data/02_transitions')
except:
    print('data/02_transitions directory already exists!')
print("\n")

# ----------------
# Import and organise data
# ----------------

list_df = []

# path + name
path_name = '../data/01_time_series/tree_felling_over_time.csv'

# Number of trees felled as a proxy for construction activity in BM_III
df = pd.read_csv(path_name, header=0)
df = df[['Age', 'tree_felling']]
df = df[(df['Age'] >= 500) & (df['Age'] <= 705)].reset_index(drop=True)
df['Record'] = 'construction_activity_BMIII'
df['Transition_start'] = 684
df['Transition_end'] = 705
df['Period'] = 'BMIII'
df['tsid'] = 0
list_df.append(df)

# Number of trees felled as a proxy for construction activity in PI
df = pd.read_csv(path_name, header=0)
df = df[['Age', 'tree_felling']]
df = df[(df['Age'] >= 705) & (df['Age'] <= 873)].reset_index(drop=True)
df['Record'] = 'construction_activity_PI'
df['Transition_start'] = 871
df['Transition_end'] = 873
df['Period'] = 'PI'
df['tsid'] = 1
list_df.append(df)

# Number of trees felled as a proxy for construction activity in PII
df = pd.read_csv(path_name, header=0)
df = df[['Age', 'tree_felling']]
df = df[(df['Age'] >= 873) & (df['Age'] <= 1157)].reset_index(drop=True)
df['Record'] = 'construction_activity_PII'
df['Transition_start'] = 1118
df['Transition_end'] = 1157
df['Period'] = 'PII'
df['tsid'] = 2
list_df.append(df)

# Number of trees felled as a proxy for construction activity in EPIII
df = pd.read_csv(path_name, header=0)
df = df[['Age', 'tree_felling']]
df = df[(df['Age'] >= 1157) & (df['Age'] <= 1217)].reset_index(drop=True)
df['Record'] = 'construction_activity_EPIII'
df['Transition_start'] = 1215
df['Transition_end'] = 1217
df['Period'] = 'EPIII'
df['tsid'] = 3
list_df.append(df)

# Number of trees felled as a proxy for construction activity in LPIII
df = pd.read_csv(path_name, header=0)
df = df[['Age', 'tree_felling']]
df = df[(df['Age'] >= 1217) & (df['Age'] <= 1300)].reset_index(drop=True)
df['Record'] = 'construction_activity_LPIII'
df['Transition_start'] = 1283
df['Transition_end'] = 1300
df['Period'] = 'LPIII'
df['tsid'] = 4
list_df.append(df)

# ------------
# Concatenate dataframes
# --------------

df_full = pd.concat(list_df)
df_full.to_csv('../data/02_transitions/tree_felling_transitions.csv', index=False)

print('\n'"-------------------------------- Completed 02 transitions data --------------------------------")
