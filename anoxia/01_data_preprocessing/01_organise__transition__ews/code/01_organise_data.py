# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:40:22 2020

Organise anoxia data into transition sections (Hennekam et al. 2020)

@author: Thomas M. Bury
"""

import os
import pandas as pd

# Make export directory if doens't exist
try:
    os.mkdir('../data/01_transitions')
except:
    print('data/01_transitions directory already exists!')



# Raw data (excel file)
xlsx_path = '../data/raw_anoxia/anoxia_data.xlsx'

# Columns of dataset to keep
cols = ['Age [ka BP]', 'Mo [ppm]', 'U [ppm]']

# Import each sheet individually (different cores)
df_ms21 = pd.read_excel(xlsx_path, sheet_name='XRF-CS MS21', usecols=cols, engine='openpyxl')
df_ms66 = pd.read_excel(xlsx_path, sheet_name='XRF-CS MS66', usecols=cols, engine='openpyxl')
df_64pe = pd.read_excel(xlsx_path, sheet_name='XRF-CS 64PE406E1', usecols=cols, engine='openpyxl')

# Get transition data from Hennekam et al. 2020 Figure 3
# Tuple of form (ID, t_min, t_transition_end, t_transition_start, t_max)
list_tup = [
    ('S1', 8, 9.7, 10.5, 20.5),
    ('S3', 83.7, 85.1, 85.8, 95.8),
    ('S4', 106.0, 107.2, 107.8, 117.8),
    ('S5', 125.0, 127.5, 128.35, 138.35),
    ('S6', 175.0, 176.5, 177.25, 187.25),
    ('S7', 195.0, 197.8, 198.5, 208.5),
    ('S8', 222.0, 223.6, 224.9, 234.9),
    ('S9', 238.0, 239.6, 240.3, 250.3),
]

df_transition_loc = pd.DataFrame(
    {'ID': [tup[0] for tup in list_tup],
     't_min': [tup[1] for tup in list_tup],
     't_transition_end': [tup[2] for tup in list_tup],
     't_transition_start': [tup[3] for tup in list_tup],
     't_max': [tup[4] for tup in list_tup]
     })

# --------------------
# For each transition, extract data
# ---------------------

id_vals = df_transition_loc['ID'].values

# Transition ID values analysed for each core
id_vals_ms21 = ['S1', 'S3']
id_vals_ms66 = ['S1', 'S3', 'S4', 'S5']
id_vals_64pe = ['S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

list_df = []

# Counter to create a tsid for each transition time series
# (13 in total for each variable, U and Mo)

count_tsid = 1

for id_val in id_vals:

    # Get data for transition ID
    df_temp = df_transition_loc[df_transition_loc['ID'] == id_val]
    t_min = df_temp['t_min'].iloc[0]
    t_max = df_temp['t_max'].iloc[0]
    t_transition_start = df_temp['t_transition_start'].iloc[0]
    t_transition_end = df_temp['t_transition_end'].iloc[0]

    # Extract time series data within bounds for each core if used

    # MS21
    if id_val in id_vals_ms21:
        df_extract = df_ms21[(df_ms21['Age [ka BP]'] >= t_min) & \
                             (df_ms21['Age [ka BP]'] <= t_max)].copy()
        df_extract['ID'] = id_val
        df_extract['Core'] = 'MS21'
        df_extract['t_transition_start'] = t_transition_start
        df_extract['t_transition_end'] = t_transition_end
        df_extract['tsid'] = count_tsid
        count_tsid += 1
        # Append data to list
        list_df.append(df_extract)

    # MS66
    if id_val in id_vals_ms66:
        df_extract = df_ms66[(df_ms66['Age [ka BP]'] >= t_min) & \
                             (df_ms66['Age [ka BP]'] <= t_max)].copy()
        df_extract['ID'] = id_val
        df_extract['Core'] = 'MS66'
        df_extract['t_transition_start'] = t_transition_start
        df_extract['t_transition_end'] = t_transition_end
        df_extract['tsid'] = count_tsid
        count_tsid += 1
        # Append data to list
        list_df.append(df_extract)

    # 64PE
    if id_val in id_vals_64pe:
        df_extract = df_64pe[(df_64pe['Age [ka BP]'] >= t_min) & \
                             (df_64pe['Age [ka BP]'] <= t_max)].copy()
        df_extract['ID'] = id_val
        df_extract['Core'] = '64PE'
        df_extract['t_transition_start'] = t_transition_start
        df_extract['t_transition_end'] = t_transition_end
        df_extract['tsid'] = count_tsid
        count_tsid += 1
        # Append data to list
        list_df.append(df_extract)

df_transition_data = pd.concat(list_df, ignore_index=True)

# Export transition data
df_transition_data.to_csv('../data/01_transitions/anoxia_transitions.csv', header=True, index=False)

# printing tips
print('\n'"---------------------------- Completed 01 organise data ----------------------------")
