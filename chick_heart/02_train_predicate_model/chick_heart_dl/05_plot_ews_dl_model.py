# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

Plotting pictures

# 8 parameters to set:  surr_type=str, simples=int, ID_train=str, ID_test=str,
                        Variable_label= str, DL_model=str, sw_width=int, tsid=int,

@author: Zhiqin Ma
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg') #或者其他交互式后端，如Qt5Agg
# import warnings
# warnings.filterwarnings("ignore")

print('------------------------------ 05 Start plots ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                    ## (Set it up to your needs)
# simples = 1000                        ## (Set it up to your needs)
# ID_train = '8'                        ## (Set it up to your needs)
# ID_test = '14'                        ## (Set it up to your needs)
# Variable_label = 'chick_heart'        ## (Set it up to your needs)
# DL_model = 'CNN_model'                ## (Set it up to your needs)
# # DL_model='three_head_CNN_model'     ## (Set it up to your needs)
# # DL_model='LSTM_model'               ## (Set it up to your needs)
# # DL_model='CNN_LSTM_model'           ## (Set it up to your needs)
# # DL_model='ConvLSTM_model'           ## (Set it up to your needs)
# sw_width = 150                        ## (Set it up to your needs)
# tsid = 14                             ## (Set it up to your needs)


# Setting parameters
surr_type = str(sys.argv[1])            ## (Set it up to your needs)
simples = int(sys.argv[2])              ## (Set it up to your needs)
ID_train = str(sys.argv[3])             ## (Set it up to your needs)
ID_test = str(sys.argv[4])              ## (Set it up to your needs)
Variable_label = str(sys.argv[5])       ## (Set it up to your needs)
DL_model = str(sys.argv[6])             ## (Set it up to your needs)
sw_width = int(sys.argv[7])             ## (Set it up to your needs)
tsid = int(sys.argv[8])                 ## (Set it up to your needs)


print('surr_type:', surr_type)
print('simples:', simples)
print('ID_train:', ID_train)
print('ID_test:', ID_test)
print('Variable_label:', Variable_label)
print('DL_model:', DL_model)
print('sw_width:', sw_width)
print('tsid:', tsid)
print()


try:
    os.mkdir('05_data_ews')
except:
    print('05_data_ews directory already exists!')
try:
    os.mkdir('05_data_ews/{}'.format(surr_type))
except:
    print('05_data_ews/{} directory already exists!'.format(surr_type))

try:
    os.mkdir('05_figures_ews')
except:
    print('05_figures_ews directory already exists!')
try:
    os.mkdir('05_figures_ews/{}'.format(surr_type))
except:
    print('05_figures_ews/{} directory already exists!'.format(surr_type))


# load dataset
df_transition = pd.read_csv('../../01_data_preprocessing/01_organise__transition__ews/raw_data/df_transitions.csv',header=0)
df_ews = pd.read_csv('../../01_data_preprocessing/01_organise__transition__ews/output_data/01_ews/df_ews_chick_heart.csv', header=0)
df_SD_probability_mean = pd.read_csv("04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model), header=None)
df_SD_probability_error = pd.read_csv("04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model), header=None)

Record_Core = Variable_label

# get 0 and 1 column names
column_time = df_ews.columns[0]
column_state = df_ews.columns[1]
# print(column_time, column_state)

transition = df_transition[df_transition['tsid'] == tsid]['transition'].iloc[0]
len_series = transition * 1.15 + 1
# print(transition)
df = df_ews[[column_time, column_state]][(df_ews['tsid'] == tsid) & (df_ews[column_time] <= len_series)].reset_index(drop=True)
# print(df)
# # review shape
# print('df_tsid_{}.shape:'.format(tsid), df.shape)


# set parameter
# sliding_window = df1.shape[0] - df2.shape[0]
print('sliding_window_{}:'.format(ID_test), sw_width)
# add time in predict_probability file

# for df_SD_probability_mean
value_mean = []
for t in range(len(df_SD_probability_mean)):
    value_mean.append(df_ews.iloc[t + sw_width - 1, 0])
df_SD_probability_mean.insert(loc=0, column='Time', value=pd.Series(value_mean))
# renaem column
df_SD_probability_mean.rename(columns={'Time': 'Time', 0: 'SD_probability_mean'}, inplace=True)
# record
x_SD = df_SD_probability_mean['Time']
y_SD = df_SD_probability_mean['SD_probability_mean']

# for df_SD_probability_error
value_error = []
for t in range(len(df_SD_probability_error)):
    value_error.append(df_ews.iloc[t + sw_width - 1, 0])
df_SD_probability_error.insert(loc=0, column='Time', value=pd.Series(value_error))
# renaem column
df_SD_probability_error.rename(columns={'Time': 'Time', 0: 'SD_probability_error'}, inplace=True)
# record
y_SD_error = df_SD_probability_error['SD_probability_error']

# add SD_probability_error column
df_SD_probability = df_SD_probability_mean
df_SD_probability['SD_probability_error'] = y_SD_error
# print(df_SD_probability)

# save predict_probability
df_SD_probability.to_csv("05_data_ews/{}/prediction_probability_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, surr_type, simples, ID_test, DL_model), header=True, index=False)

# # method 1
# # red, blue line
# # plot sub-figure 1
# plt.subplot(211)
# plt.title('construction_activity_train_{}_{}'.format(ID_train, ID_test), fontsize=14)
# plt.plot(x_transitions, y_transitions)
# plt.ylabel('Construction activity', fontsize=14)
# plt.xlim([start_age, end_age])
# # plotting vertical lines
# # plt.axvline(x=-Transition, color='red', linestyle='--')
# # plotting vertical lines with shadows
# plt.axvspan(transition, end_age, alpha=0.5, color='gray')
# # plot sub-figure 2
# plt.subplot(212)
# plt.plot(x_SD, y_SD, 'r')
# plt.fill_between(x_SD, y_SD - y_SD_error, y_SD + y_SD_error, alpha=0.2)
# # plt.errorbar(x_SD, y_SD, yerr=y_SD_error, fmt='-o', capsize=5)
# plt.xlabel('Age (kyr BP)', fontsize=14)
# plt.ylabel('SD probability', fontsize=14)
# plt.xlim([start_age, end_age])
# # plotting vertical lines
# # plt.axvline(x=-Transition, color='red', linestyle='--')
# # plotting vertical lines with shadows
# plt.axvspan(transition, end_age, alpha=0.5, color='gray')
# # save figure
# plt.savefig('figures/{}_train_{}_text_{}_{}.png'.format(surr_type, ID_train, ID_test, DL_model), dpi=300)
# plt.show()


# method 2
# Create subplots
fig, ax = plt.subplots(2, 1, sharex=False)
ax[0].set_title('{}_train_{}_test_{}'.format(Record_Core, ID_train, ID_test), fontsize=14)
ax[0].plot(df[column_time], df[column_state])
ax[0].set_ylabel('IBI (s)', fontsize=14)
ax[0].set_xlim([0, len_series])
# plotting vertical lines
ax[0].axvline(x=transition, color='gray', linestyle='--')
# plotting vertical lines with shadows
# ax[0].axvspan(transition, len_series, alpha=0.5, color='gray')
# Plot with error band for subplot 2
ax[1].plot(x_SD, y_SD)
ax[1].fill_between(x_SD, y_SD - y_SD_error, y_SD + y_SD_error, alpha=0.2)
# ax[1].errorbar(x_SD, y_SD, yerr=y_SD_error, fmt='-o', capsize=5)
ax[1].set_xlabel('Time', fontsize=14)
ax[1].set_ylabel('SDML probability', fontsize=14)
ax[1].set_xlim([0, len_series])
ax[1].axvline(x=transition, color='gray', linestyle='--')
# ax[1].axvspan(transition, len_series, alpha=0.5, color='gray')
# Adjust spacing between subplots
# plt.subplots_adjust(wspace=-1.3)

# save figure
plt.savefig('05_figures_ews/{}/{}_surrogate_{}_train_{}_test_{}_simples_{}_{}.png'.format(surr_type, Record_Core, surr_type, ID_train, ID_test, simples, DL_model), dpi=300)
# Show plot
# plt.show()

print('------------------------------ 05 Completed plots ------------------------------')
print()
