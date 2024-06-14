# -*- coding: utf-8 -*-
"""
Created on March 18, 2023 21:13:10

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
# ID_train = 'IV_III'                   ## (Set it up to your needs)
# ID_test = 'II'                        ## (Set it up to your needs)
# Variable_label = 'paleoclimate'       ## (Set it up to your needs)
# DL_model = 'CNN_model'                ## (Set it up to your needs)
# # DL_model='three_head_CNN_model'     ## (Set it up to your needs)
# # DL_model='LSTM_model'               ## (Set it up to your needs)
# # DL_model='CNN_LSTM_model'           ## (Set it up to your needs)
# # DL_model='ConvLSTM_model'           ## (Set it up to your needs)
# sw_width = 446                        ## (Set it up to your needs)
# tsid = 2                              ## (Set it up to your needs)

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
    os.mkdir('05_ews_data_and_figures')
except:
    print('05_ews_data_and_figures directory already exists!')

try:
    os.mkdir('05_ews_data_and_figures/data')
except:
    print('05_ews_data_and_figures/data directory already exists!')
try:
    os.mkdir('05_ews_data_and_figures/data/{}'.format(surr_type))
except:
    print('05_ews_data_and_figures/data/{} directory already exists!'.format(surr_type))

try:
    os.mkdir('05_ews_data_and_figures/figures')
except:
    print('05_ews_data_and_figures/figures directory already exists!')
try:
    os.mkdir('05_ews_data_and_figures/figures/{}'.format(surr_type))
except:
    print('05_ews_data_and_figures/figures/{} directory already exists!\n'.format(surr_type))



# load dataset
df_transitions = pd.read_csv(r'../../01_data_preprocessing/01_organise__transition__interpolate__ews/data/02_transitions_interpolate/paleoclimate_transitions_interpolate.csv', header=0)
df_ews = pd.read_csv(r'../../01_data_preprocessing/01_organise__transition__interpolate__ews/data/03_ews/paleoclimate_df_ews_interpolate.csv', header=0)
df_SD_probability_mean = pd.read_csv(r"04_prediction_data/prediction_probability_mean_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model), header=None)
df_SD_probability_error = pd.read_csv(r"04_prediction_data/prediction_probability_error_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test, DL_model), header=None)

# # Make Age and Transition negative
df_transitions['Age'] = -df_transitions['Age']
df_transitions['Transition_start'] = -df_transitions['Transition_start']
df_transitions['Transition_end'] = -df_transitions['Transition_end']
# print(df_transitions)

# record Record
Record_period = df_transitions['Record'][df_transitions['Period'] == ID_test].iloc[-1]
print('Record_period:', Record_period)

# record start Age
start_age = df_transitions['Age'][df_transitions['Period'] == ID_test].min()
# record transition
transition_start = df_transitions['Transition_start'][df_transitions['Period'] == ID_test].iloc[-1]
# recoed end Age
transition_end = df_transitions['Transition_end'][df_transitions['Period'] == ID_test].iloc[-1]
print('start_age, transition_start, transition_end:', start_age, transition_start, transition_end)

# select columns
df_transitions = df_transitions[['Age', 'Proxy']][(df_transitions['Period'] == ID_test)]
df_ews = df_ews[['Age', 'state']][(df_ews['Period'] == ID_test)]
# print(df_transitions)

# record
x_transitions = df_transitions['Age']
y_transitions = df_transitions['Proxy']

# set parameter
# sliding_window = df1.shape[0] - df2.shape[0]
print('sliding_window_{}:'.format(ID_test), sw_width)
# add time in predict_probability file
# for df_SD_probability_mean
value_mean = []
for t in range(len(df_SD_probability_mean)):
    value_mean.append(df_ews.iloc[t + sw_width - 1, 0])
df_SD_probability_mean.insert(loc=0, column='Time', value=value_mean)
# renaem column
df_SD_probability_mean.rename(columns={'Time': 'Age', 0: 'SD_probability_mean'}, inplace=True)
# record
x_SD = df_SD_probability_mean['Age']
y_SD = df_SD_probability_mean['SD_probability_mean']

# for df_SD_probability_error
value_error = []
for t in range(len(df_SD_probability_error)):
    value_error.append(df_ews.iloc[t + sw_width - 1, 0])
df_SD_probability_error.insert(loc=0, column='Time', value=value_error)
# renaem column
df_SD_probability_error.rename(columns={'Time': 'Age', 0: 'SD_probability_error'}, inplace=True)
# record
y_SD_error = df_SD_probability_error['SD_probability_error']

# add SD_probability_error column
df_SD_probability = df_SD_probability_mean
df_SD_probability['SD_probability_error'] = y_SD_error
# print(df_SD_probability)

# save predict_probability
df_SD_probability.to_csv("05_ews_data_and_figures/data/{}/prediction_probability_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, surr_type, simples, ID_test, DL_model), header=True, index=False)

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
ax[0].set_title('{}_train_{}_test_{}'.format(Variable_label, ID_train, ID_test), fontsize=14)
ax[0].plot(x_transitions, y_transitions)
ax[0].set_ylabel('\u03B4D (\u2030)', fontsize=14)
ax[0].set_xlim([start_age, transition_end])
# plotting vertical lines
# plt.axvline(x=-Transition, color='red', linestyle='--')
# plotting vertical lines with shadows
ax[0].axvspan(transition_start, transition_end, alpha=0.5, color='gray')
# Plot with error band for subplot 2
ax[1].plot(x_SD, y_SD)
ax[1].fill_between(x_SD, y_SD - y_SD_error, y_SD + y_SD_error, alpha=0.2)
# ax[1].errorbar(x_SD, y_SD, yerr=y_SD_error, fmt='-o', capsize=5)
ax[1].set_xlabel('Age (yr BP)', fontsize=14)
ax[1].set_ylabel('SDML probability', fontsize=14)
ax[1].set_xlim([start_age, transition_end])
ax[1].axvspan(transition_start, transition_end, alpha=0.5, color='gray')
# Adjust spacing between subplots
# plt.subplots_adjust(wspace=-1.3)

# save figure
plt.savefig('05_ews_data_and_figures/figures/{}/{}_surrogate_{}_train_{}_test_{}_simples_{}_{}.png'.format(surr_type, Variable_label, surr_type, ID_train, ID_test, simples, DL_model), dpi=300)
# Show plot
# plt.show()

print('------------------------------ 05 Completed plots ------------------------------')
print()

