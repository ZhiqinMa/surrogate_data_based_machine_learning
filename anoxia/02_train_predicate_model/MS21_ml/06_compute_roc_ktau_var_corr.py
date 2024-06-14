# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

- Compute the ROC curve and AUC for variance and lag-1 autocorrelation
- Compute kendall tau at fixed evaluation points in tree felling data

# 6 parameters to set: surr_type=str, simples=int, ID_merge_tsid=str, ML_model=str, repeats=int, features=int

@author: Zhiqin Ma
"""


# Start timer to record execution time
import time
start_time = time.time()

import os
import ast
import sys
import ewstools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# import warnings
# warnings.filterwarnings("ignore")

np.random.seed(0)
print('------------------------------ 06 Start ROC for variance and lag-1 autocorrelation ------------------------------')

# # Setting parameters
# surr_type = 'AAFT'                        ## (Set it up to your needs)
# simples = 1000                            ## (Set it up to your needs)
# ID_merge_tsid = 'all'                     ## (Set it up to your needs)
# # ID_merge_tsid = '[3,1]'                 ## (Set it up to your needs)
# # ID_merge_tsid = '[3]'                   ## (Set it up to your needs)
# # ID_merge_tsid = '[1]'                   ## (Set it up to your needs)
# ML_model = 'SVM_model'                    ## (Set it up to your needs)
# # ML_model = 'Bagging_model'              ## (Set it up to your needs)
# # ML_model = 'RF_model'                   ## (Set it up to your needs)
# # ML_model = 'GBM_model'                  ## (Set it up to your needs)
# # ML_model = 'Xgboost_model'              ## (Set it up to your needs)
# # ML_model = 'LGBM_model'                 ## (Set it up to your needs)
# repeats = 10                              ## (Set it up to your needs)
# features = 450                            ## (Set it up to your needs)

# Setting parameters
surr_type = str(sys.argv[1])                ## (Set it up to your needs)
simples = int(sys.argv[2])                  ## (Set it up to your needs)
ID_merge_tsid = str(sys.argv[3])            ## (Set it up to your needs)
ML_model = str(sys.argv[4])                 ## (Set it up to your needs)
repeats = int(sys.argv[5])                  ## (Set it up to your needs)
features = int(sys.argv[6])                 ## (Set it up to your needs)

print('surr_type:', surr_type)
print('simples:', simples)
print('ID_merge_tsid:', ID_merge_tsid)
print('ML_model:', ML_model)
print('repeats:', repeats)
print('features:', features)
print()


try:
    os.mkdir('06_roc_data_and_figures_var_corr')
except:
    print('06_roc_data_and_figures_var_corr directory already exists!')

try:
    os.mkdir('06_roc_data_and_figures_var_corr/data')
except:
    print('06_roc_data_and_figures_var_corr/data directory already exists!')
try:
    os.mkdir('06_roc_data_and_figures_var_corr/data/ROC')
except:
    print('06_roc_data_and_figures_var_corr/data/ROC directory already exists!')

try:
    os.mkdir('06_roc_data_and_figures_var_corr/figures')
except:
    print('06_roc_data_and_figures_var_corr/figures directory already exists!')
try:
    os.mkdir('06_roc_data_and_figures_var_corr/figures/ROC')
except:
    print('06_roc_data_and_figures_var_corr/figures/ROC directory already exists!')

def roc_compute(truth_vals, indicator_vals):
    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = roc_curve(truth_vals, indicator_vals)

    # Compute AUC (area under curve)
    roc_auc = auc(fpr, tpr)

    # Put into a DF
    dic_roc = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
    df_roc = pd.DataFrame(dic_roc)
    return df_roc



### ----------------------------- start code -----------------------------
if __name__ == '__main__':
    # Load in trajectory data
    df = pd.read_csv('../../01_data_preprocessing/01_organise__transition__ews/data/03_prepare_roc_data/prepare_roc_data.csv', header=0)
    df_forced = df[df['type'] == 'forced']
    df_null = df[df['type'] == 'neutral']
    # print(df_pd)

    # --------------
    # Compute EWS and kendall tau
    # ---------------
    # eval_pts = np.arange(0.64, 1.01, 0.04)    # percentage of way through pre-transition time series: 10
    eval_pts = np.arange(0.61, 1.01, 0.01)      # percentage of way through pre-transition time series: 40
    print('len_eval_pts=', len(eval_pts))
    print()

    # EWS parameters
    rw = 0.5            # half the length of the data
    span = 0.2          # span for Lowess filtering
    bandwidth = 0.09    # BW used in paper = 900yr = 0.09% of pre-transiiton data (10,000kyr)

    # # get index_column (last) column names
    index_column = df.columns[-1]
    # print('index_column', index_column)

    # --------------
    # pd trajectories
    # ---------------
    print('----- 01 forced trajectories and compute EWS -----')

    if ID_merge_tsid == 'all':
        list_tsid_forced = df_forced['tsid'].unique()                           # Select all tsid
    else:
        if isinstance([ID_merge_tsid][0], str):
            # 如果是字符串列表，使用 ast.literal_eval 进行解析
            list_tsid_forced = ast.literal_eval([ID_merge_tsid][0])         # or Select specific tsid
        else:
            # 如果已经是数字列表，直接赋值
            list_tsid_forced = ID_merge_tsid                                # or Select specific tsid

    list_ktau_forced = []                                                   # A list used to collect all predictions
    for tsid in list_tsid_forced:
        df_spec = df_forced[df_forced['tsid'] == tsid].set_index(index_column)
        transition = df_forced[df_forced['tsid'] == tsid].shape[0]
        # print('transition=', transition)
        series = df_spec['state']

        # Compute EWS
        ts = ewstools.TimeSeries(series, transition=transition)
        ts.detrend(method='Gaussian', span=span)

        ts.compute_var(rolling_window=rw)
        ts.compute_auto(rolling_window=rw, lag=1)

        for eval_pt in eval_pts:
            eval_time = transition * eval_pt

            # Compute kendall tau at evaluation points
            ts.compute_ktau(tmin=0, tmax=eval_time)
            dic_ktau_forced = ts.ktau
            dic_ktau_forced['eval_time'] = eval_time
            dic_ktau_forced['tsid'] = tsid
            list_ktau_forced.append(dic_ktau_forced)

        print('Complete for forced tsid={}'.format(tsid))

    df_ktau_forced = pd.DataFrame(list_ktau_forced)

    # -------------
    # null trajectories
    # -------------
    print('----- 01 null trajectories and compute EWS -----')

    if ID_merge_tsid == 'all':
        list_tsid_null = df_null['tsid'].unique()                           # Select all tsid
    else:
        if isinstance([ID_merge_tsid][0], str):
            # 如果是字符串列表，使用 ast.literal_eval 进行解析
            list_tsid_null = ast.literal_eval([ID_merge_tsid][0])           # or Select specific tsid
        else:
            # 如果已经是数字列表，直接赋值
            list_tsid_null = ID_merge_tsid                                  # or Select specific tsid
    list_ktau_null = []                                                     # A list used to collect all predictions
    for tsid in list_tsid_null:
        df_spec = df_null[df_null['tsid'] == tsid].set_index(index_column)
        transition = df_null[df_null['tsid'] == tsid].shape[0]
        series = df_spec['state']

        # Compute EWS
        ts = ewstools.TimeSeries(series, transition=transition)
        ts.detrend(method='Gaussian', span=span)

        ts.compute_var(rolling_window=rw)
        ts.compute_auto(rolling_window=rw, lag=1)

        for eval_pt in eval_pts:
            eval_time = transition * eval_pt

            # Compute kendall tau at evaluation points
            ts.compute_ktau(tmin=0, tmax=eval_time)
            dic_ktau_null = ts.ktau
            dic_ktau_null['eval_time'] = eval_time
            dic_ktau_null['tsid'] = tsid
            list_ktau_null.append(dic_ktau_null)

        print('Complete for null tsid={}'.format(tsid))

    df_ktau_null = pd.DataFrame(list_ktau_null)

    # Export data
    df_ktau_forced.to_csv(f'06_roc_data_and_figures_var_corr/data/df_ktau_forced_fixed_{ID_merge_tsid}.csv', index=False)
    df_ktau_null.to_csv(f'06_roc_data_and_figures_var_corr/data/df_ktau_null_fixed_{ID_merge_tsid}.csv', index=False)

    # ----------------
    # compute ROC curves
    # ----------------
    print('----- 02 Compute ROC curves -----')

    df_ktau_forced['truth_value'] = 1
    df_ktau_null['truth_value'] = 0

    df_ktau = pd.concat([df_ktau_forced, df_ktau_null])

    # Initiliase list for ROC dataframes for predicting May fold bifurcation
    list_roc = []

    # Assign indicator and truth values for variance
    indicator_vals = -df_ktau['variance']
    truth_vals = df_ktau['truth_value']
    df_roc = roc_compute(truth_vals, indicator_vals)
    df_roc['ews'] = 'Variance'
    list_roc.append(df_roc)
    # df_ktau.to_csv('output2/df_dl_variance.csv', index=False,)

    # Assign indicator and truth values for lag-1 AC
    indicator_vals = df_ktau['ac1']
    truth_vals = df_ktau['truth_value']
    df_roc = roc_compute(truth_vals, indicator_vals)
    df_roc['ews'] = 'Lag-1 AC'
    list_roc.append(df_roc)
    # df_ktau.to_csv('output2/df_dl_lag-1 AC.csv', index=False,)

    # Concatenate roc dataframes
    df_roc_var_corr = pd.concat(list_roc, ignore_index=True)

    # Export ROC data
    filepath_data = f'06_roc_data_and_figures_var_corr/data/ROC/df_roc_var_corr_{ID_merge_tsid}.csv'
    df_roc_var_corr.to_csv(filepath_data, index=False, )
    print('Exported ROC data to {}'.format(filepath_data))

    # Figures ROC curve
    df_roc_var_corr.replace([np.inf, -np.inf], 0, inplace=True)   #替换正、负inf为0
    fig, ax = plt.subplots()
    df_roc_var = df_roc_var_corr[df_roc_var_corr['ews'] == 'Variance']
    df_roc_AC = df_roc_var_corr[df_roc_var_corr['ews'] == 'Lag-1 AC']
    ax.plot(df_roc_var['fpr'], df_roc_var['tpr'], label='Variance (AUC = {:.2f})'.format(df_roc_var['auc'].iloc[0]))
    ax.plot(df_roc_AC['fpr'], df_roc_AC['tpr'], label='Lag-1 AC (AUC = {:.2f})'.format(df_roc_AC['auc'].iloc[0]))
    ax.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    # plt.show()
    # save figure
    filepath_fig = f'06_roc_data_and_figures_var_corr/figures/ROC/figure_roc_var_corr_{ID_merge_tsid}.png'
    plt.savefig(filepath_fig, dpi=300)
    print('Exported ROC figure to {}'.format(filepath_fig))

    # Time taken for script to run
    end_time = time.time()
    time_taken = end_time - start_time
    print('Ran in {:.2f}s'.format(time_taken))

    print('------------------------------ 06 Completed ROC for variance and lag-1 autocorrelation ------------------------------')
    print()
