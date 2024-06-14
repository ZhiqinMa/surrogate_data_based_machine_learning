# -*- coding: utf-8 -*-
"""
Created on January 1, 2024 14:13:19

- Compute the ROC curve and AUC for DL in chick heart data

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
import statistics
from numpy import mean, std, expand_dims
from pandas import read_csv
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# import warnings
# warnings.filterwarnings("ignore")

np.random.seed(0)
print('------------------------------ 07 Start ROC for ml ------------------------------')

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
    os.mkdir('07_roc_data_and_figures_sdml')
except:
    print('07_roc_data_and_figures_sdml directory already exists!')

try:
    os.mkdir('07_roc_data_and_figures_sdml/data')
except:
    print('07_roc_data_and_figures_sdml/data directory already exists!')
try:
    os.mkdir('07_roc_data_and_figures_sdml/data/{}'.format(surr_type))
except:
    print('07_roc_data_and_figures_sdml/data/{} directory already exists!'.format(surr_type))
try:
    os.mkdir('07_roc_data_and_figures_sdml/data/{}/ROC'.format(surr_type))
except:
    print('07_roc_data_and_figures_sdml/data/{}/ROC directory already exists!'.format(surr_type))

try:
    os.mkdir('07_roc_data_and_figures_sdml/figures')
except:
    print('07_roc_data_and_figures_sdml/figures directory already exists!')
try:
    os.mkdir('07_roc_data_and_figures_sdml/figures/{}'.format(surr_type))
except:
    print('07_roc_data_and_figures_sdml/figures/{} directory already exists!\n'.format(surr_type))
try:
    os.mkdir('07_roc_data_and_figures_sdml/figures/{}/ROC'.format(surr_type))
except:
    print('07_roc_data_and_figures_sdml/figures/{}/ROC directory already exists!\n'.format(surr_type))


def adjust_length_pad_left(data, desired_length=features):
    """
    Adjust the length of a data sequence to the desired length by padding with zeros on the left.

    Parameters:
    - data (list or pd.Series): The input data sequence.
    - desired_length (int): The desired length of the sequence. Default is 150.

    Returns:
    - pd.Series: Adjusted data sequence of the desired length.
    """
    current_length = len(data)

    if current_length > desired_length:
        # If the data is longer than the desired length, truncate it.
        return data[-desired_length:].reset_index(drop=True)
    elif current_length < desired_length:
        # If the data is shorter than the desired length, pad it with zeros on the left.
        padding = [0] * (desired_length - current_length)
        return pd.Series(padding + data.tolist()).reset_index(drop=True)
    else:
        # If the data is already the desired length, return it as is.
        return data.reset_index(drop=True)


def roc_compute(truth_vals, indicator_vals):
    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = roc_curve(truth_vals, indicator_vals)

    # Compute AUC (area under curve)
    roc_auc = auc(fpr, tpr)

    # Put into a DF
    dic_roc = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
    df_roc = pd.DataFrame(dic_roc)
    return df_roc


def SDML_classifier(df_X, surr_type, simples, ML_model):
    # Setting parameters
    list_preds = []
    # select DL model
    for r in range(repeats):
        # Define the filepath for the best model
        filepath = 'models/surrogate_{}_simples_{}_{}_{}.sav'.format(surr_type, simples, ML_model, r)
        # Load the best model
        best_model = load(filepath)

        # Predict probabilities for the data
        y_pred_proba = best_model.predict_proba(df_X)
        # print(f'r={r},y_predicts={y_pred_proba}')

        list_preds.extend(y_pred_proba[:, 1])
    df_preds = pd.DataFrame(list_preds, columns=['SDML probability'])
    return df_preds


### ----------------------------- start code -----------------------------
if __name__ == '__main__':
    # Load in trajectory data
    df = pd.read_csv('../../01_data_preprocessing/01_organise__transition__ews/data/03_prepare_roc_data/prepare_roc_data.csv', header=0)
    df_forced = df[df['type'] == 'forced']
    df_null = df[df['type'] == 'neutral']
    # print(df_forced)

    # --------------
    # Compute dl probability
    # ---------------
    # eval_pts = np.arange(0.64, 1.01, 0.04)    # percentage of way through pre-transition time series: 10
    eval_pts = np.arange(0.61, 1.01, 0.01)      # percentage of way through pre-transition time series: 40
    print('len_eval_pts =', len(eval_pts))
    print()

    # # get index_column (last) column names
    index_column = df.columns[-1]
    # print('index_column', index_column)

    # --------------
    # forced trajectories
    # ---------------
    print('----- 01 forced trajectories and compute EWS -----')

    if ID_merge_tsid == 'all':
        list_tsid_forced = df_forced['tsid'].unique()                       # Select all tsid
    else:
        if isinstance([ID_merge_tsid][0], str):
            # 如果是字符串列表，使用 ast.literal_eval 进行解析
            list_tsid_forced = ast.literal_eval([ID_merge_tsid][0])         # or Select specific tsid
        else:
            # 如果已经是数字列表，直接赋值
            list_tsid_forced = ID_merge_tsid                                # or Select specific tsid
    list_data_forced = []                                                   # A list used to collect all predictions
    for tsid in list_tsid_forced:
        df_spec = df_forced[df_forced['tsid'] == tsid].set_index(index_column)
        transition = df_forced[df_forced['tsid'] == tsid].shape[0]
        series = df_spec['state']

        for eval_pt in eval_pts:
            eval_time = transition * eval_pt
            X_adjusted_data = adjust_length_pad_left(series[:int(eval_time)], features)
            # print(adjusted_data.shape)
            list_data_forced.append(X_adjusted_data)

        print('Complete for forced tsid={}'.format(tsid))

    # concat all DataFrame
    data_forced = pd.DataFrame(list_data_forced)
    # Using SDML classifier to predictions
    df_preds_forced = SDML_classifier(data_forced, surr_type, simples, ML_model)

    # Export data
    df_preds_forced.to_csv(f'07_roc_data_and_figures_sdml/data/{surr_type}/df_forced_fixed_surrogate_{surr_type}_simples_{simples}_{ML_model}_{ID_merge_tsid}.csv', index=False)


    # -------------
    # null trajectories
    # -------------
    print('----- 01 null trajectories and compute EWS -----')

    if ID_merge_tsid == 'all':
        list_tsid_null = df_forced['tsid'].unique()                             # Select all tsid
    else:
        if isinstance([ID_merge_tsid][0], str):
            # 如果是字符串列表，使用 ast.literal_eval 进行解析
            list_tsid_null = ast.literal_eval([ID_merge_tsid][0])           # or Select specific tsid
        else:
            # 如果已经是数字列表，直接赋值
            list_tsid_null = ID_merge_tsid                                  # or Select specific tsid
    list_data_null = []                                                    # A list used to collect all predictions
    for tsid in list_tsid_null:
        df_spec = df_null[df_null['tsid'] == tsid].set_index(index_column)
        transition = df_null[df_null['tsid'] == tsid].shape[0]
        # print('transition=', transition)
        series = df_spec['state']
        # print(series)

        for eval_pt in eval_pts:
            eval_time = transition * eval_pt
            X_adjusted_data = adjust_length_pad_left(series[:int(eval_time)], features)
            # print(adjusted_data.shape)
            list_data_null.append(X_adjusted_data)

        print('Complete for null tsid={}'.format(tsid))

        # concat all DataFrame
        data_null = pd.DataFrame(list_data_null)
        # Using SDML classifier to predictions
        df_preds_null = SDML_classifier(data_null, surr_type, simples, ML_model)

    # Export data
    df_preds_null.to_csv(f'07_roc_data_and_figures_sdml/data/{surr_type}/df_null_fixed_surrogate_{surr_type}_simples_{simples}_{ML_model}_{ID_merge_tsid}.csv', index=False)


    # ----------------
    # compute ROC curves
    # ----------------
    print('----- 02 Compute ROC curves -----')

    df_preds_forced['truth_value'] = 1
    df_preds_null['truth_value'] = 0

    df_dl = pd.concat([df_preds_forced, df_preds_null])

    # Initiliase list for ROC dataframes for predicting May fold bifurcation
    list_roc = []

    # Assign indicator and truth values for variance
    indicator_vals = df_dl['SDML probability']
    truth_vals = df_dl['truth_value']
    df_roc = roc_compute(truth_vals, indicator_vals)
    df_roc['ews'] = 'SDML'
    list_roc.append(df_roc)
    # df_dl.to_csv('output2/df_dl_ml.csv', index=False,)

    # Concatenate roc dataframes
    df_roc_SDML = pd.concat(list_roc, ignore_index=True)

    # Export ROC data
    filepath_data = f'07_roc_data_and_figures_sdml/data/{surr_type}/ROC/df_roc_surrogate_{surr_type}_simples_{simples}_{ML_model}_{ID_merge_tsid}.csv'
    df_roc_SDML.to_csv(filepath_data, index=False, )
    print('Exported ROC data to {}'.format(filepath_data))

    # Load data
    df_roc_var_corr = read_csv(F'06_roc_data_and_figures_var_corr/data/ROC/df_roc_var_corr_{ID_merge_tsid}.csv', header=0)
    df_roc_var = df_roc_var_corr[df_roc_var_corr['ews'] == 'Variance']
    df_roc_AC = df_roc_var_corr[df_roc_var_corr['ews'] == 'Lag-1 AC']

    # Figures ROC curve
    df_roc_SDML.replace([np.inf, -np.inf], 0, inplace=True)   #替换正、负inf为0
    fig, ax = plt.subplots()
    ax.plot(df_roc_SDML['fpr'], df_roc_SDML['tpr'], label='SDML (AUC = {:.2f})'.format(df_roc_SDML['auc'].iloc[0]))
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
    filepath_fig = f'07_roc_data_and_figures_sdml/figures/{surr_type}/ROC/figure_roc_surrogate_{surr_type}_simples_{simples}_{ML_model}_{ID_merge_tsid}.png'
    plt.savefig(filepath_fig, dpi=300)
    print('Exported ROC figure to {}'.format(filepath_fig))

    # Time taken for script to run
    end_time = time.time()
    time_taken = end_time - start_time
    print('Ran in {:.2f}s'.format(time_taken))

    print('------------------------------ 07 Completed ROC for ml ------------------------------')
    print()