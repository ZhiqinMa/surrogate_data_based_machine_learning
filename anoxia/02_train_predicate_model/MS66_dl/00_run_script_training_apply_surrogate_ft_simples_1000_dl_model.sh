#!/bin/bash

# Run commands by git bash
# Scripts for evaluation, training, prediction and graphing

# exec &> oe_log.txt # Standard output into a log file

# ./00_run_script.sh | tee oe.log  # run in git bash

echo '---------------------------- Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

## 1、CNN_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: MS66
Data_folder='MS66'                      ## (Set it up to your needs)
surr_type='FT'                          ## (Set it up to your needs)
simples=1000                            ## (Set it up to your needs)
ID_train='S5_S4'                        ## (Set it up to your needs)

DL_model='CNN_model'                    ## (Set it up to your needs)
# DL_model='three_head_CNN_model'       ## (Set it up to your needs)
# DL_model='LSTM_model'                 ## (Set it up to your needs)
# DL_model='CNN_LSTM_model'             ## (Set it up to your needs)
# DL_model='ConvLSTM_model'             ## (Set it up to your needs)
repeats=10                              ## (Set it up to your needs)
python 01_create_fit_evaluate_save_dl_model.py $Data_folder $surr_type $simples $ID_train $DL_model $repeats &&


# # Setting parameters: MS66
# # Period=S1

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
# ID_merge_test='S3_S1'                 ## (Set it up to your needs)
ID_test='S1'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=2                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=208                            ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $DL_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&

ID_merge_tsid='[4,2]'                   ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6]'                   ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6,4,2]'               ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='all'                     ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&


# # Setting parameters: MS66
# # Period=S3

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S3'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=4                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=148                            ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&



## 2、three_head_CNN_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: MS66
# Data_folder='MS66'                    ## (Set it up to your needs)
# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)

# DL_model='CNN_model'                  ## (Set it up to your needs)
DL_model='three_head_CNN_model'         ## (Set it up to your needs)
# DL_model='LSTM_model'                 ## (Set it up to your needs)
# DL_model='CNN_LSTM_model'             ## (Set it up to your needs)
# DL_model='ConvLSTM_model'             ## (Set it up to your needs)
repeats=10                              ## (Set it up to your needs)
python 01_create_fit_evaluate_save_dl_model.py $Data_folder $surr_type $simples $ID_train $DL_model $repeats &&


# # Setting parameters: MS66
# # Period=S1

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S1'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=2                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=208                            ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $DL_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&

ID_merge_tsid='[4,2]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6,4,2]'               ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='all'                     ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&


# # Setting parameters: MS66
# # Period=S3

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S3'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=4                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=148                            ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&



## 3、LSTM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: MS66
# Data_folder='MS66'                    ## (Set it up to your needs)
# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)

# DL_model='CNN_model'                  ## (Set it up to your needs)
# DL_model='three_head_CNN_model'       ## (Set it up to your needs)
DL_model='LSTM_model'                   ## (Set it up to your needs)
# DL_model='CNN_LSTM_model'             ## (Set it up to your needs)
# DL_model='ConvLSTM_model'             ## (Set it up to your needs)
repeats=10                              ## (Set it up to your needs)
python 01_create_fit_evaluate_save_dl_model.py $Data_folder $surr_type $simples $ID_train $DL_model $repeats &&


# # Setting parameters: MS66
# # Period=S1

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S1'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=2                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=208                            ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $DL_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&

ID_merge_tsid='[4,2]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6,4,2]'               ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='all'                     ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&


# # Setting parameters: MS66
# # Period=S3

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S3'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=4                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=148                            ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&



## 4、CNN_LSTM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: MS66
# Data_folder='MS66'                    ## (Set it up to your needs)
# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)

# DL_model='CNN_model'                  ## (Set it up to your needs)
# DL_model='three_head_CNN_model'       ## (Set it up to your needs)
# DL_model='LSTM_model'                 ## (Set it up to your needs)
DL_model='CNN_LSTM_model'               ## (Set it up to your needs)
# DL_model='ConvLSTM_model'             ## (Set it up to your needs)
repeats=10                              ## (Set it up to your needs)
python 01_create_fit_evaluate_save_dl_model.py $Data_folder $surr_type $simples $ID_train $DL_model $repeats &&


# # Setting parameters: MS66
# # Period=S1

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S1'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=2                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=208                            ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $DL_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&

ID_merge_tsid='[4,2]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6,4,2]'               ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='all'                     ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&


# # Setting parameters: MS66
# # Period=S3

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S3'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=4                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=148                            ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&



## 5、ConvLSTM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: MS66
# Data_folder='MS66'                    ## (Set it up to your needs)
# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)

# DL_model='CNN_model'                  ## (Set it up to your needs)
# DL_model='three_head_CNN_model'       ## (Set it up to your needs)
# DL_model='LSTM_model'                 ## (Set it up to your needs)
# DL_model='CNN_LSTM_model'             ## (Set it up to your needs)
DL_model='ConvLSTM_model'               ## (Set it up to your needs)
repeats=10                              ## (Set it up to your needs)
python 01_create_fit_evaluate_save_dl_model.py $Data_folder $surr_type $simples $ID_train $DL_model $repeats &&


# # Setting parameters: MS66
# # Period=S1

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S1'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=2                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=208                            ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $DL_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&

ID_merge_tsid='[4,2]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6]'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='[8,6,4,2]'               ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&

ID_merge_tsid='all'                     ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&
python 07_compute_roc_ktau_dl.py $surr_type $simples $ID_merge_tsid $DL_model $repeats $features &&


# # Setting parameters: MS66
# # Period=S3

# surr_type='FT'                        ## (Set it up to your needs)
# simples=1000                          ## (Set it up to your needs)
# ID_train='S5_S4'                      ## (Set it up to your needs)
ID_test='S3'                            ## (Set it up to your needs)
Variable_label='Mo'                     ## (Set it up to your needs)
tsid=4                                  ## (Set it up to your needs)
features=220                            ## (Set it up to your needs)
sw_width=148                            ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_dl_model.py $surr_type $simples $ID_train $ID_test $DL_model $repeats &&
python 05_plot_ews_dl_model.py $surr_type $simples $ID_train $ID_test $Variable_label $DL_model $sw_width $tsid &&



echo '---------------------------- complete end ----------------------------'

