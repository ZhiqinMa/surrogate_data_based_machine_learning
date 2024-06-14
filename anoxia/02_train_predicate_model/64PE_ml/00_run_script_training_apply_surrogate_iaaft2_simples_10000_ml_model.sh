#!/bin/bash

# Run commands by git bash
# Scripts for evaluation, training, prediction and graphing

# exec &> oe_log.txt # Standard output into a log file

# ./00_run_script.sh | tee oe.log  # run in git bash

echo '---------------------------- Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

## 1、SVM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: 64PE
Data_folder='64PE'                              ## (Set it up to your needs)
surr_type='IAAFT2'                              ## (Set it up to your needs)
simples=10000                                   ## (Set it up to your needs)
ID_train='S9_S8_S7'                             ## (Set it up to your needs)

ML_model='SVM_model'                            ## (Set it up to your needs)
# ML_model='Bagging_model'                      ## (Set it up to your needs)
# ML_model='RF_model'                           ## (Set it up to your needs)
# ML_model='GBM_model'                          ## (Set it up to your needs)
# ML_model='Xgboost_model'                      ## (Set it up to your needs)
# ML_model='LGBM_model'                         ## (Set it up to your needs)
repeats=10                                      ## (Set it up to your needs)
python 01_create_fit_evaluate_save_ml_model.py $Data_folder $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: 64PE
# # Period=S3

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
# ID_merge_test='S6_S5_S4_S3'                   ## (Set it up to your needs)
ID_test='S3'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=5                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=188                                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_tsid='[10,9,7,5]'                      ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11]'                      ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11,10,9,7,5]'             ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='all'                             ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&


# # Setting parameters: 64PE
# # Period=S4

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S4'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=7                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=217                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S5

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S5'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=9                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=162                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S6

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S6'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=10                                         ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=106                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&



## 2、Bagging_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: 64PE
# Data_folder='64PE'                            ## (Set it up to your needs)
# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)

# ML_model='SVM_model'                          ## (Set it up to your needs)
ML_model='Bagging_model'                        ## (Set it up to your needs)
# ML_model='RF_model'                           ## (Set it up to your needs)
# ML_model='GBM_model'                          ## (Set it up to your needs)
# ML_model='Xgboost_model'                      ## (Set it up to your needs)
# ML_model='LGBM_model'                         ## (Set it up to your needs)
repeats=10                                      ## (Set it up to your needs)
python 01_create_fit_evaluate_save_ml_model.py $Data_folder $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: 64PE
# # Period=S3

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S3'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=5                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=188                                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_tsid='[10,9,7,5]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11,10,9,7,5]'             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='all'                             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&


# # Setting parameters: 64PE
# # Period=S4

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S4'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=7                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=217                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S5

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S5'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=9                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=162                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S6

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S6'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=10                                         ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=106                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&



## 3、RF_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: 64PE
# Data_folder='64PE'                            ## (Set it up to your needs)
# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)

# ML_model='SVM_model'                          ## (Set it up to your needs)
# ML_model='Bagging_model'                      ## (Set it up to your needs)
ML_model='RF_model'                             ## (Set it up to your needs)
# ML_model='GBM_model'                          ## (Set it up to your needs)
# ML_model='Xgboost_model'                      ## (Set it up to your needs)
# ML_model='LGBM_model'                         ## (Set it up to your needs)
repeats=10                                      ## (Set it up to your needs)
python 01_create_fit_evaluate_save_ml_model.py $Data_folder $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: 64PE
# # Period=S3

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S3'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=5                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=188                                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_tsid='[10,9,7,5]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11,10,9,7,5]'             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='all'                             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&


# # Setting parameters: 64PE
# # Period=S4

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S4'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=7                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=217                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S5

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S5'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=9                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=162                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S6

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S6'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=10                                         ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=106                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&



## 4、GBM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: 64PE
# Data_folder='64PE'                            ## (Set it up to your needs)
# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)

# ML_model='SVM_model'                          ## (Set it up to your needs)
# ML_model='Bagging_model'                      ## (Set it up to your needs)
# ML_model='RF_model'                           ## (Set it up to your needs)
ML_model='GBM_model'                            ## (Set it up to your needs)
# ML_model='Xgboost_model'                      ## (Set it up to your needs)
# ML_model='LGBM_model'                         ## (Set it up to your needs)
repeats=10                                      ## (Set it up to your needs)
python 01_create_fit_evaluate_save_ml_model.py $Data_folder $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: 64PE
# # Period=S3

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S3'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=5                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=188                                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_tsid='[10,9,7,5]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11,10,9,7,5]'             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='all'                             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&


# # Setting parameters: 64PE
# # Period=S4

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S4'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=7                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=217                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S5

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S5'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=9                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=162                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S6

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S6'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=10                                         ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=106                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&



## 5、Xgboost_model 
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: 64PE
# Data_folder='64PE'                            ## (Set it up to your needs)
# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)

# ML_model='SVM_model'                          ## (Set it up to your needs)
# ML_model='Bagging_model'                      ## (Set it up to your needs)
# ML_model='RF_model'                           ## (Set it up to your needs)
# ML_model='GBM_model'                          ## (Set it up to your needs)
ML_model='Xgboost_model'                        ## (Set it up to your needs)
# ML_model='LGBM_model'                         ## (Set it up to your needs)
repeats=10                                      ## (Set it up to your needs)
python 01_create_fit_evaluate_save_ml_model.py $Data_folder $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: 64PE
# # Period=S3

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S3'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=5                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=188                                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_tsid='[10,9,7,5]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11,10,9,7,5]'             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='all'                             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&


# # Setting parameters: 64PE
# # Period=S4

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S4'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=7                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=217                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S5

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S5'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=9                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=162                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S6

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S6'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=10                                         ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=106                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&



## 6、LGBM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: 64PE
# Data_folder='64PE'                            ## (Set it up to your needs)
# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)

# ML_model='SVM_model'                          ## (Set it up to your needs)
# ML_model='Bagging_model'                      ## (Set it up to your needs)
# ML_model='RF_model'                           ## (Set it up to your needs)
# ML_model='GBM_model'                          ## (Set it up to your needs)
# ML_model='Xgboost_model'                      ## (Set it up to your needs)
ML_model='LGBM_model'                           ## (Set it up to your needs)
repeats=10                                      ## (Set it up to your needs)
python 01_create_fit_evaluate_save_ml_model.py $Data_folder $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: 64PE
# # Period=S3

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S3'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=5                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=188                                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_tsid='[10,9,7,5]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11]'                      ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='[13,12,11,10,9,7,5]'             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&

ID_merge_tsid='all'                             ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_tsid $ML_model $repeats $features &&


# # Setting parameters: 64PE
# # Period=S4

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S4'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=7                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=217                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S5

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S5'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=9                                          ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=162                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&


# # Setting parameters: 64PE
# # Period=S6

# surr_type='IAAFT2'                            ## (Set it up to your needs)
# simples=10000                                 ## (Set it up to your needs)
# ID_train='S9_S8_S7'                           ## (Set it up to your needs)
ID_test='S6'                                    ## (Set it up to your needs)
Variable_label='Mo'                             ## (Set it up to your needs)
tsid=10                                         ## (Set it up to your needs)
features=220                                    ## (Set it up to your needs)
sw_width=106                                    ## (Set it up to your needs)
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&



echo '---------------------------- complete end ----------------------------'

