#!/bin/bash

# Run commands by git bash
# Scripts for evaluation, training, prediction and graphing

# exec &> oe_log.txt # Standard output into a log file

# ./00_run_script.sh | tee oe.log  # run in git bash

echo '---------------------------- Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

## 1、SVM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: chick_heart
surr_type='AAFT'                ## (Set it up to your needs)
simples=1000                    ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)

ML_model='SVM_model'            ## (Set it up to your needs)
# ML_model='Bagging_model'      ## (Set it up to your needs)
# ML_model='RF_model'           ## (Set it up to your needs)
# ML_model='GBM_model'          ## (Set it up to your needs)
# ML_model='Xgboost_model'      ## (Set it up to your needs)
# ML_model='LGBM_model'         ## (Set it up to your needs)
repeats=10                      ## (Set it up to your needs)

python 01_create_fit_evaluate_save_ml_model.py $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: chick_heart
# # Period=14

# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)
ID_test='14'                    ## (Set it up to your needs)
Variable_label='chick_heart'    ## (Set it up to your needs)
tsid=14                         ## (Set it up to your needs)
features=150                    ## (Set it up to your needs)
sw_width=150                    ## (Set it up to your needs)  
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&                
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_test='[14]'                  ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8]'                  ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8,14]'                ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='all'                   ## (Set it up to your needs)
python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&


## 2、Bagging_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: chick_heart
# Data_folder='chick_heart'     ## (Set it up to your needs)
# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
# ID_train='8'                  ## (Set it up to your needs)

# ML_model='SVM_model'          ## (Set it up to your needs)
ML_model='Bagging_model'        ## (Set it up to your needs)
# ML_model='RF_model'           ## (Set it up to your needs)
# ML_model='GBM_model'          ## (Set it up to your needs)
# ML_model='Xgboost_model'      ## (Set it up to your needs)
# ML_model='LGBM_model'         ## (Set it up to your needs)
repeats=10                      ## (Set it up to your needs)

python 01_create_fit_evaluate_save_ml_model.py $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: chick_heart
# # Period=14

# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)
ID_test='14'                    ## (Set it up to your needs)
Variable_label='chick_heart'    ## (Set it up to your needs)
tsid=14                         ## (Set it up to your needs)
features=150                    ## (Set it up to your needs)
sw_width=150                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_test='[14]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8,14]'                ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='all'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&



## 3、RF_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: chick_heart
# Data_folder='chick_heart'     ## (Set it up to your needs)
# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
# ID_train='8'                  ## (Set it up to your needs)

# ML_model='SVM_model'          ## (Set it up to your needs)
# ML_model='Bagging_model'      ## (Set it up to your needs)
ML_model='RF_model'             ## (Set it up to your needs)
# ML_model='GBM_model'          ## (Set it up to your needs)
# ML_model='Xgboost_model'      ## (Set it up to your needs)
# ML_model='LGBM_model'         ## (Set it up to your needs)
repeats=10                      ## (Set it up to your needs)

python 01_create_fit_evaluate_save_ml_model.py $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: chick_heart
# # Period=14

# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)
ID_test='14'                    ## (Set it up to your needs)
Variable_label='chick_heart'    ## (Set it up to your needs)
tsid=14                         ## (Set it up to your needs)
features=150                    ## (Set it up to your needs)
sw_width=150                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_test='[14]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8,14]'                ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='all'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&



## 4、GBM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: chick_heart
# Data_folder='chick_heart'     ## (Set it up to your needs)
# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
# ID_train='8'                  ## (Set it up to your needs)

# ML_model='SVM_model'          ## (Set it up to your needs)
# ML_model='Bagging_model'      ## (Set it up to your needs)
# ML_model='RF_model'           ## (Set it up to your needs)
ML_model='GBM_model'            ## (Set it up to your needs)
# ML_model='Xgboost_model'      ## (Set it up to your needs)
# ML_model='LGBM_model'         ## (Set it up to your needs)
repeats=10                      ## (Set it up to your needs)

python 01_create_fit_evaluate_save_ml_model.py $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: chick_heart
# # Period=14

# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)
ID_test='14'                    ## (Set it up to your needs)
Variable_label='chick_heart'    ## (Set it up to your needs)
tsid=14                         ## (Set it up to your needs)
features=150                    ## (Set it up to your needs)
sw_width=150                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_test='[14]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8,14]'                ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='all'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&



## 5、Xgboost_model 
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: chick_heart
# Data_folder='chick_heart'     ## (Set it up to your needs)
# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
# ID_train='8'                  ## (Set it up to your needs)

# ML_model='SVM_model'          ## (Set it up to your needs)
# ML_model='Bagging_model'      ## (Set it up to your needs)
# ML_model='RF_model'           ## (Set it up to your needs)
# ML_model='GBM_model'          ## (Set it up to your needs)
ML_model='Xgboost_model'        ## (Set it up to your needs)
# ML_model='LGBM_model'         ## (Set it up to your needs)
repeats=10                      ## (Set it up to your needs)

python 01_create_fit_evaluate_save_ml_model.py $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: chick_heart
# # Period=14

# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)
ID_test='14'                    ## (Set it up to your needs)
Variable_label='chick_heart'    ## (Set it up to your needs)
tsid=14                         ## (Set it up to your needs)
features=150                    ## (Set it up to your needs)
sw_width=150                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_test='[14]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8,14]'                ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='all'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&



## 6、LGBM_model
echo '-------------------------------------------------------------------------------------'
# # Setting parameters: chick_heart
# Data_folder='chick_heart'     ## (Set it up to your needs)
# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
# ID_train='8'                  ## (Set it up to your needs)

# ML_model='SVM_model'          ## (Set it up to your needs)
# ML_model='Bagging_model'      ## (Set it up to your needs)
# ML_model='RF_model'           ## (Set it up to your needs)
# ML_model='GBM_model'          ## (Set it up to your needs)
# ML_model='Xgboost_model'      ## (Set it up to your needs)
ML_model='LGBM_model'           ## (Set it up to your needs)
repeats=10                      ## (Set it up to your needs)

python 01_create_fit_evaluate_save_ml_model.py $surr_type $simples $ID_train $ML_model $repeats &&


# # Setting parameters: chick_heart
# # Period=14

# surr_type='AAFT'              ## (Set it up to your needs)
# simples=1000                  ## (Set it up to your needs)
ID_train='8'                    ## (Set it up to your needs)
ID_test='14'                    ## (Set it up to your needs)
Variable_label='chick_heart'    ## (Set it up to your needs)
tsid=14                         ## (Set it up to your needs)
features=150                    ## (Set it up to your needs)
sw_width=150                    ## (Set it up to your needs)
python 02_evaluation.py $surr_type $simples $ID_train $ML_model $repeats &&
python 03_sliding_window.py $surr_type $simples $ID_train $ID_test $Variable_label $tsid $features $sw_width &&
python 04_predicate_ml_model.py $surr_type $simples $ID_train $ID_test $ML_model $repeats &&
python 05_plot_ews_ml_model.py $surr_type $simples $ID_train $ID_test $Variable_label $ML_model $sw_width $tsid &&

ID_merge_test='[14]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8]'                  ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='[8,14]'                ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&

ID_merge_test='all'                   ## (Set it up to your needs)
# python 06_compute_roc_ktau_var_corr.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&
python 07_compute_roc_ktau_ml.py $surr_type $simples $ID_merge_test $ML_model $repeats $features &&



echo '---------------------------- complete end ----------------------------'

