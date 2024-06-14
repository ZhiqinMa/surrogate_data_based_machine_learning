#!/bin/bash


echo '--------------------------------'

echo which conda:
which conda

echo which python:
which python

echo python version:
python --version

echo start date:
date
echo '--------------------------------'
echo

# Run the Python script on cpu
bash  00_run_script_training_apply_surrogate_rp_simples_1000_ml_model.sh  | tee  oe_surrogate_rp_simples_1000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_ft_simples_1000_ml_model.sh  | tee  oe_surrogate_ft_simples_1000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_aaft_simples_1000_ml_model.sh  | tee  oe_surrogate_aaft_simples_1000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_iaaft1_simples_1000_ml_model.sh  | tee  oe_surrogate_iaaft1_simples_1000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_iaaft2_simples_1000_ml_model.sh  | tee  oe_surrogate_iaaft2_simples_1000_ml_model.log  &&

bash  00_run_script_training_apply_surrogate_rp_simples_10000_ml_model.sh  | tee  oe_surrogate_rp_simples_10000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_ft_simples_10000_ml_model.sh  | tee  oe_surrogate_ft_simples_10000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_aaft_simples_10000_ml_model.sh  | tee  oe_surrogate_aaft_simples_10000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_iaaft1_simples_10000_ml_model.sh  | tee  oe_surrogate_iaaft1_simples_10000_ml_model.log  &&
bash  00_run_script_training_apply_surrogate_iaaft2_simples_10000_ml_model.sh  | tee  oe_surrogate_iaaft2_simples_10000_ml_model.log  &&

wait

echo end date:
date

