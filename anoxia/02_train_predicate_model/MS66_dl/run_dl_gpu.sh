#!/bin/bash
#PBS -q GPU
#PBS -l host=gpu02
#PBS -o 0205.out
#PBS -e 0206.err
#PBS -N MS66_DL_gpu

echo '--------------------------------'
# Go to the directory where the Python script is located
cd $PBS_O_WORKDIR

echo review path:
echo $PWD

# source some shell environment settings
source ~/.bashrc
conda activate surr_ews

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

# Run the Python script on gpu
export CUDA_VISIBLE_DEVICES=0
bash  00_run_script_training_apply_surrogate_rp_simples_1000_dl_model.sh  | tee  oe_surrogate_rp_simples_1000_dl_model.log
bash  00_run_script_training_apply_surrogate_ft_simples_1000_dl_model.sh  | tee  oe_surrogate_ft_simples_1000_dl_model.log
bash  00_run_script_training_apply_surrogate_aaft_simples_1000_dl_model.sh  | tee  oe_surrogate_aaft_simples_1000_dl_model.log
bash  00_run_script_training_apply_surrogate_iaaft1_simples_1000_dl_model.sh  | tee  oe_surrogate_iaaft1_simples_1000_dl_model.log
bash  00_run_script_training_apply_surrogate_iaaft2_simples_1000_dl_model.sh  | tee  oe_surrogate_iaaft2_simples_1000_dl_model.log

bash  00_run_script_training_apply_surrogate_rp_simples_10000_dl_model.sh  | tee  oe_surrogate_rp_simples_10000_dl_model.log
bash  00_run_script_training_apply_surrogate_ft_simples_10000_dl_model.sh  | tee  oe_surrogate_ft_simples_10000_dl_model.log
bash  00_run_script_training_apply_surrogate_aaft_simples_10000_dl_model.sh  | tee  oe_surrogate_aaft_simples_10000_dl_model.log
bash  00_run_script_training_apply_surrogate_iaaft1_simples_10000_dl_model.sh  | tee  oe_surrogate_iaaft1_simples_10000_dl_model.log
bash  00_run_script_training_apply_surrogate_iaaft2_simples_10000_dl_model.sh  | tee  oe_surrogate_iaaft2_simples_10000_dl_model.log

wait

echo end date:
date

