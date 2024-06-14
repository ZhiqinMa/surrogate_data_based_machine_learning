#!/bin/bash
# Run commands by git bash

echo '---------------------------- 01 Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

# setting parameters
# Period=MS66_S1
tsid=2
Period='S1'
simples=10000
surr_type='IAAFT2'
Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=MS66_S3
tsid=4
Period='S3'
# simples=10000
# surr_type='IAAFT2'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=MS66_S4
tsid=6
Period='S4'
# simples=10000
# surr_type='IAAFT2'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=MS66_S5
tsid=8
Period='S5'
# simples=10000
# surr_type='IAAFT2'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&
echo '---------------------------- 01 complete end ----------------------------'
echo

echo '---------------------------- 02 Run merge_train ----------------------------'
tsid_S5=8                   ## (Set it up to your needs)
tsid_S4=6                   ## (Set it up to your needs)
ID_train_S5='S5'            ## (Set it up to your needs)
ID_train_S4='S4'            ## (Set it up to your needs)
# surr_type='IAAFT2'        ## (Set it up to your needs)
n_features=220              ## (Set it up to your needs)
# simples=10000
python merge_train_S5_S4.py $tsid_S5 $tsid_S4 $ID_train_S5 $ID_train_S4 $surr_type $n_features $simples &&


tsid_S3=4                   ## (Set it up to your needs)
tsid_S1=2                   ## (Set it up to your needs)
ID_test_S3='S3'             ## (Set it up to your needs)
ID_test_S1='S1'             ## (Set it up to your needs)
# surr_type='IAAFT2'        ## (Set it up to your needs)
n_features=220              ## (Set it up to your needs)
# simples=10000
python merge_test_S3_S1.py $tsid_S3 $tsid_S1 $ID_test_S3 $ID_test_S1 $surr_type $n_features $simples &&
echo '---------------------------- 02 complete end ----------------------------'