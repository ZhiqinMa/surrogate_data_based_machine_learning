#!/bin/bash
# Run commands by git bash

echo '---------------------------- 01 Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

# setting parameters
# Period=MS21_S1
tsid=1
Period='S1'
simples=1000
surr_type='AAFT'
Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=MS21_S3
tsid=3
Period='S3'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&
echo '---------------------------- 01 complete end ----------------------------'
echo

echo '---------------------------- 02 Run merge_train ----------------------------'
tsid_S3=3               ## (Set it up to your needs)
ID_train_S3='S3'        ## (Set it up to your needs)
# surr_type='AAFT'      ## (Set it up to your needs)
n_features=450          ## (Set it up to your needs)
# simples = 1000        ## (Set it up to your needs)
python merge_train_S3.py $tsid_S3 $ID_train_S3 $surr_type $n_features $simples  &&


tsid_S1=1               ## (Set it up to your needs)
ID_test_S1='S1'         ## (Set it up to your needs)
# surr_type='AAFT'      ## (Set it up to your needs)
n_features=450          ## (Set it up to your needs)
# simples = 1000        ## (Set it up to your needs)
python merge_test_S1.py $tsid_S1 $ID_test_S1 $surr_type $n_features $simples  &&
echo '---------------------------- 02 complete end ----------------------------'