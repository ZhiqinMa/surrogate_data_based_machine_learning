#!/bin/bash
# Run commands by git bash

echo '---------------------------- 01 Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

# setting parameters
# Period=64PE_S3
tsid=5
Period='S3'
simples=1000
surr_type='AAFT'
Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=64PE_S4
tsid=7
Period='S4'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=64PE_S5
tsid=9
Period='S5'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=64PE_S6
tsid=10
Period='S6'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=64PE_S7
tsid=11
Period='S7'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

# Period=64PE_S8
tsid=12
Period='S8'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&


# Period=64PE_S9
tsid=13
Period='S9'
# simples=1000
# surr_type='AAFT'
# Variable_label='Mo'
python generate_surrogate.py $tsid $Period $simples $surr_type $Variable_label &&

echo '---------------------------- 01 complete end ----------------------------'
echo


echo '---------------------------- 02 Run merge_train ----------------------------'
tsid_S9=13                          ## (Set it up to your needs)
tsid_S8=12                          ## (Set it up to your needs)
tsid_S7=11                          ## (Set it up to your needs)
ID_train_S9='S9'                    ## (Set it up to your needs)
ID_train_S8='S8'                    ## (Set it up to your needs)
ID_train_S7='S7'                    ## (Set it up to your needs)
# surr_type='AAFT'                  ## (Set it up to your needs)
n_features=220                      ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)
python merge_train_S9_S8_S7.py $tsid_S9 $tsid_S8 $tsid_S7 $ID_train_S9 $ID_train_S8 $ID_train_S7 $surr_type $n_features $simples &&


tsid_S6=10                          ## (Set it up to your needs)
tsid_S5=9                           ## (Set it up to your needs)
tsid_S4=7                           ## (Set it up to your needs)
tsid_S3=5                           ## (Set it up to your needs)
ID_test_S6='S6'                     ## (Set it up to your needs)
ID_test_S5='S5'                     ## (Set it up to your needs)
ID_test_S4='S4'                     ## (Set it up to your needs)
ID_test_S3='S3'                     ## (Set it up to your needs)
# surr_type='AAFT'                  ## (Set it up to your needs)
n_features=220                      ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)
python merge_test_S6_S5_S4_S3.py $tsid_S6 $tsid_S5 $tsid_S4 $tsid_S3 $ID_test_S6 $ID_test_S5 $ID_test_S4 $ID_test_S3 $surr_type $n_features $simples &&
echo '---------------------------- 02 complete end ----------------------------'