#!/bin/bash
# Run commands by git bash

echo '---------------------------- 01 Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

# setting parameters
# Period=I
tsid=1
Period='I'
simples=1000
surr_type='FT'
n_features=300
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=II
tsid=2
Period='II'
# simples=1000
# surr_type='FT'
n_features=300
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=III
tsid=3
Period='III'
# simples=1000
# surr_type='FT'
n_features=300
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=IV
tsid=4
Period='IV'
# simples=1000
# surr_type='FT'
n_features=300
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&
echo '---------------------------- 01 complete end ----------------------------'
echo


echo '---------------------------- 02 Run merge_train ----------------------------'
tsid_4=4              ## (Set it up to your needs)
tsid_3=3              ## (Set it up to your needs)
ID_train_4='IV'       ## (Set it up to your needs)
ID_train_3='III'      ## (Set it up to your needs)
surr_type='FT'      ## (Set it up to your needs)
n_features=300        ## (Set it up to your needs)
# simples=1000
python merge_train_4_3.py $tsid_4 $tsid_3 $ID_train_4 $ID_train_3 $surr_type $n_features $simples &&


tsid_2=2              ## (Set it up to your needs)
tsid_1=1              ## (Set it up to your needs)
ID_test_2='II'        ## (Set it up to your needs)
ID_test_1='I'         ## (Set it up to your needs)
surr_type='FT'      ## (Set it up to your needs)
n_features=300        ## (Set it up to your needs)
# simples=1000
python merge_test_2_1.py $tsid_2 $tsid_1 $ID_test_2 $ID_test_1 $surr_type $n_features $simples &&
echo '---------------------------- 02 complete end ----------------------------'