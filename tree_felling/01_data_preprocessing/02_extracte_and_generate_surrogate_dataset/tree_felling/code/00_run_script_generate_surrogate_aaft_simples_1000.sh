#!/bin/bash
# Run commands by git bash

echo '---------------------------- 01 Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

# setting parameters
# Period=BMIII
tsid=0
Period='BMIII'
simples=1000
surr_type='AAFT'
n_features=100
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=PI
tsid=1
Period='PI'
# simples=1000
# surr_type='AAFT'
# n_features=100
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=PII
tsid=2
Period='PII'
# simples=1000
# surr_type='AAFT'
# n_features=100
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=EPIII
tsid=3
Period='EPIII'
# simples=1000
# surr_type='AAFT'
# n_features=100
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&

# Period=LPIII
tsid=4
Period='LPIII'
# simples=1000
# surr_type='AAFT'
# n_features=100
python generate_surrogate.py $tsid $Period $simples $surr_type $n_features &&
echo '---------------------------- 01 complete end ----------------------------'
echo

echo '---------------------------- 02 Run start merge ----------------------------'
tsid_0=0                ## (Set it up to your needs)
tsid_1=1                ## (Set it up to your needs)
ID_train_0='BMIII'      ## (Set it up to your needs)
ID_train_1='PI'         ## (Set it up to your needs)
# surr_type='AAFT'      ## (Set it up to your needs)
n_features=100          ## (Set it up to your needs)
# simples=1000          ## (Set it up to your needs)
python merge_train_0_1.py $tsid_0 $tsid_1 $ID_train_0 $ID_train_1 $surr_type $n_features $simples &&


tsid_2=2                ## (Set it up to your needs)
tsid_3=3                ## (Set it up to your needs)
tsid_4=4                ## (Set it up to your needs)
ID_test_2='PII'         ## (Set it up to your needs)
ID_test_3='EPIII'       ## (Set it up to your needs)
ID_test_4='LPIII'       ## (Set it up to your needs)
# surr_type='AAFT'      ## (Set it up to your needs)
n_features=100          ## (Set it up to your needs)
# simples=1000          ## (Set it up to your needs)
python merge_test_2_3_4.py $tsid_2 $tsid_3 $tsid_4 $ID_test_2 $ID_test_3 $ID_test_4 $surr_type $n_features $simples &&
echo '---------------------------- 02 complete merge end ----------------------------'