#!/bin/bash
# Run commands by git bash

echo '---------------------------- 01 Run start python scripts -----------------------------'
# After the first one runs successfully, only then run the next one (Serial run)

# setting parameters
tsid=8
simples=10000
surr_type='FT'
# python generate_surrogate.py $tsid $simples $surr_type &&
python generate_surrogate.py --tsid $tsid --simples $simples --surr_type $surr_type &&

tsid=14
# simples=10000
# surr_type='FT'
# python generate_surrogate.py $tsid $simples $surr_type &&
python generate_surrogate.py --tsid $tsid --simples $simples --surr_type $surr_type &&
echo '---------------------------- 01 complete end ----------------------------'
echo

echo '---------------------------- 02 Run merge_train ----------------------------'
tsid_merge=8            ## (Set it up to your needs)
# surr_type='FT'  
n_features=150          ## (Set it up to your needs)
# simples = 10000
python merge_train_8.py $tsid_merge $surr_type $n_features $simples  &&


tsid_merge=14           ## (Set it up to your needs)
# surr_type='FT'
n_features=150          ## (Set it up to your needs)
# simples = 10000
python merge_test_14.py $tsid_merge $surr_type $n_features $simples  &&
echo '---------------------------- 02 complete end ----------------------------'