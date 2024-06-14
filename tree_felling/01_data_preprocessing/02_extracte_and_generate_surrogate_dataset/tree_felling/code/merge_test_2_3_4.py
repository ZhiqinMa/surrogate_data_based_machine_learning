"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge PII, EPIII and LPIII

# 9 parameters to set:  tsid_2=int, tsid_3=int, tsid_4=int,
                        ID_train_2=str, ID_train_3=str, ID_train_4=str,
                        surr_type=str, simples=int, simples=int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_2, tsid_3, tsid_4 = 2, 3, 4      ## (Set it up to your needs)
# ID_test_2 = 'PII'                     ## (Set it up to your needs)
# ID_test_3 = 'EPIII'                   ## (Set it up to your needs)
# ID_test_4 = 'LPIII'                   ## (Set it up to your needs)
# surr_type = 'AAFT'                    ## (Set it up to your needs)
# n_features = 100                      ## (Set it up to your needs)
# simples = 1000

# passing parameters
tsid_2 = int(sys.argv[1])               ## (Set it up to your needs)
tsid_3 = int(sys.argv[2])               ## (Set it up to your needs)
tsid_4 = int(sys.argv[3])               ## (Set it up to your needs)
ID_test_2 = str(sys.argv[4])            ## (Set it up to your needs)
ID_test_3 = str(sys.argv[5])            ## (Set it up to your needs)
ID_test_4 = str(sys.argv[6])            ## (Set it up to your needs)
surr_type = str(sys.argv[7])            ## (Set it up to your needs)
n_features = int(sys.argv[8])           ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[9])              ## (Set it up to your needs)

print('tsid_2:', tsid_2)
print('tsid_3:', tsid_3)
print('tsid_4:', tsid_4)
print('ID_test_2:', ID_test_2)
print('ID_test_3:', ID_test_3)
print('ID_test_4:', ID_test_4)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path_2 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_2, surr_type, simples, ID_test_2)
path_3 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_3, surr_type, simples, ID_test_3)
path_4 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_4, surr_type, simples, ID_test_4)
# load dataset
df2 = read_csv(path_2, header=None)
df3 = read_csv(path_3, header=None)
df4 = read_csv(path_4, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df2.shape: ", df2.shape)
print("df3.shape: ", df3.shape)
print("df4.shape: ", df4.shape)
print("---------------------")

# create 0 dataframe
df2_0 = pd.DataFrame(data=0.0, index=range(df2.shape[0]), columns=range(n_features - df2.shape[1]))
df3_0 = pd.DataFrame(data=0.0, index=range(df3.shape[0]), columns=range(n_features - df3.shape[1]))
df4_0 = pd.DataFrame(data=0.0, index=range(df4.shape[0]), columns=range(n_features - df4.shape[1]))
# review shape
# print(df2_0)

# concat dataframe axis=1
df2_concat = concat([df2_0, df2], axis=1, ignore_index=True)
df3_concat = concat([df3_0, df3], axis=1, ignore_index=True)
df4_concat = concat([df4_0, df4], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df2_concat.shape: ", df2_concat.shape)
print("df3_concat.shape: ", df3_concat.shape)
print("df4_concat.shape: ", df4_concat.shape)
print("---------------------")

# concat dataframe axis=0
df = concat([df2_concat, df3_concat, df4_concat], axis=0, ignore_index=True)
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_test_{}_{}_{}.csv".format(surr_type, simples, ID_test_2, ID_test_3, ID_test_4),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
