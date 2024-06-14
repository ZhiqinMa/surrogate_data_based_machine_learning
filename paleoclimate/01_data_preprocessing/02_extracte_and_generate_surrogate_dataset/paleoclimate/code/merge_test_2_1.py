"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge BM_III and PI to predict PII, EPIII and LPIII

# 7 parameters to set:  tsid_4=int, tsid_3=int,
                        ID_train_4=str, ID_train_3=str, surr_type=str,
                        n_features=int, simples=int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_2, tsid_1 = 2, 1                                 ## (Set it up to your needs)
# ID_test_2 = 'II'                                      ## (Set it up to your needs)
# ID_test_1 = 'I'                                       ## (Set it up to your needs)
# surr_type = 'AAFT'                                    ## (Set it up to your needs)
# n_features = 500 + 1                                  ## (Set it up to your needs)
# simples = 1000                                        ## (Set it up to your needs)

# passing parameters
tsid_2, tsid_1 = int(sys.argv[1]), int(sys.argv[2])     ## (Set it up to your needs)
ID_test_2 = str(sys.argv[3])                            ## (Set it up to your needs)
ID_test_1 = str(sys.argv[4])                            ## (Set it up to your needs)
surr_type = str(sys.argv[5])                            ## (Set it up to your needs)
n_features = int(sys.argv[6])                           ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[7])                              ## (Set it up to your needs)

print('tsid_2:', tsid_2)
print('tsid_1:', tsid_1)
print('ID_test_2:', ID_test_2)
print('ID_test_1:', ID_test_1)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path_2 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_2, surr_type, simples, ID_test_2)
path_1 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_1, surr_type, simples, ID_test_1)
# load dataset
df2 = read_csv(path_2, header=None)
df1 = read_csv(path_1, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df2.shape: ", df2.shape)
print("df1.shape:", df1.shape)
print("---------------------")

# create 0 dataframe
df2_0 = pd.DataFrame(data=0.0, index=range(df2.shape[0]), columns=range(n_features - df2.shape[1]))
df1_0 = pd.DataFrame(data=0.0, index=range(df1.shape[0]), columns=range(n_features - df1.shape[1]))
# review shape
# print(df2_0)

# concat dataframe axis=1
df2_concat = concat([df2_0, df2], axis=1, ignore_index=True)
df1_concat = concat([df1_0, df1], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df2_concat.shape: ", df2_concat.shape)
print("df1_concat.shape: ", df1_concat.shape)
print("---------------------")

# concat dataframe axis=0
df = concat([df2_concat, df1_concat], axis=0, ignore_index=True)
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type,
                                                                         simples,
                                                                         ID_test_2,
                                                                         ID_test_1),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
