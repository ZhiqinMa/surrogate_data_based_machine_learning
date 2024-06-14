"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge BM_III and PI to predict PII, EPIII and LPIII

# 9 parameters to set: tsid_S9=int, tsid_S8=int, tsid_S7=int,
                    ID_train_S9=str, ID_train_S8=str, ID_train_S7=str,
                    surr_type=str, n_features=int, simples = int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_S9 = 13                      ## (Set it up to your needs)
# tsid_S8 = 12                      ## (Set it up to your needs)
# tsid_S7 = 11                      ## (Set it up to your needs)
# ID_train_S9 = 'S9'                ## (Set it up to your needs)
# ID_train_S8 = 'S8'                ## (Set it up to your needs)
# ID_train_S7 = 'S7'                ## (Set it up to your needs)
# surr_type = 'IAAFT1'              ## (Set it up to your needs)
# n_features = 220                  ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)

# passing parameters
tsid_S9 = int(sys.argv[1])          ## (Set it up to your needs)
tsid_S8 = int(sys.argv[2])          ## (Set it up to your needs)
tsid_S7 = int(sys.argv[3])          ## (Set it up to your needs)
ID_train_S9 = str(sys.argv[4])      ## (Set it up to your needs)
ID_train_S8 = str(sys.argv[5])      ## (Set it up to your needs)
ID_train_S7 = str(sys.argv[6])      ## (Set it up to your needs)
surr_type = str(sys.argv[7])        ## (Set it up to your needs)
n_features = int(sys.argv[8])       ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[9])          ## (Set it up to your needs)

print('tsid_S9:', tsid_S9)
print('tsid_S8:', tsid_S8)
print('tsid_S7:', tsid_S7)
print('ID_train_S9:', ID_train_S9)
print('ID_train_S8:', ID_train_S8)
print('ID_train_S7:', ID_train_S7)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path_9 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S9, surr_type, simples, ID_train_S9)
path_8 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S8, surr_type, simples, ID_train_S8)
path_7 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S7, surr_type, simples, ID_train_S7)
# load dataset
df9 = read_csv(path_9, header=None)
df8 = read_csv(path_8, header=None)
df7 = read_csv(path_7, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df9.shape: ", df9.shape)
print("df8.shape: ", df8.shape)
print("df7.shape: ", df7.shape)
print("---------------------")

# create 0 dataframe
df9_0 = pd.DataFrame(data=0.0, index=range(df9.shape[0]), columns=range(n_features - df9.shape[1]))
df8_0 = pd.DataFrame(data=0.0, index=range(df8.shape[0]), columns=range(n_features - df8.shape[1]))
df7_0 = pd.DataFrame(data=0.0, index=range(df7.shape[0]), columns=range(n_features - df7.shape[1]))
# review shape
# print(df9_0)

# concat dataframe axis=1
df9_concat = concat([df9_0, df9], axis=1, ignore_index=True)
df8_concat = concat([df8_0, df8], axis=1, ignore_index=True)
df7_concat = concat([df7_0, df7], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df9_concat.shape: ", df9_concat.shape)
print("df8_concat.shape: ", df8_concat.shape)
print("df7_concat.shape: ", df7_concat.shape)
print("---------------------")

# concat dataframe axis=0
df = concat([df9_concat, df8_concat, df7_concat], axis=0, ignore_index=True)
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_train_{}_{}_{}.csv".format(surr_type,
                                                                               simples,
                                                                               ID_train_S9,
                                                                               ID_train_S8,
                                                                               ID_train_S7,
                                                                            ),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
