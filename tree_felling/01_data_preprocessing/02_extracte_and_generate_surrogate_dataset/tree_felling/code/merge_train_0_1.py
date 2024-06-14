"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge BM_III and PI to predict PII, EPIII and LPIII

# 7 parameters to set:  tsid_0=int, tsid_1=int,
                        ID_train_0=str, ID_train_1=str,
                        surr_type=str, simples=int, simples=int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_0, tsid_1 = 0, 1
# ID_train_0 = 'BMIII'    ## (Set it up to your needs)
# ID_train_1 = 'PI'       ## (Set it up to your needs)
# surr_type = 'AAFT'      ## (Set it up to your needs)
# n_features = 100        ## (Set it up to your needs)
# simples = 1000

# passing parameters
tsid_0, tsid_1 = int(sys.argv[1]), int(sys.argv[2])  ## (Set it up to your needs)
ID_train_0 = str(sys.argv[3])  ## (Set it up to your needs)
ID_train_1 = str(sys.argv[4])  ## (Set it up to your needs)
surr_type = str(sys.argv[5])  ## (Set it up to your needs)
n_features = int(sys.argv[6])  ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[7])  ## (Set it up to your needs)

print('tsid_0:', tsid_0)
print('tsid_1:', tsid_1)
print('ID_train_0:', ID_train_0)
print('ID_train_1:', ID_train_1)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path_0 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_0, surr_type, simples, ID_train_0)
path_1 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_1, surr_type, simples, ID_train_1)
# load dataset
df0 = read_csv(path_0, header=None)
df1 = read_csv(path_1, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df0.shape: ", df0.shape)
print("df1.shape:", df1.shape)
print("---------------------")

# create 0 dataframe
df0_0 = pd.DataFrame(data=0.0, index=range(df0.shape[0]), columns=range(n_features - df0.shape[1]))
df1_0 = pd.DataFrame(data=0.0, index=range(df1.shape[0]), columns=range(n_features - df1.shape[1]))
# review shape
# print(df4_0)

# concat dataframe axis=1
df0_concat = concat([df0_0, df0], axis=1, ignore_index=True)
df1_concat = concat([df1_0, df1], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df4_concat.shape: ", df0_concat.shape)
print("df3_concat.shape: ", df1_concat.shape)
print("---------------------")

# concat dataframe axis=0
df = concat([df0_concat, df1_concat], axis=0, ignore_index=True)
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_train_{}_{}.csv".format(surr_type, simples, ID_train_0, ID_train_1),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
