"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge BM_III and PI to predict PII, EPIII and LPIII

# 5 parameters to set:  tsid_S3=int,
                        ID_train_S3=str,
                        surr_type=str,
                        n_features=int,
                        simples = int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_S3 = 3                       ## (Set it up to your needs)
# ID_train_S3 = 'S3'                ## (Set it up to your needs)
# surr_type = 'AAFT'                ## (Set it up to your needs)
# n_features = 450                  ## (Set it up to your needs)
# simples = 1000                    ## (Set it up to your needs)

# passing parameters
tsid_S3 = int(sys.argv[1])          ## (Set it up to your needs)
ID_train_S3 = str(sys.argv[2])      ## (Set it up to your needs)
surr_type = str(sys.argv[3])        ## (Set it up to your needs)
n_features = int(sys.argv[4])       ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[5])          ## (Set it up to your needs)

print('tsid_S3:', tsid_S3)
print('ID_train_S3:', ID_train_S3)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path_4 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S3, surr_type, simples, ID_train_S3)
# load dataset
df4 = read_csv(path_4, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df4.shape: ", df4.shape)
print("---------------------")

# create 0 dataframe
df4_0 = pd.DataFrame(data=0.0, index=range(df4.shape[0]), columns=range(n_features - df4.shape[1]))
# review shape
# print(df4_0)

# concat dataframe axis=1
df4_concat = concat([df4_0, df4], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df4_concat.shape: ", df4_concat.shape)
print("---------------------")

# concat dataframe axis=0
# df = concat([df4_concat], axis=0, ignore_index=True)
df = df4_concat
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_train_{}.csv".format(surr_type, simples, ID_train_S3),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
