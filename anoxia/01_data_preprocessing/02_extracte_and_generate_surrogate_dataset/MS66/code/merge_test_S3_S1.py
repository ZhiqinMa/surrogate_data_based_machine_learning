"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge BM_III and PI to predict PII, EPIII and LPIII

# 7 parameters to set:  tsid_S3=int, tsid_S1=int,
                        ID_test_S3=str, ID_test_S1=str, surr_type=str,
                        n_features=int, simples = int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_S3 = 4                           ## (Set it up to your needs)
# tsid_S1 = 2                           ## (Set it up to your needs)
# ID_test_S3 = 'S3'                     ## (Set it up to your needs)
# ID_test_S1 = 'S1'                     ## (Set it up to your needs)
# surr_type = 'IAAFT1'                  ## (Set it up to your needs)
# n_features = 220                      ## (Set it up to your needs)
# simples = 1000                        ## (Set it up to your needs)

# passing parameters
tsid_S3 = int(sys.argv[1])              ## (Set it up to your needs)
tsid_S1 = int(sys.argv[2])              ## (Set it up to your needs)
ID_test_S3 = str(sys.argv[3])           ## (Set it up to your needs)
ID_test_S1 = str(sys.argv[4])           ## (Set it up to your needs)
surr_type = str(sys.argv[5])            ## (Set it up to your needs)
n_features = int(sys.argv[6])           ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[7])              ## (Set it up to your needs)

print('tsid_S3:', tsid_S3)
print('tsid_S1:', tsid_S1)
print('ID_test_S3:', ID_test_S3)
print('ID_test_S1:', ID_test_S1)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print('\n')

# Path
path_3 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S3, surr_type, simples, ID_test_S3)
path_1 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S1, surr_type, simples, ID_test_S1)
# load dataset
df3 = read_csv(path_3, header=None)
df1 = read_csv(path_1, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df3.shape: ", df3.shape)
print("df1.shape: ", df1.shape)
print("---------------------")

# create 0 dataframe
df3_0 = pd.DataFrame(data=0.0, index=range(df3.shape[0]), columns=range(n_features - df3.shape[1]))
df1_0 = pd.DataFrame(data=0.0, index=range(df1.shape[0]), columns=range(n_features - df1.shape[1]))
# review shape
# print(df3_0)

# concat dataframe axis=1
df3_concat = concat([df3_0, df3], axis=1, ignore_index=True)
df1_concat = concat([df1_0, df1], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df3_concat.shape: ", df3_concat.shape)
print("df1_concat.shape: ", df1_concat.shape)
print("---------------------")

# concat dataframe axis=0
df = concat([df3_concat, df1_concat], axis=0, ignore_index=True)
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_test_{}_{}.csv".format(surr_type, simples, ID_test_S3, ID_test_S1),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
