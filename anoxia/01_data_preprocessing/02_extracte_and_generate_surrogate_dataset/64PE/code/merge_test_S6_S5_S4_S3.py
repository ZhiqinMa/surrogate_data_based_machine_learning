"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge BM_III and PI to predict PII, EPIII and LPIII

# 11 parameters to set: tsid_S6=int, tsid_S5=int, tsid_S4=int, tsid_S3=int,
                    ID_train_S6=str, ID_train_S5=str, ID_train_S4=str, ID_train_S3=str,
                    surr_type=str, n_features=int, simples = int

@author: Zhiqin Ma
"""

import sys
import pandas as pd
from pandas import read_csv
from pandas import concat

# # Setting parameters
# tsid_S6 = 10                          ## (Set it up to your needs)
# tsid_S5 = 9                           ## (Set it up to your needs)
# tsid_S4 = 7                           ## (Set it up to your needs)
# tsid_S3 = 5                           ## (Set it up to your needs)
# ID_test_S6 = 'S6'                     ## (Set it up to your needs)
# ID_test_S5 = 'S5'                     ## (Set it up to your needs)
# ID_test_S4 = 'S4'                     ## (Set it up to your needs)
# ID_test_S3 = 'S3'                     ## (Set it up to your needs)
# surr_type = 'AAFT'                    ## (Set it up to your needs)
# n_features = 220                      ## (Set it up to your needs)
# simples = 1000                        ## (Set it up to your needs)

# passing parameters
tsid_S6 = int(sys.argv[1])              ## (Set it up to your needs)
tsid_S5 = int(sys.argv[2])              ## (Set it up to your needs)
tsid_S4 = int(sys.argv[3])              ## (Set it up to your needs)
tsid_S3 = int(sys.argv[4])              ## (Set it up to your needs)
ID_test_S6 = str(sys.argv[5])           ## (Set it up to your needs)
ID_test_S5 = str(sys.argv[6])           ## (Set it up to your needs)
ID_test_S4 = str(sys.argv[7])           ## (Set it up to your needs)
ID_test_S3 = str(sys.argv[8])           ## (Set it up to your needs)
surr_type = str(sys.argv[9])            ## (Set it up to your needs)
n_features = int(sys.argv[10])          ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[11])             ## (Set it up to your needs)

print('tsid_S6:', tsid_S6)
print('tsid_S5:', tsid_S5)
print('tsid_S4:', tsid_S4)
print('tsid_S3:', tsid_S3)
print('ID_test_S6:', ID_test_S6)
print('ID_test_S5:', ID_test_S5)
print('ID_test_S4:', ID_test_S4)
print('ID_test_S3:', ID_test_S3)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path_6 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S6, surr_type, simples, ID_test_S6)
path_5 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S5, surr_type, simples, ID_test_S5)
path_4 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S4, surr_type, simples, ID_test_S4)
path_3 = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_S3, surr_type, simples, ID_test_S3)
# load dataset
df6 = read_csv(path_6, header=None)
df5 = read_csv(path_5, header=None)
df4 = read_csv(path_4, header=None)
df3 = read_csv(path_3, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df6.shape: ", df6.shape)
print("df5.shape: ", df5.shape)
print("df4.shape: ", df4.shape)
print("df3.shape: ", df3.shape)
print("---------------------")

# create 0 dataframe
df6_0 = pd.DataFrame(data=0.0, index=range(df6.shape[0]), columns=range(n_features - df6.shape[1]))
df5_0 = pd.DataFrame(data=0.0, index=range(df5.shape[0]), columns=range(n_features - df5.shape[1]))
df4_0 = pd.DataFrame(data=0.0, index=range(df4.shape[0]), columns=range(n_features - df4.shape[1]))
df3_0 = pd.DataFrame(data=0.0, index=range(df3.shape[0]), columns=range(n_features - df3.shape[1]))
# review shape
# print(df6_0)

# concat dataframe axis=1
df6_concat = concat([df6_0, df6], axis=1, ignore_index=True)
df5_concat = concat([df5_0, df5], axis=1, ignore_index=True)
df4_concat = concat([df4_0, df4], axis=1, ignore_index=True)
df3_concat = concat([df3_0, df3], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df6_concat.shape: ", df6_concat.shape)
print("df5_concat.shape: ", df5_concat.shape)
print("df4_concat.shape: ", df4_concat.shape)
print("df3_concat.shape: ", df3_concat.shape)
print("---------------------")

# concat dataframe axis=0
df = concat([df6_concat, df5_concat, df4_concat, df3_concat], axis=0, ignore_index=True)
# revire shape
print("concat dataframe axis=0(row):")
print("df.shape: ", df.shape)

# save dataframe file
df.to_csv("../data/merge_surrogate_{}_simples_{}_test_{}_{}_{}_{}.csv".format(surr_type,
                                                                              simples,
                                                                              ID_test_S6,
                                                                              ID_test_S5,
                                                                              ID_test_S4,
                                                                              ID_test_S3
                                                                              ),
          sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
