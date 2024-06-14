"""
Created on Wed Mar 20 23:04:16 2023

Use MatLab toolbox for many of the surrogate methods as given in Gemma Lancaster 2018 (Physics Reports)
Paper URL: https://www.sciencedirect.com/science/article/pii/S0370157318301340?via%3Dihub
MatLab toolbox URL: http://py-biomedical.lancs.ac.uk/

Function: Extraction of pre-critical transition or non-critical transition features from real data sets

# merge 8 to predict 14

# 4 parameters to set:  tsid_merge=int,
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
# tsid_merge = 8         ## (Set it up to your needs)
# surr_type = 'AAFT'     ## (Set it up to your needs)
# n_features = 150       ## (Set it up to your needs)
# simples = 1000         ## (Set it up to your needs)

# passing parameters
tsid_merge = int(sys.argv[1])   ## (Set it up to your needs)
surr_type = str(sys.argv[2])    ## (Set it up to your needs)
n_features = int(sys.argv[3])   ## (Set it up to your needs)
n_features = n_features + 1
simples = int(sys.argv[4])      ## (Set it up to your needs)

print('tsid_merge:', tsid_merge)
print('surr_type:', surr_type)
print('n_features:', n_features)
print('simples:', simples)
print()

# Path
path = "../data/0{}_surrogate_{}_simples_{}_period_{}.csv".format(tsid_merge, surr_type, simples, tsid_merge)
# load dataset
df = read_csv(path, header=None)
# review
print("--------------- start ---------------")
print("raw dataset shape:")
print("df.shape: ", df.shape)
print("---------------------")

# create 0 dataframe
df0 = pd.DataFrame(data=0.0, index=range(df.shape[0]), columns=range(n_features - df.shape[1]))
# review shape
# print(df0)

# concat dataframe axis=1
df_concat = concat([df0, df], axis=1, ignore_index=True)
# review shape
print("concat dataframe: axis=1(col):")
print("df_concat.shape: ", df_concat.shape)
print("---------------------")

# concat dataframe axis=0
# df = concat([df4_concat], axis=0, ignore_index=True)
df_merge = df_concat
# revire shape
print("concat dataframe axis=0(row):")
print("df_merge.shape: ", df_merge.shape)

# save dataframe file
df_merge.to_csv("../data/merge_surrogate_{}_simples_{}_train_{}.csv".format(surr_type, simples, tsid_merge), sep=',', header=False, index=False)

print('\n'"------------------------------ Successful ------------------------------")
print("------------------------------ Completed ------------------------------")
