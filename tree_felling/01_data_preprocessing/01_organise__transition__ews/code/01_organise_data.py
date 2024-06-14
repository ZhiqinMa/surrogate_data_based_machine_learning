"""
Created on Wed Mar 20 23:04:16 2023

Organise construction activity data in same format as in Marten Scheffer 2021
Use time ranges and transition times as given in Figures 2 and 3 Marten 2021

@author: Zhiqin Ma
"""

import os
import pandas as pd
import matplotlib.pyplot as plt



# Make export directory if doens't exist
try:
    os.mkdir('../data')
except:
    print('data directory already exists!')

try:
    os.mkdir('../data/01_time_series')
except:
    print('data/01_time_series directory already exists!')
print("\n")

# Replace 'your_file.xlsx' with the path to your Excel file
file_path = '../data/raw_tree_felling/BOCINSKY_ET_AL_2015_TREE_RINGS no lat long.xlsx'

# Read the Excel file
df = pd.read_excel(file_path, sheet_name='Sheet 1 - BOCINSKY_ET_AL_2015_T', header=1)
# print(df)

# Count the occurrences of each unique value in the 'Outer_Date_AD' column
value_counts = df['Outer_Date_AD'].value_counts().sort_index()

# Convert the Series to a DataFrame
counts_df = value_counts.reset_index()
counts_df.columns = ['Age', 'tree_felling']

# Filter the data to keep only the data with a time range between 500 and 1300 years
# 过滤数据，仅保留时间范围在 500 到 1300 年之间的数据
filtered_counts_df = counts_df[(counts_df['Age'] >= 500) & (counts_df['Age'] <= 1300)]

# Print the DataFrame
# print(filtered_counts_df)
filtered_counts_df.to_csv("../data/01_time_series/tree_felling_over_time.csv", index=False)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(counts_df['Age'], counts_df['tree_felling'])
plt.title('Construction Activity Over Time')
plt.xlabel('Time (yr AD)')
plt.ylabel('Construction Activity')
plt.xlim(500, 1300)  # Setting the x-axis limits
plt.grid(True)
plt.savefig('../data/01_time_series/tree_felling_over_time.png')
plt.show()

print('\n'"-------------------------------- Completed 01 read tree felling --------------------------------")