"""
Created by Matt Myers
Due 02/27/20223
Hw 003

Description:

1. (10 pts) Use the whole D2z.txt as training set. Use Euclidean distance (i.e. A = I). Visualize the predictions
of 1NN on a 2D grid [-2 : 0.1 : 2]2. That is, you should produce test points whose first feature goes over
-2, -1.9, -1.8, . . . , 1.9, 2, so does the second feature independent of the first feature. You should overlay
the training set in the plot, just make sure we can tell which points are training, which are grid.
The expected figure looks like this.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Test

# File options
files = ['D2z.txt','emails.csv']

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
data = {}
for file in os.listdir(os.getcwd()):
    if file[0] == '.': continue
    if '.csv' in file:
        data[file]=(pd.read_csv(file))
    else:
        data[file]=(pd.read_csv(file, sep=',', names=["x_n1","x_n2","y_n"], index_col=False))

# Defining which data set to use for the code
df = data[files[1]]

# taking off the useless column counters
for col_name in list(df):
    if 'No.' in col_name:
        df.pop(col_name)

print(df)

# Organize the data
def sortData(D, Xi):
    return D.sort_values(by=Xi).reset_index(drop=True)