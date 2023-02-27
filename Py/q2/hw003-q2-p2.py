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
from sklearn.neighbors import KNeighborsClassifier as KNN
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import os

# N neighbors
num_neigh = 1

# Options
q2_2 = False

# File options
file = 'emails.csv'

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
df=pd.read_csv(file)

# taking off the useless column counters
for col_name in list(df):
    if 'No.' in col_name:
        df.pop(col_name)

# Seperating test and train data
# train_df = df[:4000]
# test_df = df[4000:]

# Creating the folds
fold_num = 5
folds = {}
splits = np.arange(0,len(df),len(df)/fold_num, dtype=int)
for i, slice in enumerate(splits):
    cur_fold_num = i+1
    if i == 0:
        test_df = df[:slice+splits[1]]
        train_df = df[slice+splits[1]:]
    elif i == len(splits):
        test_df = df[slice:]
        train_df = df[:slice]
    else:
        test_df = df[slice:slice+splits[1]]
        train_df = df.iloc[np.r_[0:slice, slice+splits[1]:len(df)]]
    folds[cur_fold_num] = {'test_set': test_df, 'train_set': train_df}
    # print('test: ',len(folds[cur_fold_num]['test_set']),'train: ',len(folds[cur_fold_num]['train_set']))

fold_analysis = {}
for fold, data in folds.items():
    # Creating classifier for the fold
    knn = KNN(n_neighbors=num_neigh)

    # Training the fold
    knn.fit(data['train_set'].drop('Prediction', axis=1), data['train_set']['Prediction'])

    # Predicting the fold
    pred = knn.predict(data['test_set'].drop('Prediction', axis=1))

    # finding the information
    acc = metrics.accuracy_score(data['test_set']['Prediction'], pred)
    prec = metrics.precision_score(data['test_set']['Prediction'], pred)
    recall = metrics.recall_score(data['test_set']['Prediction'], pred)
    fold_analysis[fold] = {'Fold': fold, 'Accuracy': acc, 'Precision': prec, 'Recall': recall}

# Printing the answers for each fold for question 2.2
if q2_2:
    for key, val in  fold_analysis.items():
        print(val)