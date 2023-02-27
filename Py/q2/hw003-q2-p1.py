"""
Created by Matt Myers
Due 02/27/20223
Hw 003

Description:

1. (10 pts) Use the whole D2z.txt as training set. Use Euclidean distance (i.e. A = I). Visualize the predictions
of 1NN on a 2D grid [-2 : 0.1 : 2]^2. That is, you should produce test points whose first feature goes over
-2, -1.9, -1.8, . . . , 1.9, 2, so does the second feature independent of the first feature. You should overlay
the training set in the plot, just make sure we can tell which points are training, which are grid.
The expected figure looks like this.

"""
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import pandas as pd
import os

# Figure display options
plot_show = True

# File options
file = 'D2z.txt'

# KNN information
num_neigh = 1

# read in info
path_to_data = r'\Data'
os.chdir(os.getcwd()+path_to_data)
df=(pd.read_csv(file, sep=' ', names=["x_n1","x_n2","y_n"], index_col=False))

# Setting up the training data
x_train = df.loc[:, df.columns != 'y_n']
y_train = df.loc[:, df.columns == 'y_n']

# Setting up the test data from the predictions
twodGrid = np.arange(-2,2,0.1)
prediction_grid = [[x_n1,x_n2] for x_n1 in twodGrid for x_n2 in  twodGrid]

# Setting up the KNN
knn = KNN(n_neighbors=num_neigh)
knn.fit(x_train, y_train)
preds = knn.predict(prediction_grid)
print(np.unique(preds))

# Plotting the predictions
pred_x_n1 = [elem[0] for elem in prediction_grid]
pred_x_n2 = [elem[1] for elem in prediction_grid]
x_pred = pd.DataFrame({'x_n1': pred_x_n1, 'x_n2': pred_x_n2})
pred_colors = ['blue', 'red']
for i, y_val in enumerate(np.unique(preds)):
    plt.scatter(x_pred.loc[preds==y_val, 'x_n1'],
                x_pred.loc[preds==y_val, 'x_n2'],
                c=pred_colors[i], s=2.5, label=f'Pred {i}')
    # plt.scatter(pred_x_n1, pred_x_n2, c=preds, cmap=pred_colors, s=2.5, label='pred')

# Scatter plot for training data
train_markers = ['o', '+']
colors = ['None','black']
for i, y_val in enumerate(y_train['y_n'].unique()):
    plt.scatter(x_train.loc[y_train['y_n']==y_val, 'x_n1'],
                x_train.loc[y_train['y_n']==y_val, 'x_n2'],
                c=colors[i], marker=train_markers[i], edgecolors='black' if train_markers[i] != '+' else 'None', linewidths=1, s=30, label=f'Train {i}')
plt.title("D2z.txt KNN")
plt.xlabel('x_n1')
plt.ylabel('x_n2')
plt.legend()

# Deciding whether to show plot or save it
if plot_show:
    plt.show()
else:
    fig_path = r'D:\Documents\UW-Madison\CourseWork\Spring\MachineLearning\Hw\003\figs\q2\q2-p1\\'
    fig_title = 'D2z.txt_KNN.png'

    plt.savefig(fig_path+fig_title, dpi = 300)