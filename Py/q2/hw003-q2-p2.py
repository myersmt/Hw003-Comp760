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

# options
q2_2 = False
q2_3 = False
q2_4 = False
q2_5 = True

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

def avg_vals(folds):
    acc_lis = []
    prec_lis = []
    recall_lis = []
    for key, val in folds.items():
        acc_lis.append(val['Accuracy'])
        prec_lis.append(val['Precision'])
        recall_lis.append(val['Recall'])
    return(np.mean(acc_lis), np.mean(prec_lis), np.mean(recall_lis))

def five_fold_crossvalidation(df, fold_num = 5, num_neigh = 1, learning_rate = None, num_iterations = 1000, roc = False):
    # Options
    print_vals = False
    # Creating the folds
    folds = {}
    if fold_num > 1:
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
    else:
        splits = np.arange(0,len(df),len(df)/5, dtype=int)
        test_df = df[splits[-1]:]
        train_df = df[:splits[-1]]
        #print(test_df, train_df)
        folds[fold_num] = {'test_set': test_df, 'train_set': train_df}
        # print('test: ',len(folds[cur_fold_num]['test_set']),'train: ',len(folds[cur_fold_num]['train_set']))

    fold_analysis = {}
    for fold, data in folds.items():
        # Creating classifier for the fold
        if learning_rate is None:
            if roc:
                knn = KNN(n_neighbors=num_neigh)
                # Training the fold
                knn.fit(data['train_set'].drop('Prediction', axis=1), data['train_set']['Prediction'])
                # Predicting the fold
                pred = knn.predict_proba(data['test_set'].drop('Prediction', axis=1))
                # Getting the ROC information
                roc_data = metrics.roc_curve(data['test_set']['Prediction'], pred[:,1])
                knn_acc = round(metrics.roc_auc_score(data['test_set']['Prediction'], pred[:,1]), 2)
            else:
                knn = KNN(n_neighbors=num_neigh)
                # Training the fold
                knn.fit(data['train_set'].drop('Prediction', axis=1), data['train_set']['Prediction'])
                # Predicting the fold
                pred = knn.predict(data['test_set'].drop('Prediction', axis=1))
        else:
            lin_reg = LinearRegression(learning_rate=learning_rate, num_iterations=num_iterations)
            # Training the fold
            x_train = data['train_set'].drop('Prediction', axis=1).values
            y_train = data['train_set']['Prediction'].values
            lin_reg.fit(x_train, y_train)
            # Predicting the fold
            x_test = data['test_set'].drop('Prediction', axis=1).values
            y_test = data['test_set']['Prediction'].values
            pred = lin_reg.predict(x_test)
            if roc:
                log_reg = metrics.roc_curve(data['test_set']['Prediction'], pred)
                log_acc = round(metrics.roc_auc_score(data['test_set']['Prediction'], pred), 2)

        # finding the information
        if roc:
            pass
        else:
            acc = metrics.accuracy_score(data['test_set']['Prediction'], pred)
            prec = metrics.precision_score(data['test_set']['Prediction'], pred)
            recall = metrics.recall_score(data['test_set']['Prediction'], pred)
            fold_analysis[fold] = {'Fold': fold, 'Accuracy': acc, 'Precision': prec, 'Recall': recall}

    # Finding the average values
    if not roc:
        avg_acc, _, _ = avg_vals(fold_analysis)

    # Printing the answers for each fold
    if print_vals:
        if learning_rate is None:
            print('Question 2.2:')
            for key, val in  fold_analysis.items():
                print(val)
        else:
            print(f'Question 2.3')
            print(f'Learning rate: {learning_rate}, Number of iterations: {num_iterations}')
            for key, val in  fold_analysis.items():
                print(val)
    if roc and learning_rate is None:
        return(roc_data, knn_acc)
    elif roc and learning_rate is not None:
        return(log_reg, log_acc)
    else:
        return(avg_acc)

# Question 2.2 answer
if q2_2:
    # N neighbors
    five_fold_crossvalidation(df)

# Creating a linear regression class
class LinearRegression():
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    # Sigmoid function
    def sigmoid(self, z):
        pos_mask = (z >= 0)
        neg_mask = (z < 0)
        result = np.zeros_like(z)
        result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        result[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
        return result

    # Loss function (cross-entropy)
    def loss(self, y_pred, y_true):
        return -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

    # Gradient of loss function
    def gradient(self, X, y, weights):
        y_pred = self.sigmoid(np.dot(X, weights))
        error = y_pred - y
        return np.dot(X.T, error) / len(y)

    # Gradient descent algorithm
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for i in range(self.num_iterations):
            grad = self.gradient(X, y, self.weights)
            self.weights -= self.learning_rate * grad

    # Predict function
    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights))
        # return [1 if p >= 0.5 else 0 for p in y_pred]
        return y_pred



# Question 2.3 answer
if q2_3:
    print(five_fold_crossvalidation(df, learning_rate= 0.1))

# Question 2.4 answer
if q2_4:
    save_fig = False
    avg_accuracies = []
    avg_acc_dict = {}
    knn_lis = [1,3,5,7,10]

    # Finding the average accuracy for each Knn
    for ind, knn in enumerate(knn_lis):
        avg_accuracies.append(five_fold_crossvalidation(df, num_neigh = knn))
        avg_acc_dict[knn] = avg_accuracies[ind]

    # Printing the acccuracies for each Knn
    for key, val in avg_acc_dict.items():
        print(f'Knn: {key}, Avg Accuracy: {val}')

    # Plotting
    plt.style.use('dark_background')
    plt.plot(knn_lis, avg_accuracies, c='orange')
    plt.scatter(knn_lis, avg_accuracies, c='orange')
    plt.title('kNN 5-Fold Cross validation', color='white')
    plt.xlabel('k', color='white')
    plt.ylabel('Average accuracy', color='white')
    plt.grid(True, color='white')
    if save_fig:
        fig_path = r'D:\Documents\UW-Madison\CourseWork\Spring\MachineLearning\Hw\003\figs\q2\q2-p4\\'
        fig_title = 'kNN_5-Fold_Cross_validation.png'
        plt.savefig(fig_path+fig_title, dpi = 300)
    else:
        plt.show()

# Question 2.4 answer
if q2_5:
    save_fig = False
    k = 5

    # Getting the data from both methods
    roc_data, avg_acc_roc = five_fold_crossvalidation(df, fold_num = 1, num_neigh = k, roc = True)
    log_reg, avg_acc_log = five_fold_crossvalidation(df, fold_num = 1, num_neigh = k, learning_rate= 0.1, num_iterations=100, roc = True)
    #print(log_reg, avg_acc_log)

    # Plotting
    plt.style.use('dark_background')
    plt.plot(roc_data[0], roc_data[1], c='orange', label=f'KNeighborsClassifier (AUC = {avg_acc_roc})')
    plt.plot(log_reg[0], log_reg[1], c='blue', label=f'LogisticRegression (AUC = {avg_acc_log})')
    plt.title('kNN ROC Curves, k = 5', color='white')
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.legend()
    plt.grid(True, color='white')
    if save_fig:
        fig_path = r'D:\Documents\UW-Madison\CourseWork\Spring\MachineLearning\Hw\003\figs\q2\q2-p5\\'
        fig_title = 'kNN_ROC_Curves.png'
        plt.savefig(fig_path+fig_title, dpi = 300)
    else:
        plt.show()