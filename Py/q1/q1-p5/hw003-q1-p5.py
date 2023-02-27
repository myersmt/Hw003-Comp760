"""
Created by Matt Myers
Due 02/27/20223
Hw 003

Description:

(9pts) Again, suppose you trained a classifier for a spam filter. The prediction result on the test set is
summarized in the following table. Here, ”+” represents spam, and ”-” means not spam.

Confidence positive   |   Correct class
0.95 +
0.85 +
0.80 -
0.70 +
0.55 +
0.45 -
0.40 +
0.30 +
0.20 -
0.10 -

(a) (6pts) Draw a ROC curve based on the above table.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Figure display options
plot_show = True

# File options
file = 'spam_filter.txt'

# read in info
path_to_data = r'\Py\q1\q1-p5'
os.chdir(os.getcwd()+path_to_data)
data = pd.read_csv(file, sep=' ', names=["confidence_positive","correct_class"])

# Initializing the maximum values
true_pos_tot = pd.Series(data['correct_class']).value_counts()[1]
false_pos_tot = pd.Series(data['correct_class']).value_counts()[0]
tot = true_pos_tot + false_pos_tot

# Calculating roc information
#   First roc is 0
true_pos, false_pos = 0, 0
roc_true_pos, roc_false_pos = [0.0], [0.0]

#   Looping through each row of the data frame
for index, row in data.iterrows():
    #   Finding the class_value for the line
    class_value = row['correct_class']
    #   Skipping the first points as the roc will be 0
    if index == 0 :
        pass
    else:
        #   If the class is not positive and also is different than the previous value append it
        if class_value != 1 and data.correct_class[index] != data.correct_class[index-1]:
            roc_false_pos.append(false_pos / false_pos_tot)
            roc_true_pos.append(true_pos / true_pos_tot)
    #   If the class value is one update true positive otherwise update false positive
    if class_value == 1.0:
        true_pos += 1
    else:
        false_pos += 1
roc_false_pos.append(1.0)
roc_true_pos.append(1.0)

#Plotting
#   Plot Theme
plt.style.use('dark_background')

#   Plot attributes
fig, ax = plt.subplots()
ax.plot(roc_false_pos, roc_true_pos, color='purple', linewidth=2, label='ROC Curve')
ax.scatter(roc_false_pos, roc_true_pos, color='purple', s=40)

#   Plot Description
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: Question 1 part 5b')

#   Y-label scatters
for x, y in zip(roc_false_pos, roc_true_pos):
    ax.text(x+0.01, y, f'{y:.3f}', fontsize=8, color='white', ha='left', va='top')

#   Plot legend
ax.legend()

# Save figure
if plot_show:
    plt.show()
else:
    fig_path = r'D:\Documents\UW-Madison\CourseWork\Spring\MachineLearning\Hw\003\figs\q1\q1-p5'
    fig_title = 'roc_curve_q1-p5b.png'

    plt.savefig(fig_path+fig_title, dpi = 300)