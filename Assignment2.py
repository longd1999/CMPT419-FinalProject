import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import seaborn as sns

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Part 1
cwd = os.getcwd()  # get the current working directory 
files = os.listdir(cwd)  # get all the files in that directory

df = pd.read_csv(files[1]) # whole dataset

#14th col is class, 0 is none, >0 is infected, type: int64
column_14 = df.iloc[:, 13]
# assign name to columns
column_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 
                'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10',
                'Feature_11', 'Feature_12', 'Feature_13', 'Target']

# assign column names
df.columns = column_names

# map 2,3,4 into 1s
def map_values(value):
    if value >= 1:
        return 1
    else:
        return value

# apply mapping function to the column
df['Target'] = df['Target'].map(map_values)

#tuning
param_dist = {'n_estimators': randint(50,500),
            'max_depth': randint(1,20)}
# split the data into training and test sets
Data_train, Data_test = train_test_split(df, test_size=0.5,random_state=42)
"""
print("print the training data \n")
print(Data_train)
print("print the test data \n")
print(Data_test)
"""

def rand_forest(Data_train, Data_test):
    # bagging on train data
    X_train = Data_train.sample(frac=1, replace=True)
    """
    print("print the bagged data \n")
    print(X_train)
    """
    # split train data

    X1 = X_train.drop('Target', axis=1)
    y1 = X_train['Target']
    # split test data
    X2 = Data_test.drop('Target', axis=1)
    y2 = Data_test['Target']
    # create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

    # fit using train data
    rand_search.fit(X1, y1)
    # predict on test data
    y_pred = rand_search.predict(X2)
    accuracy = accuracy_score(y2, y_pred)
    # generate conf matrix
    conf_matrix = confusion_matrix(y2, y_pred)
    return accuracy, conf_matrix
    
t1acc, t1conf = rand_forest(Data_train, Data_test)
print("Task 1 confusion matrix:\n")
print(t1conf)

"""
#Part 2 brute force LOO
print("Task 2:\n")
def compute_loo_influence(Data_train, Data_test):
    loo_influence_m = []
    rows_to_drop = Data_train.sample(n=10).index
    for i in range(10):
        print("Row before dropping:", Data_train.loc[rows_to_drop[i]])
        # exclude one row from training data
        X_train_loo = Data_train.drop(index=rows_to_drop[i])
        acc, conf_matrix = rand_forest(X_train_loo, Data_test)
        # compute influence matrix
        diff = t1conf - conf_matrix
        # store influence matrix
        loo_influence_m .append(diff)

    return loo_influence_m 

# perform LOO influence calculation
loo_influence_m  = compute_loo_influence(Data_train, Data_test)
for i, matrix in enumerate(loo_influence_m):
    print(f"Confusion Matrix {i + 1}:")
    df = pd.DataFrame(matrix, columns=["Negative", "Positive"], index=["False ", "True"])
    print(df)
    print()
    
"""
"""
# Part 3
def compute_loo_influence(Data_train, Data_test):
    loo_influence_m = []
    num_rows_to_delete = 10
    for i in range(10):
        # exclude rows from training data
        X_train_loo = Data_train.iloc[num_rows_to_delete:]
        acc, conf_matrix = rand_forest(X_train_loo, Data_test)
        # compute influence acc
        diff = t1acc - acc
        # store influence acc
        loo_influence_m .append(diff)
        # increase of 10
        num_rows_to_delete += 10

    return loo_influence_m 

# perform LOO influence calculation
loo_influence_m  = compute_loo_influence(Data_train, Data_test)
group_sizes = np.arange(10, 10 * (len(loo_influence_m) + 1), 10)

plt.plot(group_sizes, loo_influence_m, marker='o', linestyle='-')
plt.xlabel('Group Size')
plt.ylabel('LOO influence')
plt.title('LOO influence vs Group Size')
plt.grid(True)
plt.show()
"""
"""
#Part 4
def compute_shapley_values(Data_train, Data_test, num_permutations=10):
    shapley_values = np.zeros(len(Data_train))
    
    for j in range(num_permutations):
        print("Permutation: ", j )
        # generate random permutation of the dataset
        permuted_indices = np.random.permutation(len(Data_train))
        
        predictions = np.zeros(len(Data_train))
        
        for i in range(len(Data_train)):
            # exclude one observation from training data
            X_train_loo = Data_train.iloc[np.concatenate((permuted_indices[:i], permuted_indices[i+1:]))]
            
            # train model and predict on test data
            acc, _ = rand_forest(X_train_loo, Data_test)
            
            # compute sv for each training data point
            shapley_values[i] += (acc - shapley_values[i]) / (i + 1)
    
    return shapley_values / num_permutations

# perform sv computation
shapley_values = compute_shapley_values(Data_train, Data_test)

# plot sv distribution
plt.hist(shapley_values, bins=20, edgecolor='black')
plt.xlabel('Shapley Value')
plt.ylabel('Frequency')
plt.title('Shapley Value Distribution')
plt.grid(True)
plt.show()
"""