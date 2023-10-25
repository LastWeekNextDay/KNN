from collections import deque

import pandas as pd

from ImprovedKnn import ImprovedKNN
from KNN import KNN
from openpyxl import load_workbook
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# training data
wb = load_workbook(filename='Data.xlsx')
ws = wb.active
colA = ws['A']
colB = ws['B']
X_set = np.array([])
a = deque()
for X in colA:
    T = X.value.split(' ')
    if len(T) > 2:
        T.pop(2)
    R = map(int, T)
    E = list(R)
    a.append(E)
X_set = np.array(a)
Y_set = np.array([])
for Y in colB:
    Y_set = np.append(Y_set, Y.value)

#Test data
wb_test = load_workbook(filename='Input.xlsx')
ws_test = wb_test.active
colA_test = ws_test['A']
X_test = np.array([])
a.clear()
for X in colA_test:
    T = X.value.split(' ')
    if len(T) > 2:
        T.pop(2)
    R = map(int, T)
    E = list(R)
    a.append(E)
X_test = np.array(a)

#Prediction
knn = KNN(k=1)
knn.fit(X_set, Y_set)
Y_pred = knn.predict(X_test)

##Write to file
#X_set = np.append(X_set, X_test, axis=0)
#Y_set = np.append(Y_set, Y_pred, axis=0)
#resultX = []
#resultY = []
#iter = 0
#for i in X_set:
#    in_list = any(np.array_equal(i, arr) for arr in resultX)
#    if not in_list:
#        resultX.append(i)
#        resultY.append(Y_set[iter])
#    iter = iter + 1
#X_set = np.array(resultX)
#Y_set = np.array(resultY)
#X_set_save = np.array([])
#for i in X_set:
#    ch = str(i[0]) + " " + str(i[1])
#    X_set_save = np.append(X_set_save, ch)
#data_set = np.array([])
#for i in range(len(X_set)):
#    data_set = np.append(data_set, ([[X_set_save[i]], [Y_set[i]]]))
#new_list = []
## Iterate over the original list up to the second-to-last element with a step of 2
#for i in range(0, len(data_set) - 1, 2):
#    # Create a 2D NumPy array with the first two elements as a single string and the third element
#    elements = [data_set[i], data_set[i + 1]]
#    new_array = np.array(elements).reshape(1, -1)
#    new_list.append(new_array)
#
## If there's a last element, add it separately
#if len(data_set) % 2 != 0:
#    new_list.append(np.array([data_set[-1], ""]))
#
## Combine the 2D arrays into a single 2D NumPy array
#result_array = np.vstack(new_list)
#
#df = pd.DataFrame(result_array)
#df.to_excel('Data.xlsx', index=False, header=False)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(X_set)

# Create a scatter plot for each cluster
for cluster_num in range(3):
    plt.scatter(X_set[cluster_labels == cluster_num, 0], X_set[cluster_labels == cluster_num, 1], label=f'Cluster {cluster_num + 1}')

# Optionally, plot the cluster centroids
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='black', marker='X', label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Cluster Visualization')
plt.show()

# Create and fit the improved KNN model with cluster labels
improved_knn = ImprovedKNN(k=1)
improved_knn.fit(X_set, cluster_labels)

# Getting the number of iterations
iterations = kmeans.n_iter_
print("Number of iterations:", iterations)

# Predict cluster labels for the test data
cluster_predictions = improved_knn.predict(X_test)

# Output cluster predictions
print("Cluster Predictions:")
for i, cluster_label in enumerate(cluster_predictions):
    cluster_visual = f"[{cluster_label}]"
    print(f"Data Point {i + 1}: {cluster_visual}", end=' ')
    if (i + 1) % 10 == 0:
        print()  # Start a new line after every 10 data points

# Print a legend
print("\nCluster Legend:")
for cluster_label in set(cluster_predictions):
    cluster_visual = f"[{cluster_label}]"
    print(f"Cluster {cluster_label}: {cluster_visual}")

