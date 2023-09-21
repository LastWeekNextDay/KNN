from collections import deque

import pandas as pd

from KNN import KNN
from openpyxl import load_workbook
import numpy as np

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

#Write to file
X_set = np.append(X_set, X_test, axis=0)
Y_set = np.append(Y_set, Y_pred, axis=0)

#seen = set()
#X_set = [item for item in X_set if not(tuple(item) in seen or seen.add(tuple(item)))]
#print(X_set)



resultX = []
resultY = []
iter = 0
for i in X_set:
    in_list = any(np.array_equal(i, arr) for arr in resultX)
    if not in_list:
        resultX.append(i)
        resultY.append(Y_set[iter])
    iter = iter + 1
X_set = np.array(resultX)
Y_set = np.array(resultY)
df = pd.DataFrame(X_set, Y_set)
df.to_excel('Data.xlsx', index=False, header=False)

