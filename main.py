from collections import deque

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
a = deque()
for X in colA_test:
    T = X.value.split(' ')
    if len(T) > 2:
        T.pop(2)
    R = map(int, T)
    E = list(R)
    a.append(E)
X_test = np.array(a)
print(X_test)

#Prediction
knn = KNN(k=3)
knn.fit(X_set, Y_set)
Y_pred = knn.predict(X_test)

print(Y_pred)
