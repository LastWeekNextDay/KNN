import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_set = X
        self.Y_set = y

    def _euclidean_distance(self, x1, x2):
        print(f"x1({x1}) with x2({x2})")
        print(f"Result: {np.sqrt(np.sum((x1 - x2) ** 2))}")
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        Y_pred = [self._predict(x) for x in X]
        return np.array(Y_pred)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_in_set) for x_in_set in self.X_set]
        k_indexes = np.argsort(distances)[:self.k]
        print(k_indexes)
        k_labels = [self.Y_set[i] for i in k_indexes]
        prediction = Counter(k_labels).most_common(1)
        return prediction[0][0]

# Sample data
X_set = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
Y_set = np.array(["A", "B", "A", "B", "B"])
X_test = np.array([[4, 5]])

# Create a KNN classifier with k=2
knn = KNN(k=3)
knn.fit(X_set, Y_set)

# Perform KNN classification
y_pred = knn.predict(X_test)

# Print the predicted class label for the new data point
print(y_pred)
print(f"The new data point belongs to class: {y_pred[0]}")