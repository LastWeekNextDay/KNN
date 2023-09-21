from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_set = X
        self.Y_set = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        Y_pred = [self._predict(x) for x in X]
        return np.array(Y_pred)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_in_set) for x_in_set in self.X_set]
        k_indexes = np.argsort(distances)[:self.k]
        k_labels = [self.Y_set[i] for i in k_indexes]
        prediction = Counter(k_labels).most_common(1)
        return prediction[0][0]