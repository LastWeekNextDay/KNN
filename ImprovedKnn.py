from collections import Counter
import numpy as np
from sklearn.cluster import KMeans

class ImprovedKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_set = X
        self.cluster_labels = y  # Use cluster labels as "y"

    def predict(self, X):
        Y_pred = [self._predict(x) for x in X]
        return np.array(Y_pred)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_in_set) for x_in_set in self.X_set]
        k_indexes = np.argsort(distances)[:self.k]
        k_cluster_labels = [self.cluster_labels[i] for i in k_indexes]
        prediction = Counter(k_cluster_labels).most_common(1)
        return prediction[0][0]


