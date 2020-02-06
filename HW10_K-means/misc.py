import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

class KMeans(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max=100):
        all_centers = []
        I = np.eye(self.n_clusters)
        centers = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        for _ in range(iter_max):
            prev_centers = np.copy(centers)
            D = cdist(X, centers)
            cluster_index = np.argmin(D, axis=1)
            cluster_index = I[cluster_index]
            centers = np.sum(X[:, None, :] * cluster_index[:, :, None], axis=0) / np.sum(cluster_index, axis=0)[:, None]
            all_centers.append(centers)
            if np.allclose(prev_centers, centers):
                break
        self.centers = centers
        return all_centers

    def predict(self, X):
        D = cdist(X, self.centers)
        return np.argmin(D, axis=1)