"""
@author: kevinpark
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class Clustering:
    def __init__(self, data):
        """
        Initializes the Clustering object with the dataset.

        :param data: Data (features) for clustering
        """
        self.data = data

    def apply_clustering(self, algo, data):
        # Apply 3 clustering algorithms
        if isinstance(algo, KMeans):
            labels = algo.fit_predict(data)
        elif isinstance(algo, DBSCAN):
            labels = algo.fit_predict(data)
        elif isinstance(algo, AgglomerativeClustering):
            labels = algo.fit_predict(data)

        algo_name = algo.__class__.__name__ if hasattr(algo, "__class__") else str(algo)
        print(f"{algo_name}:")

        # Evaluation metrics
        silhouette_score_ = silhouette_score(data, labels)
        calinski_harabasz_score_ = calinski_harabasz_score(data, labels)
        davies_bouldin_score_ = davies_bouldin_score(data, labels)

        print(f"Silhouette Score: {silhouette_score_:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz_score_:.3f}")
        print(f"Davies-Bouldin Score: {davies_bouldin_score_:.3f}\n")

        self.visualize_clusters(algo, data, labels)

    def visualize_clusters(self, algo, X, labels):
        # Visualize the clusters produced by the algorithm.
        """pca = PCA(n_components=2)
        X = pca.fit_transform(X)"""
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")

        algo_name = algo.__class__.__name__ if hasattr(algo, "__class__") else str(algo)
        plt.title(f"{algo_name} Clusters (PCA-reduced)")

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.colorbar(scatter, label="Cluster Label")
        plt.show()

    def apply_kmeans(self):
        kmeans = KMeans(n_clusters=5, random_state=25)  # kmeans++ initialization
        self.apply_clustering(kmeans, self.data)

    def apply_dbscan(self):
        # k-distance plot for tuning hyperparameter
        k = 2
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)

        # Sort the distances for plotting
        distances = np.sort(distances[:, k - 1], axis=0)
        plt.plot(distances)
        plt.title("k-distance graph")
        plt.xlabel("Data Points")
        plt.ylabel(f"{k}-distance")
        plt.show()

        dbscan = DBSCAN(eps=0.7, min_samples=3)
        self.apply_clustering(dbscan, self.data)

    def apply_agg(self):
        agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
        self.apply_clustering(agg, self.data)

    def run_all(self):
        # Run all clustering algorithms and visualize results.
        print("Applying KMeans Clustering:")
        self.apply_kmeans()

        print("Applying DBSCAN Clustering:")
        self.apply_dbscan()

        print("Applying Agglomerative Clustering:")
        self.apply_agg()
