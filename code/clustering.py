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


class ClusteringAlgorithm:
    def apply_clustering(self, algo, data):

        if isinstance(algo, KMeans):
            labels = algo.fit_predict(data)
        elif isinstance(algo, DBSCAN):
            labels = algo.fit_predict(data)
        elif isinstance(algo, AgglomerativeClustering):
            labels = algo.fit_predict(data)

        self.visualize_clusters(algo, data, labels)

        silhouette_score_ = silhouette_score(data, labels)
        calinski_harabasz_score_ = calinski_harabasz_score(data, labels)
        davies_bouldin_score_ = davies_bouldin_score(data, labels)

        print(f"Silhouette Score: {silhouette_score_:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz_score_:.3f}")
        print(f"Davies-Bouldin Score: {davies_bouldin_score_:.3f}")

    def visualize_clusters(self, algo, X, labels):

        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")

        algo_name = algo.__class__.__name__ if hasattr(algo, "__class__") else str(algo)
        plt.title(f"{algo_name} Clusters (PCA-reduced)")

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.colorbar(scatter, label="Cluster Label")

        plt.show()


class KMeansAlgo(ClusteringAlgorithm):
    def apply_kmeans(self, data):
        kmeans = KMeans(n_clusters=3)  # kmeans++ initialization
        self.apply_clustering(kmeans, data)


class DBSCANAlgo(ClusteringAlgorithm):
    def apply_dbscan(self, data):
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.apply_clustering(dbscan, data)


class HierchicalAlgo(ClusteringAlgorithm):
    def apply_agg(self, data):
        agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
        self.apply_clustering(agg, data)
