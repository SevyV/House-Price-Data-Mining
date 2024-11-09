"""
@author: kevinpark
"""

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class ClusteringAlgorithm:
    def apply_clustering(self, algo, X_train, X_test):

        algo.fit(X_train)

        if isinstance(algo, KMeans):
            train_labels = algo.predict(X_train)
            test_labels = algo.predict(X_test)
        elif isinstance(algo, DBSCAN):
            train_labels = algo.labels_
            test_labels = algo.fit_predict(X_test)

        self.visualize_clusters(algo, X_test, test_labels)

        silhouette_score_ = silhouette_score(X_test, test_labels)
        calinski_harabasz_score_ = calinski_harabasz_score(X_test, test_labels)
        davies_bouldin_score_ = davies_bouldin_score(X_test, test_labels)

        print(f"Silhouette Score: {silhouette_score_:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz_score_:.3f}")
        print(f"Davies-Bouldin Score: {davies_bouldin_score_:.3f}")

    def visualize_clusters(self, algo, X, labels):

        X = PCA(X, 2)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")

        algo_name = algo.__class__.__name__ if hasattr(algo, "__class__") else str(algo)
        plt.title(f"{algo_name} Clusters (PCA-reduced)")

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.colorbar(scatter, label="Cluster Label")

        plt.show()


class KMeansAlgo(ClusteringAlgorithm):
    def apply_kmeans(self, X_train, X_test):
        kmeans = KMeans(n_clusters=3)  # kmeans++ initialization
        self.apply_clustering(kmeans, X_train, X_test)


class DBSCANAlgo(ClusteringAlgorithm):
    def apply_dbscan(self, X_train, X_test):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.apply_clustering(dbscan, X_train, X_test)
