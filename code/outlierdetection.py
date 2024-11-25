import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


class OutlierDetection:
    def __init__(self, X):
        """
        Initializes the OutlierDetection object with the data to be analyzed.

        :param X: Data (features) for outlier detection
        """
        self.X = X

    def isolation_forest(self):
        # Perform Isolation Forest outlier detection
        iso_forest = IsolationForest()
        outliers_iso = iso_forest.fit_predict(self.X)
        return outliers_iso

    def local_outlier_factor(self):
        # Perform Local Outlier Factor (LOF) outlier detection
        lof = LocalOutlierFactor(n_neighbors=40)
        outliers_lof = lof.fit_predict(self.X)
        return outliers_lof

    def elliptic_envelope(self):
        # Perform Elliptic Envelope outlier detection
        elliptic_env = EllipticEnvelope()
        outliers_elliptic = elliptic_env.fit_predict(self.X)
        return outliers_elliptic

    def plot_outlier_detection(self):
        # Plot the visualization of all outlier detection methods
        outliers_iso = self.isolation_forest()
        outliers_lof = self.local_outlier_factor()
        outliers_elliptic = self.elliptic_envelope()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        # Plot Isolation Forest results
        axes[0].scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=(outliers_iso == 1),
            cmap="coolwarm",
            edgecolor="k",
            s=40,
        )
        axes[0].set_title("Isolation Forest")

        # Plot LOF results
        axes[1].scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=(outliers_lof == 1),
            cmap="coolwarm",
            edgecolor="k",
            s=40,
        )
        axes[1].set_title("Local Outlier Factor")

        # Plot Elliptic Envelope results
        axes[2].scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=(outliers_elliptic == 1),
            cmap="coolwarm",
            edgecolor="k",
            s=40,
        )
        axes[2].set_title("Elliptic Envelope")

        plt.suptitle("Outlier Detection Visualization")
        plt.show()

    def run_all_outlier_detection(self):
        # Run all outlier detection methods and plot the results."""
        self.plot_outlier_detection()
