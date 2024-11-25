"""
Created on Thu Nov  7 20:27:24 2024

@author: sevyveeken
"""

# This will be the main file where we run everything
import argparse
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.decomposition import PCA
from preprocessing import Preprocessor
from clustering import KMeansAlgo, DBSCANAlgo, HierchicalAlgo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from randomforest_finetuning import RandomForest


def parse_args():
    parser = argparse.ArgumentParser(description="data files")
    parser.add_argument("--train", type=str, default="data/train.csv", help="data path")
    parser.add_argument("--test", type=str, default="data/test.csv", help="data path")

    a = parser.parse_args()
    return (a.train, a.test)


def main():
    train_path, test_path = parse_args()

    data = pd.read_csv(train_path)
    # NOT SURE IF WE NEED TO USE TEST SINCE THE CSV DOES NOT INCLUDE LABELS
    # WE CAN JUST SPLIT TRAIN DATASET USING SKLEARNS TESET-TRAIN-SPLIT FUNCTION
    # test = pd.read_csv(test_path)

    """ PRE-PROCESSING """
    preprocessor = Preprocessor()
    data = preprocessor.preprocess(data)

    # Do feature selection if necessary before splitting train into X and y
    # for feature selection
    # X, y = preprocessor.feature_selection(train)
    # print(train.head())

    # FOR pca use
    # X, y = preprocessor.pca(data)
    # print(X.head())

    # SPLIT INTO TEST-TRAIN-SPLIT IF NECESSARY
    # SEPERATE y and X IF NECESSARY
    # Label columns is called "PriceCategory"
    X = data.drop(columns=["PriceCategory"])
    y = data["PriceCategory"]

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    """ END PRE PROCESSING """

    """ CLUSTERING AND VISUALIZATION """

    kmeans = KMeansAlgo()
    dbscan = DBSCANAlgo()
    agg = HierchicalAlgo()
    kmeans.apply_kmeans(X)
    agg.apply_agg(X)
    dbscan.apply_dbscan(X)

    """ END CLUSTERING AND VISUALIZATION """

    """ OUTLIER DETECTION AND VISUALIZATION """

    iso_forest = IsolationForest()
    outliers_iso = iso_forest.fit_predict(X)

    lof = LocalOutlierFactor(n_neighbors=40)
    outliers_lof = lof.fit_predict(X)

    elliptic_env = EllipticEnvelope()
    outliers_elliptic = elliptic_env.fit_predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Plot Isolation Forest results
    axes[0].scatter(
        X[:, 0], X[:, 1], c=(outliers_iso == 1), cmap="coolwarm", edgecolor="k", s=40
    )
    axes[0].set_title("Isolation Forest")

    # Plot LOF results
    axes[1].scatter(
        X[:, 0], X[:, 1], c=(outliers_lof == 1), cmap="coolwarm", edgecolor="k", s=40
    )
    axes[1].set_title("Local Outlier Factor")

    # Plot Elliptic Envelope results
    axes[2].scatter(
        X[:, 0],
        X[:, 1],
        c=(outliers_elliptic == 1),
        cmap="coolwarm",
        edgecolor="k",
        s=40,
    )
    axes[2].set_title("Elliptic Envelope")

    plt.suptitle("Outlier Detection Visualization")
    plt.show()

    """ END OUTLIER DETECTION AND VISUALIZATION """

    """ CLASSIFICIATION AND EVALUATION """

    # k-NN classification and evaluation
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    y_prediction = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    print(f"Accuracy: {accuracy:.2f}")

    print("k-NN Classification Report:")
    print(classification_report(y_test, y_prediction))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))

    n_classes = 5
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    y_prob = knn_classifier.predict_proba(X_test)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class Receiver Operating Characteristic (ROC) Curve for k-NN")
    plt.legend(loc="lower right")
    plt.show()

    # SVM Classification and Evaluation
    svm_classifier = SVC(kernel="linear", random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_prediction = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print(f"Accuracy: {accuracy:.2f}")
    # accuracy score without feature selection 0.81

    print(" SVM Classification Report:")
    print(classification_report(y_test, y_prediction))
    # macro avg's: precision 0.88, recall 0.67, f1-score 0.74, support 291 (idk wtf this is)
    # weighted avg's: precision 0.82, recall 0.81, f1-score 0.81, support 291

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_prediction))
    """
    Confusion Matrix
    [[  8  12   0   0   0]
    [  0 170  12   0   0]
    [  0  20  40   0   0]
    [  0   2   6  13   0]
    [  0   0   0   2   6]]    
    """

    # Random Forest Results
    results_initial = RandomForest.random_forest(X_train, X_test, y_train, y_test)
    print("Initial RF Results:", results_initial)
    # Fine-tuning RF results
    results_tuned = RandomForest.tune_rf(X_train, X_test, y_train, y_test)
    print("Fine-Tuned RF Results:", results_tuned)


main()

# python3 main.py --train /Users/kevinpark/Desktop/459/project/House-Price-Data-Mining/code/data/train.csv --test /Users/kevinpark/Desktop/459/project/House-Price-Data-Mining/code/data/test.csv
