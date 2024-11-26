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
from clustering import ClusteringAlgorithm
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
from outlierdetection import OutlierDetection
from classification import Classification


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

    partClustering = ClusteringAlgorithm(X)
    partClustering.run_all()

    partOutlierDetection = OutlierDetection(X)
    partOutlierDetection.run_all_outlier_detection()
    
    # to increased dimensions
    #X, y = preprocessor.feature_selection(data, 10)
    #X, y = preprocessor.pca(data, 10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    partClassification = Classification(X_train, X_test, y_train, y_test)
    partClassification.run_all_classifications()


main()

# python3 main.py --train /Users/kevinpark/Desktop/459/project/House-Price-Data-Mining/code/data/train.csv --test /Users/kevinpark/Desktop/459/project/House-Price-Data-Mining/code/data/test.csv
