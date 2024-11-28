"""
Created on Thu Nov  7 20:27:24 2024

@author: sevyveeken
"""

# This will be the main file where we run everything
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from preprocessing import Preprocessor
from clustering import Clustering
from sklearn.model_selection import train_test_split
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

    """ PRE-PROCESSING """
    preprocessor = Preprocessor()
    data = preprocessor.preprocess(data)

    # Do feature selection if necessary before splitting train into X and y
    # for feature selection
    # X, y = preprocessor.feature_selection(train)
    # X, y = preprocessor.lasso_feature_selection(train)
    # print(train.head())

    X = data.drop(columns=["PriceCategory"])
    y = data["PriceCategory"]

    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)

    """ END PRE PROCESSING """

    partClustering = Clustering(X_2)
    partClustering.run_all()

    partOutlierDetection = OutlierDetection(X)
    partOutlierDetection.run_all_outlier_detection()

    # to increased dimensions
    # X, y = preprocessor.feature_selection(data)
    # X, y = preprocessor.pca(data, 10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    partClassification = Classification(X_train, X_test, y_train, y_test)
    partClassification.run_all_classifications()


main()
