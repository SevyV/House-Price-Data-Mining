
"""
Created on Thu Nov  7 20:27:24 2024

@author: sevyveeken
"""
# This will be the main file where we run everything
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from preprocessing import Preprocessor
from clustering import KMeansAlgo, DBSCANAlgo, HierchicalAlgo
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    # clustering and evaluation

    kmeans = KMeansAlgo()
    dbscan = DBSCANAlgo()
    agg = HierchicalAlgo()
    kmeans.apply_kmeans(X)
    agg.apply_agg(X)
    dbscan.apply_dbscan(X)

    # SVM Classification and Evaluation 
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train,y_train)
    y_prediction = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_prediction)
    print(f"Accuracy: {accuracy:.2f}")
    # accuracy score without feature selection 0.81

    print("Classification Report:")
    print(classification_report(y_test,y_prediction))
    # macro avg's: precision 0.88, recall 0.67, f1-score 0.74, support 291 (idk wtf this is)
    # weighted avg's: precision 0.82, recall 0.81, f1-score 0.81, support 291

    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_prediction))
    """
    Confusion Matrix
    [[  8  12   0   0   0]
    [  0 170  12   0   0]
    [  0  20  40   0   0]
    [  0   2   6  13   0]
    [  0   0   0   2   6]]    
    """
    
    #Random Forest Results
    results_initial = RandomForest.random_forest(X_train, X_test, y_train, y_test)
    print("Initial RF Results:", results_initial)
    #Fine-tuning RF results
    results_tuned = RandomForest.tune_rf(X_train, X_test, y_train, y_test)
    print("Fine-Tuned RF Results:", results_tuned)


main()

# python3 main.py --train /Users/kevinpark/Desktop/459/project/House-Price-Data-Mining/code/data/train.csv --test /Users/kevinpark/Desktop/459/project/House-Price-Data-Mining/code/data/test.csv
