import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from randomforest_finetuning import RandomForest
import time


class Classification:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initializes the classification object with training and testing data.

        :param X_train: Training data
        :param X_test: Test data
        :param y_train: Training labels
        :param y_test: Test labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def knn_classification(self):
        start_time = time.time()
        knn_classifier = KNeighborsClassifier(n_neighbors=5)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)
        cv_scores = cross_val_score(
            knn_classifier, self.X_train, self.y_train, cv=cv, scoring="accuracy"
        )

        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(
            f"Mean accuracy from cross-validation: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}"
        )

        knn_classifier.fit(self.X_train, self.y_train)
        y_prediction = knn_classifier.predict(self.X_test)
        end_time = time.time()

        accuracy = accuracy_score(self.y_test, y_prediction)

        print(f"Accuracy (k-NN): {accuracy:.2f}")
        print("k-NN Classification Report:")
        print(classification_report(self.y_test, y_prediction))
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_prediction)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=np.unique(self.y_test)
        )
        disp.plot()
        print("k-NN time (seconds) : ", end_time - start_time)

        # Multi-class ROC Curve
        n_classes = 5
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2, 3, 4])
        y_prob = knn_classifier.predict_proba(self.X_test)

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

    def svm_classification(self):
        start_time = time.time()
        svm_classifier = SVC(kernel="linear", random_state=42)
        svm_classifier.fit(self.X_train, self.y_train)
        y_prediction = svm_classifier.predict(self.X_test)
        end_time = time.time()

        accuracy = accuracy_score(self.y_test, y_prediction)
        print(f"Accuracy (SVM): {accuracy:.2f}")
        print("SVM Classification Report:")
        print(classification_report(self.y_test, y_prediction))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_prediction))
        print("SVM time (seconds) : ", end_time - start_time)

    def random_forest_classification(self):
        # Initial Random Forest results
        results_initial = RandomForest.random_forest(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("Initial RF Results:", results_initial)

        # Fine-tuned Random Forest results
        results_tuned = RandomForest.tune_rf(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("Fine-Tuned RF Results:", results_tuned)

    def run_all_classifications(self):
        # Run all classification methods and evaluations
        self.knn_classification()
        self.svm_classification()
        self.random_forest_classification()
