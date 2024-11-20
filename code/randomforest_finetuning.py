#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:11:30 2024

@author: sevyveeken
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             accuracy_score, precision_score, recall_score, f1_score, roc_curve)
import matplotlib.pyplot as plt
import numpy as np

class RandomForest:
    def random_forest(X_train, X_test, y_train, y_test):
        # Initialize the Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)
        
        # Perform 10-fold cross-validation on the training data (only for training/fit process)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred = cross_val_predict(rf, X_train, y_train, cv=cv)
        
        # Fit the model on the entire training set
        rf.fit(X_train, y_train)
    
        # Predict on the test set
        y_pred_test = rf.predict(X_test)
        y_pred_proba_test = rf.predict_proba(X_test)
        
        # Calculate evaluation metrics on test data
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Confusion matrix on the test data
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
        
        # ROC curve for multiclass classification (on test data)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i in range(len(np.unique(y_test))):  # Loop over the number of classes
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba_test[:, i])
            ax_roc.plot(fpr, tpr, lw=2, label=f"Class {i}")
        
        # Add diagonal line for random guessing
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves (Multiclass) on Test Data")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)
        plt.show()
        
        # Plot confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        # Return results
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }

    
    
    def tune_rf(X_train, X_test, y_train, y_test, param_grid=None):
        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [45, 50, 60, 100],
            'max_depth': [10,20,30],
            'min_samples_split': [8, 10, 12, 15],
            'min_samples_leaf': [2, 6, 10]
        }
    
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    
        grid_search.fit(X_train, y_train)
    
        best_rf = grid_search.best_estimator_
    
        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)
    
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
    
        # ROC curve
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i in range(len(np.unique(y_train))):  # Loop over the number of classes
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            ax_roc.plot(fpr, tpr, lw=2, label=f"Class {i}")
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves (Multiclass) After Fine-Tuning")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)
        plt.show()
    
        # Return results
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
    
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Best params": grid_search.best_params_
        }
    
    