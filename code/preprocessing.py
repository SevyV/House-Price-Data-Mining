#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:27:24 2024

@author: sevyveeken
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from collections import Counter


class Preprocessor:
    
    def preprocess(self, train):
        # Handle numerical na values
        train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
        train.dropna(subset=['MasVnrArea'], inplace=True)
        train['GarageYrBlt'].fillna(train['YearBuilt'], inplace=True)
        
        train.drop(columns=['Id'], inplace=True)
        
        # Normalize numerical features
        numerical_features = train.select_dtypes(include=['int64', 'float64']).columns
        numerical_features = numerical_features[numerical_features != 'SalePrice'] 
        scaler = StandardScaler()
        train[numerical_features] = scaler.fit_transform(train[numerical_features])

        # Handle categorical values
        train = self.__hot_encode(train)
        
        # Create bin labels
        train = self.__create_bin_labels(train)
        
        # Data reduction
        label_counts = Counter(train['PriceCategory'])
        print("Number of data points for each label:")
        print(label_counts)
        
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        max_label, second_max_label = sorted_labels[0][0], sorted_labels[1][0]
        
        filtered_data = train[~train['PriceCategory'].isin([max_label, second_max_label])]  
        
        label1_data = train[train['PriceCategory'] == max_label]
        label1_sample = label1_data.sample(n=350, random_state=42)
        
        label2_data = train[train['PriceCategory'] == second_max_label]
        label2_sample = label2_data.sample(n=350, random_state=42)
        
        balanced_train = pd.concat([filtered_data, label1_sample, label2_sample])
        balanced_train = balanced_train.reset_index(drop=True)
        

        # Feature selection 
        mi_selected_feat, mi_y = self.feature_selection(balanced_train)
        lasso_selected_feat, lasso_y = self.lasso_feature_selection(balanced_train)
        
        mi_features = set(mi_selected_feat.columns)
        lasso_features = set(lasso_selected_feat.columns)

        common_features = mi_features.intersection(lasso_features)

        print("Features selected by Mutual Information: ",  mi_features)
        print("Features selected by Lasso Regression: ", lasso_features)
        print("Common features selected by both methods: ", common_features)

        return balanced_train 
        
    def lasso_feature_selection(self, train: pd.DataFrame, alpha=0.067):
        # 0.067
        y_train = train['PriceCategory']
        x_train = train.drop(columns=['PriceCategory'])

        # standardize the features before applying lasso
        lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha, random_state=42))
        lasso.fit(x_train, y_train)

        selected_columns = x_train.columns[(lasso.named_steps['lasso'].coef_ != 0)].tolist()
        x_train_selected_df = x_train[selected_columns]

        return x_train_selected_df, y_train
    
    def feature_selection(self, train, num_features=20):
        # Assumes data has already been preprocessed
        y_train = train['PriceCategory']
        X_train = train.drop(columns=['PriceCategory'])
        
        selector = SelectKBest(mutual_info_classif, k=num_features)
        selector.fit(X_train, y_train)
        
        selected_columns = X_train.columns[selector.get_support()]
        X_train_selected_df = X_train.loc[:, selected_columns]  # Use loc for indexing consistency
        
        self.selected_columns = selected_columns
        
        print(X_train_selected_df.head())
        
        return X_train_selected_df, y_train


    def __create_bin_labels(self, data):
        # Check for NaN values in 'SalePrice' and handle them if necessary
        if data['SalePrice'].isnull().sum() > 0:
            data.dropna(subset=['SalePrice'], inplace=True)
    
        # for binning
        def map_price_to_category(price):
            if price < 100000:
                return 0
            elif 100000 <= price < 150000:
                return 1
            elif 150000 <= price < 200000:
                return 2
            elif 200000 <= price < 300000:
                return 3
            else:
                return 4
    
        # Apply the function to create the 'PriceCategory' column
        data['PriceCategory'] = data['SalePrice'].apply(map_price_to_category)
        data.drop(columns=['SalePrice'], inplace=True)
    
        return data
    
    def pca(self, train, num_features=16):
        y = train['PriceCategory'].copy()
        train.drop(columns=['PriceCategory'], inplace=True)
        X = train
        # assumes data has already be preprocessed
        pca = PCA(n_components=num_features)
        
        pca.fit(X)
        train_pca = pca.transform(X)
       
        # Convert the transformed data into DataFrames 
        train_pca_df = pd.DataFrame(train_pca, columns=[f'PC{i+1}' for i in range(num_features)])
        
        return train_pca_df, y 
    
    def __hot_encode(self, train):
        categorical_columns = train.select_dtypes(include=['object']).columns
        # Fill missing values (NaN) with a placeholder "unknown"
        for col in categorical_columns:
            train[col] = train[col].fillna('unknown')
        train_encoded = pd.get_dummies(train, columns=categorical_columns, drop_first=True)
        
        return train_encoded 
    
    
    
    
    
