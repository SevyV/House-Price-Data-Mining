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
class Preprocessor:
    
    def preprocess(self, train):
        # Handle numerical na values
        train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
        #test['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
        
        train.dropna(subset=['MasVnrArea'], inplace=True)
        #test.dropna(subset=['MasVnrArea'], inplace=True)
        
        train['GarageYrBlt'].fillna(train['YearBuilt'], inplace=True)
        #test['GarageYrBlt'].fillna(train['YearBuilt'], inplace=True)
        
        train.drop(columns=['Id'], inplace=True)
        
        
        
        # Normalize numerical features
        numerical_features = train.select_dtypes(include=['int64', 'float64']).columns
        numerical_features = numerical_features[numerical_features != 'SalePrice'] 
        scaler = StandardScaler()
        train[numerical_features] = scaler.fit_transform(train[numerical_features])
        #test[numerical_features] = scaler.fit_transform(test[numerical_features])
        
        
        train = self.__hot_encode(train)
        train = self.__create_bin_labels(train)
        
        
        #test = self.__create_bin_labels(test)
        return train #, test
        
    
    
    def feature_selection(self, train, num_features=20):
        #assumes data has already be preprocessed
        y_train = train['PriceCategory']
        X_train = train.drop(columns=['PriceCategory'])
        
        
  
        selector = SelectKBest(mutual_info_classif, k=num_features)
        selector.fit(X_train, y_train)
        selected_columns = X_train.columns[selector.get_support()].tolist()
        X_train_selected_df = X_train[selected_columns]
        #X_test_selected_df = test[selected_columns]

        return X_train_selected_df, y_train #, X_test_selected_df, selected_columns
        
    def __create_bin_labels(self, data):
        # Check for NaN values in 'SalePrice' and handle them if necessary
        if data['SalePrice'].isnull().sum() > 0:
            # Handle missing values (you can either drop them or impute)
            data.dropna(subset=['SalePrice'], inplace=True)
    
        # Define function to map SalePrice to PriceCategory manually
        def map_price_to_category(price):
            if price < 100000:
                return 0
            elif 100000 <= price < 200000:
                return 1
            elif 200000 <= price < 300000:
                return 2
            elif 300000 <= price < 400000:
                return 3
            else:
                return 4
    
        # Apply the function to create the 'PriceCategory' column
        data['PriceCategory'] = data['SalePrice'].apply(map_price_to_category)
    
        # Map the categorical labels to numerical values
        #price_mapping = {'<100,000': 0, '100,000-199,999': 1, '200,000-299,999': 2, '300,000-399,999': 3, '>=400,000': 4}
        #data['PriceCategory'] = data['PriceCategory'].map(price_mapping)
    
        # Drop the original 'SalePrice' column
        data.drop(columns=['SalePrice'], inplace=True)
    
        return data
    
    def pca(self, train, num_features=20):
        y = train['PriceCategory'].copy()
        train.drop(columns=['PriceCategory'], inplace=True)
        X = train
        # assumes data has already be preprocessed
        pca = PCA(n_components=num_features)
        
        # Fit PCA on the training data
        pca.fit(X)
        
        # Transform both training data using the fitted PCA
        train_pca = pca.transform(X)
        #test_pca = pca.transform(test)
        
        # Convert the transformed data into DataFrames with appropriate column names
        train_pca_df = pd.DataFrame(train_pca, columns=[f'PC{i+1}' for i in range(num_features)])
        #test_pca_df = pd.DataFrame(test_pca, columns=[f'PC{i+1}' for i in range(num_features)])
        
        return train_pca_df, y #, test_pca_df
    
    def __hot_encode(self, train):
        categorical_columns = train.select_dtypes(include=['object']).columns
        # Fill missing values (NaN) with a placeholder "unknown"
        for col in categorical_columns:
            train[col] = train[col].fillna('unknown')
        train_encoded = pd.get_dummies(train, columns=categorical_columns, drop_first=True)
        #test_encoded = pd.get_dummies(test, columns=categorical_columns, drop_first=True)
        
        # Ensure that both train and test datasets have the same columns
        # This will add missing columns to the test set with 0s
        #test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
        
        return train_encoded #, test_encoded
        
        
    
    
    
    
    