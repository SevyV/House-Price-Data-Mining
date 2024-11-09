#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:27:24 2024

@author: sevyveeken
"""
#This will be the main file where we run everything 
import argparse
import pandas as pd
from preprocessing import Preprocessor
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression

def parse_args():
    parser = argparse.ArgumentParser(description='data files')
    parser.add_argument('--train', type=str, default='data/train.csv',
                        help='data path')
    parser.add_argument('--test', type=str, default='data/test.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.train, a.test)

def main():
    train_path, test_path = parse_args()
    
    train = pd.read_csv(train_path)
    #NOT SURE IF WE NEED TO USE TEST SINCE THE CSV DOES NOT INCLUDE LABELS
    #WE CAN JUST SPLIT TRAIN DATASET USING SKLEARNS TESET-TRAIN-SPLIT FUNCTION
    test = pd.read_csv(test_path)
    
    preprocessor = Preprocessor()
    train = preprocessor.preprocess(train)
    
    #Do feature selection if necessary before splitting train into X and y
    # for feature selection
    X, y = preprocessor.feature_selection(train)
    print(train.head())
    
    
    
    
    
    #FOR pca use
    #X, y = preprocessor.pca(X)
    
    #SPLIT INTO TEST-TRAIN-SPLIT IF NECESSARY
    #SEPERATE y and X IF NECESSARY
    # Label columns is called "PriceCategory"
    
    print(X.head())
    

        
    
main()