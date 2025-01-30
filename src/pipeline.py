import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import os
import random
import pandas as pd
import shutil
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def pipeline_preprocessor (**kwargs) :
    """
    this function builds our transformer part of pipe line 
    it uses standard scaler for standard scaler columns
    robust scaler for robust scaler columns
    baseN encoder for baseN encoder columns
    one hot encoder for one hot encoder columns

    based on the keywork arguments passed to the function
    it creates a processor that can be used in the pipeline

    this function returnes a column transformer object that can be used in the pipeline

    """

    # process the keyword arguments: 

    # we need to check if the user has passed the columns to be used in the pipeline
    # if not we wont add that part to the processor

    standard_scaler_cols = kwargs.get('standard_scaler_cols', [])
    robust_scaler_cols = kwargs.get('robust_scaler_cols', [])
    baseN_enc_cols = kwargs.get('baseN_enc_cols', [])
    one_hot_cols = kwargs.get('one_hot_cols', [])
    ordinal_enc_cols = kwargs.get('ordinal_enc_cols', [])

    # buiding pipeline based on existing arguments : 

    preprocessor = ColumnTransformer(
        transformers=[
            ('standard_scaler', StandardScaler(), standard_scaler_cols),
            ('robust_scaler', RobustScaler(), robust_scaler_cols),
            ('baseN_encoder', ce.BaseNEncoder(cols=baseN_enc_cols), baseN_enc_cols),
            #('ordinal_encoder', OrdinalEncoder(categories=[...], handle_unknown='use_encoded_value', unknown_value=-1), ordinal_enc_cols),
            ('one_hot_encoder', OneHotEncoder(), one_hot_cols),
        ])
    
    return preprocessor

def classifier_pipeline (preprocessor, model) :
    """
    this function takes the preprocessor and the model and returns a pipeline object
    """
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])

    return pipeline

def model_training( X_train, y_train, classifier_pipeline) :
    
    """
    this function trains the model and returns trained model
    """
    classifier_pipeline.fit(X_train, y_train)
    return classifier_pipeline



# make a main for testing the function:

if __name__ == '__main__' :
    # make a preprocessor with just standard scaler and ordinal encoder :

    preprocessor = pipeline_preprocessor(standard_scaler_cols=['age', 'height'], ordinal_enc_cols=['kind', 'type'])
    print(preprocessor)    



