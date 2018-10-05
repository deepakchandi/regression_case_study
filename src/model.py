import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: train test split? where?

def import_data(filepath):
    """Read file from the path and return Pandas DataFrame"""
    return pd.read_csv(filepath)


def select_features(df):
    """Input the full dataset DataFrame and return features frame and taret frame."""
    X_df = df[[ 'MachineID', 
                'ModelID', 
                'auctioneerID', 
                'YearMade', 
                'state',
                'MachineHoursCurrentMeter']] # being called into question
    
    y_df = df['SalePrice']

    return X_df, y_df

# TODO: Feature clean-up -> input: x array -> cleaned x array
# def feature_eng(x_df):
#     x_df.get_dummies

def clean_features(x_df):
    """Input features DataFrame, clean features by columns, 
    and return cleaned DataFrame"""
    x_cleaned = x_df.copy()
    
    # State: dummy columns
    states = pd.get_dummies(x_df['state'])
    states.drop(['Unspecified', 'Washington DC'], axis=1, inplace=True)
    x_cleaned.drop('state', axis=1, inplace=True)
    x_cleaned = x_cleaned.merge(states, left_index=True, right_index=True)

    return x_cleaned


# TODO: Create pipeline -> input: x array, y array -> pipeline


# TODO: Grid search and cross-validate: input pipeline; return best hyper-parameters and score?
    