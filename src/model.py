import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error

def rmsle(y_pred, y_true): 
    """Compute the Root Mean Squared Log Error of the y_pred and y_true values"""

    return np.sqrt(mean_squared_log_error(y_true, y_pred))


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


def one_hot(input_df, column_name):
    df = input_df.copy()
    dummies = pd.get_dummies(df[column_name].str.lower())
    dummies.drop(dummies.columns[-1], axis=1, inplace=True)
    return df.drop(column_name, axis=1).merge(dummies, left_index=True, right_index=True)


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

    # fill Nan values with median of column
    # x_cleaned.fillna(x_cleaned.median(axis=0), axis=0, inplace=True)

    return x_cleaned


# TODO: Create pipeline -> input: x array, y array -> pipeline


# TODO: Grid search and cross-validate: input pipeline; return best hyper-parameters and score?
    