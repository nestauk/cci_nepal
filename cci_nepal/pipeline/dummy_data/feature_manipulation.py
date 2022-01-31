# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from cci_nepal.pipeline.dummy_data import nfri_list_file as nlf


def input_feature_selection(df, divider_index):

    """
    Returns a dataframe with input feature columns only for Model Fitting step
    """

    input_feature = df.iloc[:, :-divider_index]
    return input_feature


def output_feature_selection(df, nfri_type):

    """
    Returns a dataframe with output feature columns only for Model Fitting step
    """

    if nfri_type == "nfri-basic":
        return df.loc[:, nlf.nfri_basic]
    elif nfri_type == "nfri-non-basic":
        return df.loc[:, nlf.nfri_non_basic]
    else:
        return print("Please enter the correct nfri type.")


def feature_scaling(df):

    """
    Takes in a dataframe and scales the numerical features using Min-Max scaling.
    Performs inplace change into the dataframe.
    """

    scaler = MinMaxScaler()
    numerical_df = df.select_dtypes(include="number")
    df[numerical_df.columns] = scaler.fit_transform(numerical_df)


def feature_dummying(df):

    """
    Takes in a dataframe and makes dummy variables for all categorical variables.
    Returns a dataframe instead of performing inplace change.
    """

    return pd.get_dummies(df, drop_first=True)


def feature_creation(df):

    """
    This will be created once we understand the various nature of features to be created in real data.
    """
