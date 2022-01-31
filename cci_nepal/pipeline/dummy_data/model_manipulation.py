# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from operator import itemgetter

from cci_nepal.pipeline.dummy_data import data_manipulation as dm
from cci_nepal.pipeline.dummy_data import nfri_list_file as nlf


def select_model(model_type):

    """
    Takes in the model_type string and returns the chosen model.
    The available model_type are: "linear-regression", "k-nearest", "decision-tree" and "random-forest".

    """

    if model_type == "linear-regression":
        return LinearRegression()
    elif model_type == "k-nearest":
        return KNeighborsRegressor()
    elif model_type == "decision-tree":
        return DecisionTreeRegressor()
    elif model_type == "random-forest":
        return RandomForestRegressor()
    else:
        return print("Please enter the correct nfri type.")


def train_test_validation(X, y, train_size=0.8):

    """
    Takes in the Input and Output features and returns train, validation and test set.
    The default size of train is 80 percent of the dataset.
    The remaining validation and test set are of equal sizes.
    """

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def accuracy_per_item(model, nfri_type, test_input, test_output):

    """
    Takes the test input and output and outputs the accuracy of the Model per NFRI item.

    Parameters:

    model: Trained Machine Learning model
    nfri_type: Type of NFRI to predict (basic-nfri or non-basic-nfri)
    test_input: The test set input dataframe
    test_output: The test set output dataframe

    Returns:

    A pandas dataframe with accuracy per NFRI item
    """

    if nfri_type == "nfri-basic":
        nfri_list = nlf.nfri_basic
    elif nfri_type == "nfri-non-basic":
        nfri_list = nlf.nfri_non_basic
    else:
        return print("Please enter the correct nfri type.")

    test_prediction = model.predict(test_input)
    test_prediction_label = [
        [dm.numeric_score_transformer(i) for i in nested] for nested in test_prediction
    ]
    test_prediction_label_transformed = list(map(list, zip(*test_prediction_label)))
    accuracy_list = []
    for i in range(0, len(nfri_list)):
        accuracy_list.append(
            accuracy_score(test_prediction_label_transformed[i], test_output.iloc[:, i])
        )

    return pd.DataFrame(accuracy_list, test_output.columns, columns=["Accuracy"])


def nfri_calculator(model, nfri_type, input_feature):

    """
    Takes the input features and outputs the sorted NFRI items for the selected features.

    Parameters:

    model: Trained Machine Learning model
    nfri_type: Type of NFRI to predict (basic or non-basic)
    input_feature: A list of input features for which the output is to be predicted

    Returns:

    A pandas dataframe with NFRI Items, Importance Score, and Labels, sorted wrt Importance Score (desc order)
    """

    if nfri_type == "nfri-basic":
        nfri_list = nlf.nfri_basic
    elif nfri_type == "nfri-non-basic":
        nfri_list = nlf.nfri_non_basic
    else:
        return print("Please enter the correct nfri type.")

    yhat = model.predict([input_feature])
    sorted_indices = np.flip(np.argsort(yhat[0]))
    sorted_items = itemgetter(*sorted_indices)(nfri_list)
    sorted_scores = np.sort(yhat[0])[::-1].tolist()
    sorted_labels = [dm.label_transformer(d) for d in -np.sort(-yhat[0])]
    return pd.DataFrame(
        {
            "Sorted Items": sorted_items,
            "Sorted Scores": sorted_scores,
            "Sorted labels": sorted_labels,
        }
    )
