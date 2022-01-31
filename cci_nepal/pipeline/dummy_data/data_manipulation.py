# Import libraries
import pandas as pd
from cci_nepal.pipeline.dummy_data import nfri_list_file as nlf


def label_transformer(d):

    """
    Takes a numerical score and assigns that back to categorical labels.
    If the score is less than 1.5, it is assigned as Unnecessar.
    If less than 2.5 as Desirable, and else as Essential.
    """

    if d < 1.5:
        return "Unnecessary"
    elif d < 2.5:
        return "Desirable"
    else:
        return "Essential"


def numeric_score_transformer(d):

    """
    Takes a numerical score and tranforms in into the following three integer: 1, 2 or 3.
    Round off operation not used as we need any numerical scores greater than 3 to be transformed into 3.
    """

    if d < 1.5:
        return 1
    elif d < 2.5:
        return 2
    else:
        return 3


def replace_columns_with_number(df, columns, values):

    """
    Takes Pandas Dataframe Columns and fills (inplace) all the values in the selected columns with selected Numerical values.
    """

    for one_column, one_value in zip(columns, values):
        df[one_column] = one_value


def nfri_preferences_to_numbers(df):

    """
    Takes in a dataframe a returns a dataframe with nfri categorical preferences transformed into numbers.
    """

    mapping = {"Essential": 3, "Desirable": 2, "Unnecessary": 1}
    return df.applymap(lambda s: mapping.get(s) if s in mapping else s)


def categorize_dataframe_variables(df):

    """
    Takes in a dataframe a returns (in place) dataframe with categorical variables categorized.
    """

    num_cols = df._get_numeric_data().columns
    columns_to_categorize = list(set(df.columns) - set(num_cols))
    df[columns_to_categorize] = df[columns_to_categorize].astype("category")
