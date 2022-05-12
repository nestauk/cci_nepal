# -*- coding: utf-8 -*-
# %%
# Import libraries
import pandas as pd
import numpy as np
from cci_nepal.pipeline.classification_model import nfri_list_file as nlf


def label_transformer(d):

    """
    Takes a numerical score and assigns that back to categorical labels.
    If the score is less than 1.5, it is assigned as Unnecessar.
    If less than 2.5 as Desirable, and else as Essential.
    """

    if d < 1.5:
        return "unnecessary"
    elif d < 2.5:
        return "desirable"
    else:
        return "essential"


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


def nfri_preferences_to_binary(df):
    """
    Takes in a dataframe a returns a dataframe with nfri categorical preferences transformed into numbers.
    """
    mapping = {
        "essential": 1,
        "desirable": 0,
        "unnecessary": 0,
        "essential (अति आवश्यक) ": 1,
        "desirable (आवश्यक)": 0,
        "unnecessary (अनावश्यक)": 0,
    }
    return df.applymap(lambda s: mapping.get(s) if s in mapping else s)


def nfri_preferences_to_binary_non_essential(df):
    """
    Takes in a dataframe a returns a dataframe with nfri categorical preferences transformed into numbers.
    """
    mapping = {
        "essential": 0,
        "desirable": 0,
        "unnecessary": 1,
        "essential (अति आवश्यक) ": 0,
        "desirable (आवश्यक)": 0,
        "unnecessary (अनावश्यक)": 1,
    }
    return df.applymap(lambda s: mapping.get(s) if s in mapping else s)


def nfri_preferences_to_numbers(df):
    """
    Takes in a dataframe a returns a dataframe with nfri categorical preferences transformed into numbers.
    """
    mapping = {
        "essential": 3,
        "desirable": 2,
        "unnecessary": 1,
        "essential (अति आवश्यक) ": 3,
        "desirable (आवश्यक)": 2,
        "unnecessary (अनावश्यक)": 1,
    }
    return df.applymap(lambda s: mapping.get(s) if s in mapping else s)


def categorize_dataframe_variables(df):

    """
    Takes in a dataframe a returns (in place) dataframe with categorical variables categorized.
    """

    num_cols = df._get_numeric_data().columns
    columns_to_categorize = list(set(df.columns) - set(num_cols))
    df[columns_to_categorize] = df[columns_to_categorize].astype("category")


# %%
def transform_sets(df, column_names):
    """
    A series of transformations applied to both the train and test including:
    Dropping un-needed columns, renaming columns, remove new nfri items, update house materials,
    dropping 'other' columns, fill empty values with 0
    """
    columns_to_drop = [5, 16, 29, 39, 41, 42, 54, 55, 67, 68, 69, 70, 71]
    df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
    df.columns = column_names
    # Update house materials column
    cemment_with_bricks = [
        "bricks with ciment",
        "btickets with ciment",
        "brickets with ciment",
    ]
    df.loc[
        df.Material_Other.isin(cemment_with_bricks), "House_Material"
    ] = "cement bonded bricks/stone"

    df["House_Material"] = np.where(
        df.Material_Other.str.contains("mato ghar|clay|mato|mud", case=False, na=False),
        "clay",
        df["House_Material"],
    )

    df.drop(
        ["Material_Other", "Previous_NFRI", "Ethnicity", "Ethnicity_Others"],
        axis=1,
        inplace=True,
    )
    df.fillna(0, inplace=True)
    df["Respondent_Age"] = df["Respondent_Age"].astype(str).map(lambda x: x.strip())


# %%
def feature_creation(df):
    df.insert(2, "household_size", df.iloc[:, 5:25].sum(axis=1))
    df.insert(3, "total_female", df.iloc[:, 14:24].sum(axis=1))
    df.insert(4, "percent_female", (df.total_female / df.household_size) * 100)
    df.drop(["total_female"], axis=1, inplace=True)
    df.insert(4, "children", df.iloc[:, [7, 8, 9, 17, 18, 19]].sum(axis=1))
    df["children"] = np.where(df.children > 0, 1, 0)
    # df.insert(5, "children_under_5", df.iloc[:, [8, 18]].sum(axis=1))
    # df["children_under_5"] = np.where(df.children_under_5 > 0, 1, 0)
    # df.insert(
    #    6,
    # "adults",
    # df.iloc[:, [12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28]].sum(
    #    axis=1
    # ),
    # )
    df.insert(
        5,
        "income_gen_ratio",
        ((df.Income_Generating_Members / df.household_size) * 100),
    )
    # df.insert(
    # 8, "income_gen_adults", ((df.Income_Generating_Members / df.adults) * 100)
    # )
    df.insert(6, "health_difficulty", df.iloc[:, [33, 34, 35, 36, 37, 38]].sum(axis=1))
    df.health_difficulty = np.where(df.health_difficulty > 0, 1, 0)
    df["respondent_female"] = np.where(df.Respondent_Gender == "female", 1, 0)
    # df["previous_nfri"] = np.where(df.Previous_NFRI == "yes", 1, 0)
    df["sindupalchowk"] = np.where(df.District == "sindupalchok", 1, 0)
    df.income_gen_ratio = df.income_gen_ratio.replace(np.inf, np.nan)
    # df.income_gen_adults = df.income_gen_adults.replace(np.inf, np.nan)
    df.fillna(0, inplace=True)


# %%