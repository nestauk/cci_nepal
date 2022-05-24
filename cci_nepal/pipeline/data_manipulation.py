# -*- coding: utf-8 -*-
# %%
# Import libraries
import pandas as pd
import numpy as np
from cci_nepal.pipeline import nfri_list_file as nlf


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
        df.material_other.isin(cemment_with_bricks), "house_material"
    ] = "cement bonded bricks/stone"

    df["house_material"] = np.where(
        df.material_other.str.contains("mato ghar|clay|mato|mud", case=False, na=False),
        "clay",
        df["house_material"],
    )

    df.drop(
        ["material_other", "previous_nfri", "ethnicity", "ethnicity_others"],
        axis=1,
        inplace=True,
    )
    df.fillna(0, inplace=True)
    df["respondent_age"] = df["respondent_age"].astype(str).map(lambda x: x.strip())


# %%
def feature_creation(df):
    df.insert(3, "household_size", df.iloc[:, 5:25].sum(axis=1))
    df.insert(4, "total_male", df.iloc[:, 6:16].sum(axis=1))
    df.insert(
        5,
        "percent_non_male",
        ((df.household_size - df.total_male) / df.household_size) * 100,
    )
    df.drop(["total_male"], axis=1, inplace=True)
    df.insert(5, "children_under_5", df.iloc[:, [7, 17]].sum(axis=1))
    df["children_under_5"] = np.where(df.children_under_5 > 0, 1, 0)
    df.insert(
        6,
        "income_gen_ratio",
        ((df.income_generating_members / df.household_size) * 100),
    )
    df.insert(7, "health_difficulty", df.iloc[:, [29, 30, 31, 32, 33, 34]].sum(axis=1))
    df["respondent_female"] = np.where(df.respondent_gender == "female", 1, 0)
    df["sindupalchowk"] = np.where(df.district == "sindupalchok", 1, 0)
    df.income_gen_ratio = df.income_gen_ratio.replace(np.inf, np.nan)
    df.fillna(0, inplace=True)


# %%
