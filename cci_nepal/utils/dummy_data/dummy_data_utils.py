# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# %load_ext nb_black


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


# +
def replace_column_with_number(df, columns, values):

    """

    Takes Pandas Dataframe Columns and fills all the values in the selected columns with selected Numerical values.

    """

    for one_column, one_value in zip(columns, values):
        df[one_column] = one_value
