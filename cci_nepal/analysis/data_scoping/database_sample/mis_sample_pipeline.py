# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Import libraries
import cci_nepal
import pandas as pd
import numpy as np
import re


# %%
# Function to clean dataset
def clean_df_columns(df):
    """Removes Nepalese text and strips white space from headers and values."""
    df.columns = df.columns.str.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)
    df.columns = df.columns.str.lstrip()
    df = df.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)
    return df


# %%
def group_counts(start, end, question, df):
    """Group and sum count columns of survey questions to display in plot."""
    value_counts = {}  # Create empty array
    count_df = df.iloc[:, start:end]  # Slice df to just counts
    count_df.columns = count_df.columns.str.replace(
        question, ""
    )  # Remove question text
    count_df.columns = count_df.columns.str.strip()  # Strip white space
    # Loop through columns creating sum of counts
    for col in count_df.columns:
        count_df[col] = count_df[col].fillna(0)
        value_counts[col] = sum(count_df[col])
    return value_counts  # Return counts
