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
import pandas as pd
import cci_nepal


# %%
# Open and read excel file
def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df
 


# %%
# Read CSV file
def read_csv_file(file_path):
    return pd.read_csv(file_path)


# %%
 
 
