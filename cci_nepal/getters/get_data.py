# %%
# Import libraries
import pandas as pd
import csv


# %%
# Read Excel file
def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df


# %%
# Read CSV file
def read_csv_file(file_path):
    return pd.read_csv(file_path)


# %%
def get_lists(file):
    results = []
    with open(file, newline="") as inputfile:
        for row in csv.reader(inputfile):
            results.append(row[0])
    return results


# %%
