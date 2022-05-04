# %%
# Import libraries
import pandas as pd
import csv

# Set the project directory
import cci_nepal

project_dir = cci_nepal.PROJECT_DIR

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


def read_train_data():
    return pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/train.csv")


def read_val_data():
    return pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/val.csv")


def read_test_hill_data():
    return pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/test_hill.csv")


def read_test_terai_data():
    return pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/test_terai.csv")


# %%
