# Import libraries
import pandas as pd
import csv

# Set the project directory
import cci_nepal

project_dir = cci_nepal.PROJECT_DIR


def read_train_data():
    return pd.read_csv(f"{project_dir}/inputs/data/exploratory_data_analysis/train.csv")


def get_lists(file):
    results = []
    with open(
        f"{project_dir}/cci_nepal/config/column_names.csv", newline=""
    ) as inputfile:
        for row in csv.reader(inputfile):
            results.append(row[0])
    return results
