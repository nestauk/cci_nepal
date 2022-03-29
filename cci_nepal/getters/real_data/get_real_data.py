# Import libraries
import pandas as pd

# Set the project directory
import cci_nepal

project_dir = cci_nepal.PROJECT_DIR


def read_complete_data():
    return pd.read_csv(f"{project_dir}/inputs/data/real_data/Complete Database.csv")


def read_free_text_activity_data():
    return pd.read_csv(
        f"{project_dir}/inputs/data/real_data/Free Text Activity - Combined Result.csv"
    )
