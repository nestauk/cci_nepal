# Import libraries
import pandas as pd

# Set the project directory
import cci_nepal

project_dir = cci_nepal.PROJECT_DIR


def read_complete_data():
    return pd.read_csv(
        f"{project_dir}/inputs/data/free_text_analysis/complete_database.csv"
    )


def read_free_text_activity_data():
    return pd.read_csv(
        f"{project_dir}/inputs/data/free_text_analysis/free_text_activity_combined_result.csv"
    )
