# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
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

import cci_nepal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import logging
from sklearn.utils import shuffle
from pathlib import Path

from cci_nepal.getters import get_data as grd

# Set project directory
project_dir = cci_nepal.PROJECT_DIR

# Read in df
# df = grd.read_excel_file(f"{project_dir}/inputs/data/survey_data.xlsx")
df = grd.read_complete_data()

# df = grd.read_dummy_data()  # Read this for dummy data

df = df[df.iloc[:, 6:16].sum(axis=1) < 11]
df = df[df.iloc[:, 17:27].sum(axis=1) < 11]

# Sort to ensure the order is consistent each time before splitting
df.sort_values(by="_index", inplace=True)

df["District"].value_counts()

df.shape

df_hill = df[df["District"] == "Sindupalchok"]
df_terai = df[df["District"] == "Mahottari"]


train_hill, test_hill = train_test_split(df_hill, test_size=150, random_state=42)
train_terai, test_terai = train_test_split(df_terai, test_size=150, random_state=42)

# Add folder if not already created
Path(f"{project_dir}/outputs/data/data_for_modelling/").mkdir(
    parents=True, exist_ok=True
)

# Split train - train/validation
train_hill, val_hill = train_test_split(train_hill, test_size=0.1, random_state=42)
train_terai, val_terai = train_test_split(train_terai, test_size=0.1, random_state=42)
# Group hill and plain
train = pd.concat([train_hill, train_terai], ignore_index=True)
val = pd.concat([val_hill, val_terai], ignore_index=True)
test = pd.concat([test_hill, test_terai], ignore_index=True)

# Re-shuffle sets
train = shuffle(train, random_state=1)
# Re-shuffle sets
val = shuffle(val, random_state=1)
# Re-shuffle sets
test = shuffle(test, random_state=1)

train.shape

# Save train and val sets
train.to_csv(f"{project_dir}/outputs/data/data_for_modelling/train.csv", index=False)
val.to_csv(f"{project_dir}/outputs/data/data_for_modelling/val.csv", index=False)
test.to_csv(f"{project_dir}/outputs/data/data_for_modelling/test.csv", index=False)
