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
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")
from cci_nepal.getters.real_data import get_real_data as grd

# Set project directory
project_dir = cci_nepal.PROJECT_DIR

# Read in df
df = grd.read_csv_file(f"{project_dir}/inputs/data/real_data/Full_Data_District.csv")

# Sort to ensure the order is consistent each time before splitting
df.sort_values(by="_index", inplace=True)

df["District"].value_counts()

df_hill = df[df["District"] == "Sindupalchowk"]
df_terai = df[df["District"] == "Mahottari"]
print(df_hill.shape)
print(df_terai.shape)

train_hill, test_hill = train_test_split(df_hill, test_size=150, random_state=42)
train_terai, test_terai = train_test_split(df_terai, test_size=150, random_state=42)

print(train_hill.shape)
print(test_hill.shape)
print(train_terai.shape)
print(test_terai.shape)

# Write into csv files
train_hill.to_csv(
    f"{project_dir}/outputs/data/data_for_modelling/train_hill.csv", index=False
)
test_hill.to_csv(
    f"{project_dir}/outputs/data/data_for_modelling/test_hill.csv", index=False
)
train_terai.to_csv(
    f"{project_dir}/outputs/data/data_for_modelling/train_terai.csv", index=False
)
test_terai.to_csv(
    f"{project_dir}/outputs/data/data_for_modelling/test_terai.csv", index=False
)

# Split train - train/validation
train_hill, val_hill = train_test_split(train_hill, test_size=0.1, random_state=42)
train_terai, val_terai = train_test_split(train_terai, test_size=0.1, random_state=42)
# Group hill and plain
train = pd.concat([train_hill, train_terai], ignore_index=True)
val = pd.concat([val_hill, val_terai], ignore_index=True)

# Re-shuffle sets
train = shuffle(train, random_state=1)
# Re-shuffle sets
val = shuffle(val, random_state=1)

# Save train and val sets
train.to_csv(f"{project_dir}/outputs/data/data_for_modelling/train.csv", index=False)
val.to_csv(f"{project_dir}/outputs/data/data_for_modelling/val.csv", index=False)
