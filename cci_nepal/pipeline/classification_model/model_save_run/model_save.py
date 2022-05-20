# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# %% [markdown]
# # Model save
# <br>
# <br>
# This script refits the best performing model types and parameters found after feature selection* and model tuning work (see the `analysis/model_development` folder for the python scripts that detail this work) and saves the resulting models to disk.
#
# These models are then used in the `model_run.py` file to predict on the test set.
#
# *To choose the final set of features we considered the results from feature selection methods (sequential feature selector using grid search across different model types and parameters) as well as the preferences of Red Cross members through evaluation and feedback workshops.

# %%
# Import libraries
import pandas as pd
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

# Project libraries
import cci_nepal
from cci_nepal.pipeline.classification_model import model_tuning_report as mtr
from cci_nepal.getters.classification_model import get_data as grd
from cci_nepal.pipeline.classification_model import data_manipulation as dm
from cci_nepal import config

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# Read data and feature names
train = grd.read_train_data()
val = grd.read_val_data()
column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")
select_features = grd.get_lists(f"{project_dir}/cci_nepal/config/select_features.csv")

# %%
train.shape

# %%
# Get parameters from config file
b_features = config["final_model"]["model_features"]
lr_solver = config["final_model"]["solver"]
b_penalty = config["final_model"]["penalty_basic"]
nb_penalty = config["final_model"]["penalty_non_basic"]

# %%
# Combine training and validation sets
train = pd.concat([train, val])
train = shuffle(train, random_state=1)

# %%
# Lowercase values
train = train.applymap(lambda s: s.lower() if type(s) == str else s)

# %%
# Items, basic and non-basic divide
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]

# %%
train.shape

# %%
# Data transformations and feature creation
dm.transform_sets(train, column_names)
dm.feature_creation(train)

# %%
# X and Y split
X_train = train[select_features]
y_train = train[nfri_items]


# %%
# Preferences to binary
y_train = dm.nfri_preferences_to_binary(y_train)

# %%
# Split basic /non basic
y_train_basic = y_train[basic]
y_train_non_basic = y_train[non_basic]

# %%
# Define the transformations to be made
transformer = mtr.col_transformer(None)

# %%
# Best performing model
logr_b = MultiOutputClassifier(LogisticRegression(solver=lr_solver, penalty=b_penalty))
logr_nb = MultiOutputClassifier(
    LogisticRegression(solver=lr_solver, penalty=nb_penalty)
)

# %%
# Apply column transformer
X_train = transformer.fit_transform(X_train)


# %%
# Assign back to dataframes - to have feature names back
X_train = pd.DataFrame(X_train, columns=list(transformer.get_feature_names_out()))


# %%
# Reduce to just chosen features for basic and non-basic
X_train_basic = X_train[b_features].copy()
X_train_non_basic = X_train[b_features].copy()

# %%
# Add folder if not already created
Path(f"{project_dir}/outputs/models/").mkdir(parents=True, exist_ok=True)

# %%
# Fit model
basic_model = logr_b.fit(X_train_basic, y_train_basic)
# Save model to disk
filename = f"{project_dir}/outputs/models/final_classification_model_basic.sav"
pickle.dump(basic_model, open(filename, "wb"))

# Fit model
non_basic_model = logr_nb.fit(X_train_non_basic, y_train_non_basic)
# Save model to disk
filename = f"{project_dir}/outputs/models/final_classification_model_non_basic.sav"
pickle.dump(non_basic_model, open(filename, "wb"))
