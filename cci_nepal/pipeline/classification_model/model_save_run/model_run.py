# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# %% [markdown]
# # Model run
# <br>
# <br>
# The purpose of this script is to run the x2 classification models (basic and non-basic NFRI items) on new data (defaulted to the test set) and save the results to excel files. The resulting files are:
#
# - `basic_test_predictions.xlsx`
# - `non_basic_test_predictions.xlsx`
#
# The classification models are loaded from saved pretrained models which are created in the `model_save.py` file in `pipeline/classification_model/`.
#
# This script relies on test datasets (one for Hill and one for Terai regions) which are subsets of the survey data collected in Mahottari and Sindhupalchok. The splitting of the data is done in the data_splitting_survey.py file in `pipeline/classification_model/`.

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

# Project libraries
import cci_nepal
from cci_nepal.getters.classification_model import get_real_data as grd
from cci_nepal.pipeline.classification_model import data_manipulation as dm
from cci_nepal.pipeline.classification_model import model_tuning_report as mtr
from cci_nepal import config

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# Get parameters from config file
b_features = config["final_model"]["basic_model_features"]
nb_features = config["final_model"]["non_basic_model_features"]

# %%
# Read data and feature names
test_hill = grd.read_test_hill_data()
test_terai = grd.read_test_terai_data()
column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")
select_features = grd.get_lists(f"{project_dir}/cci_nepal/config/select_features.csv")

# %%
# Combine test sets and shuffle
test = pd.concat([test_hill, test_terai], ignore_index=True)
test = shuffle(test, random_state=1)

# %%
# Lowercase values
test = test.applymap(lambda s: s.lower() if type(s) == str else s)

# %%
# Data transformations and feature creation
dm.transform_sets(test, column_names)
dm.feature_creation(test)

# %%
# X split
X_test = test[select_features]

# %%
# Define the transformations to be made
transformer = mtr.col_transformer()

# %%
# Apply column transformer
X_test_transform = transformer.fit_transform(X_test)

# %%
# Assign back to dataframes - to have feature names back
X_test_transform = pd.DataFrame(
    X_test_transform, columns=list(transformer.get_feature_names_out())
)

# %%
# Reduce to just chosen features for basic and non-basic
X_test_basic = X_test_transform[b_features].copy()
X_test_non_basic = X_test_transform[nb_features].copy()

# %%
# Loading models (best performing)
basic_model = pickle.load(
    open(f"{project_dir}/outputs/models/final_classification_model_basic.sav", "rb")
)

non_basic_model = pickle.load(
    open(f"{project_dir}/outputs/models/final_classification_model_non_basic.sav", "rb")
)

# %%
# Predict on test set
y_pred_basic = basic_model.predict_proba(X_test_basic)
y_pred_non_basic = non_basic_model.predict_proba(X_test_non_basic)

# %%
# Items, basic and non-basic divide
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]

# %%
# Create dataframes with model inputs and predictions for basic and non basic
basic_preds = mtr.create_predictions_files(
    y_pred_basic,
    basic,
    X_test,
    X_test.columns[~X_test.columns.isin(["income_gen_ratio", "Ethnicity"])],
)

non_basic_preds = mtr.create_predictions_files(
    y_pred_non_basic, non_basic, X_test, ["sindupalchowk", "children"]
)

# %%
# Save files
basic_preds.to_excel(
    f"{project_dir}/outputs/data/basic_test_predictions.xlsx", index=False
)
non_basic_preds.to_excel(
    f"{project_dir}/outputs/data/non_basic_test_predictions.xlsx", index=False
)
