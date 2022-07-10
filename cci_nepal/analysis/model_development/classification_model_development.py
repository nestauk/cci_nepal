# -*- coding: utf-8 -*-
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
# # Feature selection and model tuning pipeline

# %%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle
from pathlib import Path

# Project libraries
import cci_nepal
from cci_nepal.getters import get_data as grd
from cci_nepal.pipeline import data_manipulation as dm
from cci_nepal.pipeline import model_tuning_report as mtr

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
# Lowercase values
train = train.applymap(lambda s: s.lower() if type(s) == str else s)
val = val.applymap(lambda s: s.lower() if type(s) == str else s)

# %%
# Items, basic and non-basic divide
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]


# %%
# Data transformations and feature creation
dm.transform_sets(train, column_names)
dm.transform_sets(val, column_names)
dm.feature_creation(train)
dm.feature_creation(val)

# %%
# X and Y split
X_train = train[select_features]
X_val = val[select_features]
y_train = train[nfri_items]
y_val = val[nfri_items]

# %%
# Preferences to numbers
y_train = dm.nfri_preferences_to_binary(y_train)
y_val = dm.nfri_preferences_to_binary(y_val)

# %%
# Split basic /non basic
y_train_basic = y_train[basic]
y_val_basic = y_val[basic]
y_train_non_basic = y_train[non_basic]
y_val_non_basic = y_val[non_basic]

# %%
# Define the transformations to be made
# transformer = mtr.col_transformer()


# %%
# Models
logr = MultiOutputClassifier(LogisticRegression(solver="liblinear"))
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(random_state=1)
dt = DecisionTreeClassifier(random_state=1)
nb = MultiOutputClassifier(GaussianNB(), n_jobs=-1)
svm = MultiOutputClassifier(SVC(), n_jobs=-1)

# %%
# Define pipelines
pipe_lr = Pipeline(
    steps=[("pre_proc", mtr.col_transformer(None)), ("logistic", logr)]
)  # Logistic

pipe_knn = Pipeline(
    steps=[("pre_proc", mtr.col_transformer("first")), ("knn", knn)]
)  # KNN

pipe_rf = Pipeline(steps=[("pre_proc", mtr.col_transformer("first")), ("rf", rf)])  # RF

pipe_dt = Pipeline(steps=[("pre_proc", mtr.col_transformer("first")), ("dt", dt)])  # DT

pipe_nb = Pipeline(steps=[("pre_proc", mtr.col_transformer("first")), ("nb", nb)])  # NB

pipe_svm = Pipeline(
    steps=[("pre_proc", mtr.col_transformer("first")), ("svm", svm)]
)  # SVM

# %%
# Save pipes to list
pipes = [pipe_lr, pipe_knn, pipe_rf, pipe_dt, pipe_nb, pipe_svm]

# %%
# Running test_all_models function to get results
# To note: this can take a long time to run
# #%%capture
results_basic, results_non_basic = mtr.test_all_models(
    pipes,
    "f1_micro",
    X_train,
    y_train_basic,
    y_train_non_basic,
)

# Add folder if not already created
Path(f"{project_dir}/outputs/data/model_results/").mkdir(parents=True, exist_ok=True)

# %%
# Save results to outputs/data/model_results
with open(
    f"{project_dir}/outputs/data/model_results/features_params_scores_wash_filtered_features.pkl",
    "wb",
) as f:
    pickle.dump(results_non_basic, f)

with open(
    f"{project_dir}/outputs/data/model_results/features_params_scores_shelter_filtered_features.pkl",
    "wb",
) as f:
    pickle.dump(results_basic, f)

# %%
