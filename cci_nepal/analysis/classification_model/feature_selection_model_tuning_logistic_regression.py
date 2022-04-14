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
#       jupytext_version: 1.13.5
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
from matplotlib import pyplot as plt
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

# Project libraries
import cci_nepal
from cci_nepal.getters.real_data import get_real_data as grd
from cci_nepal.pipeline.real_data import data_manipulation as dm

from cci_nepal.pipeline.classification_model import model_tuning_report as mtr

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
transformer = ColumnTransformer(
    transformers=[
        (
            "rob_scaler",
            RobustScaler(),
            [
                "household_size",
                "percent_female",
                "income_gen_ratio",
                "income_gen_adults",
            ],
        ),
        (
            "one_hot",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            ["Ethnicity", "House_Material"],
        ),
    ],
    remainder="passthrough",
)

# %%
# %%capture
# Sequential feature selector
sfs_selector = SequentialFeatureSelector(
    estimator=MultiOutputClassifier(LogisticRegression()),
    n_features_to_select=18,
    cv=2,
    direction="backward",
)

# %%
# Models
# To note: solver='liblinear' - research solvers for best option
logr = MultiOutputClassifier(LogisticRegression(solver="liblinear"))
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(random_state=1)
dt = DecisionTreeClassifier(random_state=1)
nb = MultiOutputClassifier(GaussianNB(), n_jobs=-1)
svm = MultiOutputClassifier(SVC(), n_jobs=-1)

# %%
# Define pipelines
pipe_lr = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("logistic", logr)]
)  # Logistic

pipe_knn = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("knn", knn)]
)  # KNN

pipe_rf = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("rf", rf)]
)  # RF

pipe_dt = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("dt", dt)]
)  # DT

pipe_nb = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("nb", nb)]
)  # NB

pipe_svm = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("svm", svm)]
)  # SVM

# %%
pipes = [pipe_lr, pipe_knn, pipe_rf, pipe_dt, pipe_nb, pipe_svm]

# %%
# %%capture

params_lr = {"selector__n_features_to_select": [2, 5, 10, 15, 17]}
model_fit_lr_basic = mtr.perform_grid_search(pipe_lr, "f1_micro", params_lr).fit(
    X_train, y_train_basic
)
model_fit_lr_non_basic = mtr.perform_grid_search(pipe_lr, "f1_micro", params_lr).fit(
    X_train, y_train_non_basic
)

best_estimator_basic = model_fit_lr_basic.best_estimator_
best_estimator_non_basic = model_fit_lr_non_basic.best_estimator_

# %%
with open(
    f"{project_dir}/outputs/data/model_results/best_estimator_basic.pkl", "wb"
) as f:
    pickle.dump(best_estimator_basic, f)

with open(
    f"{project_dir}/outputs/data/model_results/best_estimator_non_basic", "wb"
) as f:
    pickle.dump(best_estimator_non_basic, f)
