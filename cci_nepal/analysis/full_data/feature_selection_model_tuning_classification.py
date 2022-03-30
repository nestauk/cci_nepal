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

# Project libraries
import cci_nepal
from cci_nepal.getters.real_data import get_real_data as grd
from cci_nepal.pipeline.real_data import data_manipulation as dm


# %%
def perform_grid_search(pipe, score, parameter_grid):
    """
    Setting parameters for GridSearchCV.
    """
    search = GridSearchCV(
        estimator=pipe,
        param_grid=parameter_grid,
        n_jobs=-1,
        scoring=score,
        cv=10,
        refit=True,
        verbose=3,
    )
    return search


# %%
def get_pipeline_results(search, y_type):
    """
    Fit pipeline to training set and collect results.
    """
    search.fit(X_train, y_type)
    best_score = search.best_score_
    best_params = search.best_params_
    rob_scal_feats = list(
        search.best_estimator_.named_steps["pre_proc"]
        .named_transformers_["rob_scaler"]
        .get_feature_names_out()
    )
    one_hot_feats = list(
        search.best_estimator_.named_steps["pre_proc"]
        .named_transformers_["one_hot"]
        .get_feature_names_out()
    )
    features = list(
        itertools.chain(
            rob_scal_feats,
            one_hot_feats,
            [
                "children",
                "children_under_5",
                "health_difficulty",
                "previous_nfri",
                "sindupalchowk",
            ],
        )
    )
    index_keep = list(search.best_estimator_.named_steps["selector"].get_support(True))
    features_to_keep = [features[i] for i in index_keep]
    return best_score, best_params, features_to_keep


# %%
def test_all_models(pipes, score, fs_param_name, fs_params):
    """
    Run pipelines for all models based on different scores and feature selection methods.
    """

    # Linear
    param_grid_lr = {
        "selector__" + fs_param_name: fs_params,
    }
    search_lr = perform_grid_search(pipes[0], score, param_grid_lr)
    bs_lr_non_basic, bp_lr_non_basic, ftk_lr_non_basic = get_pipeline_results(
        search_lr, y_train_non_basic
    )
    bs_lr_basic, bp_lr_basic, ftk_lr_basic = get_pipeline_results(
        search_lr, y_train_basic
    )

    # Ridge
    param_grid_rr = {
        "selector__" + fs_param_name: fs_params,
        "ridge__estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100],
    }
    search_rr = perform_grid_search(pipes[1], score, param_grid_rr)
    bs_rr_non_basic, bp_rr_non_basic, ftk_rr_non_basic = get_pipeline_results(
        search_rr, y_train_non_basic
    )
    bs_rr_basic, bp_rr_basic, ftk_rr_basic = get_pipeline_results(
        search_rr, y_train_basic
    )

    # Lasso
    param_grid_ls = {
        "selector__" + fs_param_name: fs_params,
        "lasso__estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100],
    }
    search_ls = perform_grid_search(pipes[2], score, param_grid_ls)
    bs_ls_non_basic, bp_ls_non_basic, ftk_ls_non_basic = get_pipeline_results(
        search_ls, y_train_non_basic
    )
    bs_ls_basic, bp_ls_basic, ftk_ls_basic = get_pipeline_results(
        search_ls, y_train_basic
    )

    # Decision tree
    param_grid_dt = {
        "selector__" + fs_param_name: fs_params,
        "dec_tree__estimator__max_depth": [2, 4, 6, 8, 10, 12],
    }
    search_dt = perform_grid_search(pipes[3], score, param_grid_dt)
    bs_dt_non_basic, bp_dt_non_basic, ftk_dt_non_basic = get_pipeline_results(
        search_dt, y_train_non_basic
    )
    bs_dt_basic, bp_dt_basic, ftk_dt_basic = get_pipeline_results(
        search_dt, y_train_basic
    )

    # Random Forest
    param_grid_rf = {
        "selector__" + fs_param_name: fs_params,
        "ran_for__estimator__n_estimators": [50, 100, 200],
    }
    search_rf = perform_grid_search(pipes[4], score, param_grid_rf)
    bs_rf_non_basic, bp_rf_non_basic, ftk_rf_non_basic = get_pipeline_results(
        search_rf, y_train_non_basic
    )
    bs_rf_basic, bp_rf_basic, ftk_rf_basic = get_pipeline_results(
        search_rf, y_train_basic
    )

    all_scores_basic = {
        "linear": [bs_lr_basic, bp_lr_basic, ftk_lr_basic],
        "ridge": [bs_rr_basic, bp_rr_basic, ftk_rr_basic],
        "lasso": [bs_ls_basic, bp_ls_basic, ftk_ls_basic],
        "decision_tree": [bs_dt_basic, bp_dt_basic, ftk_dt_basic],
        "random_forest": [bs_rf_basic, bp_rf_basic, ftk_rf_basic],
    }

    all_scores_non_basic = {
        "linear": [bs_lr_non_basic, bp_lr_non_basic, ftk_lr_non_basic],
        "ridge": [bs_rr_non_basic, bp_rr_non_basic, ftk_rr_non_basic],
        "lasso": [bs_ls_non_basic, bp_ls_non_basic, ftk_ls_non_basic],
        "decision_tree": [bs_dt_non_basic, bp_dt_non_basic, ftk_dt_non_basic],
        "random_forest": [bs_rf_non_basic, bp_rf_non_basic, ftk_rf_non_basic],
    }
    return all_scores_basic, all_scores_non_basic


# %%
def accuracy_per_item(model, nfri_type, test_input, test_output):

    """
    Takes the test input and output and outputs the accuracy of the Model per NFRI item.

    Parameters:

    model: Trained Machine Learning model
    nfri_type: Type of NFRI to predict (basic-nfri or non-basic-nfri)
    test_input: The test set input dataframe
    test_output: The test set output dataframe

    Returns:

    A pandas dataframe with accuracy per NFRI item
    """

    if nfri_type == "nfri-basic":
        nfri_list = basic
    elif nfri_type == "nfri-non-basic":
        nfri_list = non_basic
    else:
        return print("Please enter the correct nfri type.")

    test_prediction = model.predict(test_input)
    test_prediction_label = [
        [dm.numeric_score_transformer(i) for i in nested] for nested in test_prediction
    ]
    test_prediction_label_transformed = list(map(list, zip(*test_prediction_label)))
    accuracy_list = []
    for i in range(0, len(nfri_list)):
        accuracy_list.append(
            accuracy_score(test_prediction_label_transformed[i], test_output.iloc[:, i])
        )

    return (
        pd.DataFrame(accuracy_list, test_output.columns, columns=["Accuracy"]),
        test_prediction_label_transformed,
    )


# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# Read data and feature names
train = pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/new_data/train.csv")
val = pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/new_data/val.csv")
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
y_train = dm.nfri_preferences_to_numbers(y_train)
y_val = dm.nfri_preferences_to_numbers(y_val)

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("dt", knn)]
)  # KNN

# %%
pipes = [pipe_lr, pipe_knn]

# %%
# Logistic regression
param_grid_lr = {
    "selector__" + "n_features_to_select": [2, 5, 10, 15, 17],
    "logistic__estimator__penalty": ["l1", "l2"],
}
search_lr = perform_grid_search(pipe_lr, "f1_micro", param_grid_lr)

# %%
bs_lr_non_basic, bp_lr_non_basic, ftk_lr_non_basic = get_pipeline_results(
    search_lr, y_train_non_basic
)
bs_lr_basic, bp_lr_basic, ftk_lr_basic = get_pipeline_results(search_lr, y_train_basic)

# %%
bs_lr_basic, bp_lr_basic, ftk_lr_basic = get_pipeline_results(search_lr, y_train_basic)

# %%

# %%
# %%capture

f1_scores_basic, f1_scores_non_basic = test_all_models(
    pipes, "f1_micro", "n_features_to_select", [2, 5, 10, 15, 19]
)
