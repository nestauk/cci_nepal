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
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
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

    all_scores = {
        "linear": [bs_lr_basic, bp_lr_basic, ftk_lr_basic],
        "ridge": [bs_rr_basic, bp_rr_basic, ftk_rr_basic],
        "lasso": [bs_ls_basic, bp_ls_basic, ftk_ls_basic],
        "decision_tree": [bs_dt_basic, bp_dt_basic, ftk_dt_basic],
        "random_forest": [bs_rf_basic, bp_rf_basic, ftk_rf_basic],
    }
    return all_scores


# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# Read data and feature names
train = pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/train.csv")
val = pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/val.csv")
column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")
select_features = grd.get_lists(f"{project_dir}/cci_nepal/config/select_features.csv")

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
        ("one_hot", OneHotEncoder(drop="first"), ["Ethnicity", "House_Material"]),
    ],
    remainder="passthrough",
)

# %%
# Models
linear = MultiOutputRegressor(LinearRegression())
regr = MultiOutputRegressor(Ridge(random_state=42))
lasso = MultiOutputRegressor(Lasso(random_state=42))
decision_tree = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
random_forest = MultiOutputRegressor(RandomForestRegressor(random_state=42))

# %%
# %%capture
## Define feature selector methods ##

# Sequential feature selector
sfs_selector = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select=5,
    cv=2,
    direction="backward",
)

# Select from model
sfm_selector = SelectFromModel(estimator=LinearRegression())

# %%
# Define pipelines - sequential feature selector
pipe_lr_sfs = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("linear", linear)]
)  # Linear
pipe_rr_sfs = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("ridge", regr)]
)  # Ridge
pipe_ls_sfs = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfs_selector), ("lasso", lasso)]
)  # Lasso
pipe_dt_sfs = Pipeline(
    steps=[
        ("pre_proc", transformer),
        ("selector", sfs_selector),
        ("dec_tree", decision_tree),
    ]
)  # DT
pipe_rf_sfs = Pipeline(
    steps=[
        ("pre_proc", transformer),
        ("selector", sfs_selector),
        ("ran_for", random_forest),
    ]
)  # RF

# Define pipelines - select from models
pipe_lr_sfm = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfm_selector), ("linear", linear)]
)  # Linear
pipe_rr_sfm = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfm_selector), ("ridge", regr)]
)  # Ridge
pipe_ls_sfm = Pipeline(
    steps=[("pre_proc", transformer), ("selector", sfm_selector), ("lasso", lasso)]
)  # Lasso
pipe_dt_sfm = Pipeline(
    steps=[
        ("pre_proc", transformer),
        ("selector", sfm_selector),
        ("dec_tree", decision_tree),
    ]
)  # DT
pipe_rf_sfm = Pipeline(
    steps=[
        ("pre_proc", transformer),
        ("selector", sfm_selector),
        ("ran_for", random_forest),
    ]
)  # RF

# %%
pipes_sfs = [pipe_lr_sfs, pipe_rr_sfs, pipe_ls_sfs, pipe_dt_sfs, pipe_rf_sfs]
pipes_sfm = [pipe_lr_sfm, pipe_rr_sfm, pipe_ls_sfm, pipe_dt_sfm, pipe_rf_sfm]

# %% [markdown]
# Takes 5-10 minutes to run.

# %%
# %%capture
sfs_r2_scores = test_all_models(
    pipes_sfs, "r2", "n_features_to_select", [2, 5, 10, 15, 19]
)
sfs_mse_scores = test_all_models(
    pipes_sfs, "neg_mean_squared_error", "n_features_to_select", [2, 5, 10, 15, 19]
)

# %%
# %%capture
sfs_r2_scores = test_all_models(
    pipes_sfm, "r2", "threshold", ["median", "mean", "1.25*mean", "0.75*mean"]
)
sfs_mse_scores = test_all_models(
    pipes_sfm,
    "neg_mean_squared_error",
    "threshold",
    ["median", "mean", "1.25*mean", "0.75*mean"],
)

# %% [markdown]
# ### Output scores

# %%
sfs_r2_scores

# %%
