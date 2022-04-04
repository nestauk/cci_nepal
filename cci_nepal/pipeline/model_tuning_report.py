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

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import itertools
import cci_nepal

# Set directory
project_directory = cci_nepal.PROJECT_DIR


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
def get_pipeline_results(search, y_type, X_train):
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
def test_all_models(
    pipes, score, fs_param_name, fs_params, X_train, y_train_basic, y_train_non_basic
):
    """
    Run pipelines for all models based on different scores and feature selection methods.
    """

    # Logistic regression
    param_grid_lr = {
        "selector__" + fs_param_name: fs_params,
        "logistic__estimator__penalty": ["l1", "l2"],
    }
    search_lr = perform_grid_search(pipes[0], "f1_micro", param_grid_lr)
    bs_lr_non_basic, bp_lr_non_basic, ftk_lr_non_basic = get_pipeline_results(
        search_lr, y_train_non_basic, X_train
    )
    bs_lr_basic, bp_lr_basic, ftk_lr_basic = get_pipeline_results(
        search_lr, y_train_basic, X_train
    )

    # KNN
    param_grid_knn = {
        "selector__" + fs_param_name: fs_params,
        "knn__n_neighbors": [2, 5, 10, 50],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    search_knn = perform_grid_search(pipes[1], "f1_micro", param_grid_knn)
    bs_knn_non_basic, bp_knn_non_basic, ftk_knn_non_basic = get_pipeline_results(
        search_knn, y_train_non_basic, X_train
    )
    bs_knn_basic, bp_knn_basic, ftk_knn_basic = get_pipeline_results(
        search_knn, y_train_basic, X_train
    )

    # RF
    param_grid_rf = {
        "selector__" + fs_param_name: fs_params,
        "rf__n_estimators": [10, 50, 100, 200],
    }
    search_rf = perform_grid_search(pipes[2], "f1_micro", param_grid_rf)
    bs_rf_non_basic, bp_rf_non_basic, ftk_rf_non_basic = get_pipeline_results(
        search_rf, y_train_non_basic, X_train
    )
    bs_rf_basic, bp_rf_basic, ftk_rf_basic = get_pipeline_results(
        search_rf, y_train_basic, X_train
    )

    # DT
    param_grid_dt = {
        "selector__" + fs_param_name: fs_params,
        "dt__criterion": ["gini", "entropy"],
    }
    search_dt = perform_grid_search(pipes[3], "f1_micro", param_grid_dt)
    bs_dt_non_basic, bp_dt_non_basic, ftk_dt_non_basic = get_pipeline_results(
        search_dt, y_train_non_basic, X_train
    )
    bs_dt_basic, bp_dt_basic, ftk_dt_basic = get_pipeline_results(
        search_dt, y_train_basic, X_train
    )

    # NB
    param_grid_nb = {
        "selector__" + fs_param_name: fs_params,
    }
    search_nb = perform_grid_search(pipes[4], "f1_micro", param_grid_nb)
    bs_nb_non_basic, bp_nb_non_basic, ftk_nb_non_basic = get_pipeline_results(
        search_nb, y_train_non_basic, X_train
    )
    bs_nb_basic, bp_nb_basic, ftk_nb_basic = get_pipeline_results(
        search_nb, y_train_basic, X_train
    )

    # SVM
    param_grid_svm = {
        "selector__" + fs_param_name: fs_params,
        "svm__estimator__C": [1, 2, 3, 4],
        "svm__estimator__gamma": ["scale", "auto"],
    }
    search_svm = perform_grid_search(pipes[5], "f1_micro", param_grid_svm)
    bs_svm_non_basic, bp_svm_non_basic, ftk_svm_non_basic = get_pipeline_results(
        search_svm, y_train_non_basic, X_train
    )
    bs_svm_basic, bp_svm_basic, ftk_svm_basic = get_pipeline_results(
        search_svm, y_train_basic, X_train
    )

    all_scores_basic = {
        "logistic": [bs_lr_basic, bp_lr_basic, ftk_lr_basic],
        "knn": [bs_knn_basic, bp_knn_basic, ftk_knn_basic],
        "rf": [bs_rf_basic, bp_rf_basic, ftk_rf_basic],
        "dt": [bs_dt_basic, bp_dt_basic, ftk_dt_basic],
        "nb": [bs_nb_basic, bp_nb_basic, ftk_nb_basic],
        "svm": [bs_svm_basic, bp_svm_basic, ftk_svm_basic],
    }

    all_scores_non_basic = {
        "logistic": [bs_lr_non_basic, bp_lr_non_basic, ftk_lr_non_basic],
        "knn": [bs_knn_non_basic, bp_knn_non_basic, ftk_knn_non_basic],
        "rf": [bs_rf_non_basic, bp_rf_non_basic, ftk_rf_non_basic],
        "dt": [bs_dt_non_basic, bp_dt_non_basic, ftk_dt_non_basic],
        "nb": [bs_nb_non_basic, bp_nb_non_basic, ftk_nb_non_basic],
        "svm": [bs_svm_non_basic, bp_svm_non_basic, ftk_svm_non_basic],
    }
    return all_scores_basic, all_scores_non_basic


# %%

# %%
