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

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import itertools
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Project libraries
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
def accuracy_per_item(model, test_input, test_output):

    """
    Takes the fitted model, test input and output and outputs the accuracy of the Model per NFRI item.

    Parameters:

    model: Trained Machine Learning model
    test_input: The test set input dataframe
    test_output: The test set output dataframe

    Returns:

    A pandas dataframe with accuracy and other evaluation metrics per NFRI item
    """

    test_prediction = model.predict(test_input)
    test_prediction_transformed = list(map(list, zip(*test_prediction)))
    accuracy_list = []
    for i in range(0, test_output.shape[1]):
        accuracy_list.append(
            accuracy_score(test_output.iloc[:, i], test_prediction_transformed[i])
        )

    f1_list_binary = []
    for i in range(0, test_output.shape[1]):
        f1_list_binary.append(
            f1_score(
                test_output.iloc[:, i], test_prediction_transformed[i], average="binary"
            )
        )

    f1_list_macro = []
    for i in range(0, test_output.shape[1]):
        f1_list_macro.append(
            f1_score(
                test_output.iloc[:, i], test_prediction_transformed[i], average="macro"
            )
        )

    f1_list_micro = []
    for i in range(0, test_output.shape[1]):
        f1_list_micro.append(
            f1_score(
                test_output.iloc[:, i], test_prediction_transformed[i], average="micro"
            )
        )

    f1_list_weighted = []
    for i in range(0, test_output.shape[1]):
        f1_list_weighted.append(
            f1_score(
                test_output.iloc[:, i],
                test_prediction_transformed[i],
                average="weighted",
            )
        )

    recall_list = []
    for i in range(0, test_output.shape[1]):
        recall_list.append(
            recall_score(test_output.iloc[:, i], test_prediction_transformed[i])
        )

    precision_list = []
    for i in range(0, test_output.shape[1]):
        precision_list.append(
            precision_score(test_output.iloc[:, i], test_prediction_transformed[i])
        )

    scores_dataframe = pd.DataFrame(
        list(
            zip(
                accuracy_list,
                f1_list_binary,
                f1_list_macro,
                f1_list_micro,
                f1_list_weighted,
                recall_list,
                precision_list,
            )
        ),
        columns=[
            "Accuracy",
            "F1_Score_Binary",
            "F1_Score_Macro",
            "F1_Score_Micro",
            "F1_Score_Weighted",
            "Recall",
            "Precision",
        ],
    )
    scores_dataframe.loc[len(scores_dataframe)] = scores_dataframe.mean(axis=0)
    scores_index = test_output.columns.append(pd.Index(["Average"]))
    scores_dataframe = scores_dataframe.set_index(scores_index)
    return scores_dataframe


# %%
def create_predictions_files(y_pred, nfri_list, X_test, cols_to_include):
    """
    Create files of predictions and input features for basic and non-basic.
    """
    pred_lists = []
    for preds in y_pred:
        pred_list = list(pd.DataFrame(preds)[1])
        pred_lists.append(pred_list)
    pred_df = pd.DataFrame(np.column_stack(pred_lists), columns=nfri_list)

    input_pred_df = pd.concat(
        [
            X_test[X_test.columns[X_test.columns.isin(cols_to_include)]].reset_index(
                drop=True
            ),
            pred_df,
        ],
        axis=1,
    )
    return input_pred_df


# %%
def col_transformer():
    """
    Define the column transformations to be made
    """
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
    return transformer


# %%
def save_cm_plots(cm, model_type, items):
    """
    Save cm plot for each item for chosen model type. Note: model type needs to match folder name in outputs/figures.
    """
    # Loop through items and save cm plot to outputs/figures sub-folder.
    for i in range(0, len(items)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i])
        disp.plot()
        plt.title(items[i].replace("_", " "), pad=20)
        plt.tight_layout()
        plt.savefig(
            f"{project_directory}/outputs/figures/cm/"
            + model_type
            + "/"
            + items[i]
            + "_cm.png",
            bbox_inches="tight",
        )


# %%
