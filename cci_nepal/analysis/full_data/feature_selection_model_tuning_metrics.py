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
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
y_train.head(1)

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
    n_features_to_select=18,
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

sfs_r2_scores_basic, sfs_r2_scores_non_basic = test_all_models(
    pipes_sfs, "r2", "n_features_to_select", [2, 5, 10, 15, 19]
)
sfs_mse_scores_basic, sfs_mse_scores_non_basic = test_all_models(
    pipes_sfs, "neg_mean_squared_error", "n_features_to_select", [2, 5, 10, 15, 17]
)

# %%
# %%capture
sfm_r2_scores_basic, sfm_r2_scores_non_basic = test_all_models(
    pipes_sfm, "r2", "threshold", ["median", "mean", "1.25*mean", "0.75*mean"]
)
sfm_mse_scores_basic, sfm_mse_scores_non_basic = test_all_models(
    pipes_sfm,
    "neg_mean_squared_error",
    "threshold",
    ["median", "mean", "1.25*mean", "0.75*mean"],
)

# %% [markdown]
# ### Output scores - full data

# %%
sfs_r2_scores_basic["linear"][0]

# %% [markdown]
# #### Basic

# %%
r2_sfs = []
for item in sfs_r2_scores_basic.values():
    r2_sfs.append(item[0])

r2_sfm = []
for item in sfm_r2_scores_basic.values():
    r2_sfm.append(item[0])

# %%
mse_sfs = []
for item in sfs_mse_scores_basic.values():
    mse_sfs.append(item[0])

mse_sfm = []
for item in sfm_mse_scores_basic.values():
    mse_sfm.append(item[0])

# %%
models = ["linear", "ridge", "lasso", "decision tree", "random forest"]

# %%
zipped = list(zip(models, r2_sfs, r2_sfm, mse_sfs, mse_sfm))
df_results = pd.DataFrame(
    zipped,
    columns=[
        "model type",
        "r2 sequential feature selector",
        "r2 select from model",
        "mse sequential feature selector",
        "mse select from model",
    ],
)

# %%
df_results.set_index("model type", inplace=True)

# %%
df_results["mse select from model"] = df_results["mse select from model"].abs()
df_results["mse sequential feature selector"] = df_results[
    "mse sequential feature selector"
].abs()

# %%
df_results

# %%
plt.rcParams["axes.facecolor"] = "white"

# %%
df_results[["r2 sequential feature selector", "r2 select from model"]].plot()

# %%
df_results[["mse sequential feature selector", "mse select from model"]].plot()

# %%
sfs_r2_scores_basic

# %%
sfs_mse_scores_basic

# %%
sfm_r2_scores_basic

# %%
sfm_mse_scores_basic

# %% [markdown]
# #### Non-basic

# %%
r2_sfs = []
for item in sfs_r2_scores_non_basic.values():
    r2_sfs.append(item[0])

r2_sfm = []
for item in sfm_r2_scores_non_basic.values():
    r2_sfm.append(item[0])

# %%
mse_sfs = []
for item in sfs_mse_scores_non_basic.values():
    mse_sfs.append(item[0])

mse_sfm = []
for item in sfm_mse_scores_non_basic.values():
    mse_sfm.append(item[0])

# %%
models = ["linear", "ridge", "lasso", "decision tree", "random forest"]

# %%
zipped = list(zip(models, r2_sfs, r2_sfm, mse_sfs, mse_sfm))
df_results = pd.DataFrame(
    zipped,
    columns=[
        "model type",
        "r2 sequential feature selector",
        "r2 select from model",
        "mse sequential feature selector",
        "mse select from model",
    ],
)

# %%
df_results.set_index("model type", inplace=True)

# %%
df_results["mse select from model"] = df_results["mse select from model"].abs()
df_results["mse sequential feature selector"] = df_results[
    "mse sequential feature selector"
].abs()

# %%
df_results

# %%
plt.rcParams["axes.facecolor"] = "white"

# %%
df_results[["r2 sequential feature selector", "r2 select from model"]].plot()

# %%
df_results[["mse sequential feature selector", "mse select from model"]].plot()

# %%
sfs_r2_scores_non_basic

# %%
sfs_mse_scores_basic

# %%
sfm_r2_scores_basic

# %%
sfm_mse_scores_basic

# %% [markdown]
# ### Test final results on validation set

# %%
# Apply column transformer
X_train = transformer.fit_transform(X_train)
X_val = transformer.transform(X_val)

# %%
# Assign back to dataframes - to have feature names back
X_train = pd.DataFrame(X_train, columns=list(transformer.get_feature_names_out()))
X_val = pd.DataFrame(X_val, columns=list(transformer.get_feature_names_out()))

# %%
X_val.head(1)

# %% [markdown]
# #### Best performing from train - random forest, select from model

# %%
random_forest = MultiOutputRegressor(
    RandomForestRegressor(random_state=42, n_estimators=100)
)

# %%
random_forest.fit(
    X_train[
        [
            "one_hot__Ethnicity_madhesi",
            "one_hot__Ethnicity_other",
            "one_hot__Ethnicity_prefer_not_to_answer",
            "one_hot__House_Material_mud_bonded_bricks_stone",
            "one_hot__House_Material_other",
            "one_hot__House_Material_rcc_with_pillar",
            "one_hot__House_Material_wooden_pillar",
            "remainder__previous_nfri",
            "remainder__sindupalchowk",
        ]
    ],
    y_train_basic,
)

# %%
rf_predictions = random_forest.predict(
    X_val[
        [
            "one_hot__Ethnicity_madhesi",
            "one_hot__Ethnicity_other",
            "one_hot__Ethnicity_prefer_not_to_answer",
            "one_hot__House_Material_mud_bonded_bricks_stone",
            "one_hot__House_Material_other",
            "one_hot__House_Material_rcc_with_pillar",
            "one_hot__House_Material_wooden_pillar",
            "remainder__previous_nfri",
            "remainder__sindupalchowk",
        ]
    ]
)

# %%
from sklearn.metrics import r2_score

# %%
r2_score(y_val_basic, rf_predictions)

# %% [markdown]
# #### Best performing linear regression - sequential feature selector

# %%
linear = MultiOutputRegressor(LinearRegression())

# %%
transformer.get_feature_names_out()

# %%
linear_chosen = [
    "rob_scaler__household_size",
    "rob_scaler__percent_female",
    "rob_scaler__income_gen_ratio",
    "rob_scaler__income_gen_adults",
    "one_hot__Ethnicity_dalit",
    "one_hot__Ethnicity_madhesi",
    "one_hot__Ethnicity_other",
    "one_hot__Ethnicity_prefer_not_to_answer",
    "one_hot__House_Material_mud_bonded_bricks_stone",
    "one_hot__House_Material_other",
    "one_hot__House_Material_wooden_pillar",
    "remainder__children",
    "remainder__children_under_5",
    "remainder__previous_nfri",
    "remainder__sindupalchowk",
]

# %%
linear.fit(X_train[linear_chosen], y_train_basic)

# %%
lr_predictions = linear.predict(X_val[linear_chosen])

# %%
r2_score(y_val_basic, lr_predictions)

# %% [markdown]
# #### Classification results

# %%
df_basic_acc, predictions_basic = accuracy_per_item(
    linear, "nfri-basic", X_val[linear_chosen], y_val_basic
)

# %%
df_basic_acc.sort_values(by="Accuracy", ascending=True).plot(kind="barh")
plt.title("Accuracy per item - NFRI basic")

# %%
len(predictions_basic[0])

# %%
np.array(predictions_basic[0])

# %%
y_val_basic.columns

# %%
cm = confusion_matrix(
    list(y_val_basic[y_val_basic.columns[10]]), predictions_basic[10], labels=[1, 2, 3]
)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
disp.plot()
plt.title(y_val_basic.columns[10])
plt.show()

# %%
for i in range(0, 11):
    print(
        f1_score(
            y_val_basic[y_val_basic.columns[i]], predictions_basic[i], average="macro"
        )
    )

# %% [markdown]
# ### New non-basic

# %%
linear_chosen = [
    "rob_scaler__household_size",
    "rob_scaler__percent_female",
    "rob_scaler__income_gen_ratio",
    "rob_scaler__income_gen_adults",
    "one_hot__Ethnicity_dalit",
    "one_hot__Ethnicity_madhesi",
    "one_hot__Ethnicity_other",
    "one_hot__Ethnicity_prefer_not_to_answer",
    "one_hot__House_Material_mud_bonded_bricks_stone",
    "one_hot__House_Material_other",
    "one_hot__House_Material_wooden_pillar",
    "remainder__children",
    "remainder__children_under_5",
    "remainder__previous_nfri",
    "remainder__sindupalchowk",
]

# %%
linear.fit(X_train[linear_chosen], y_train_non_basic)

# %%
lr_predictions = linear.predict(X_val[linear_chosen])

# %%
r2_score(y_val_non_basic, lr_predictions)

# %%
df_non_basic_acc, predictions_non_basic = accuracy_per_item(
    linear, "nfri-non-basic", X_val[linear_chosen], y_val_non_basic
)

# %%
df_non_basic_acc

# %%
cm = confusion_matrix(
    list(y_val_non_basic[y_val_non_basic.columns[0]]),
    predictions_non_basic[0],
    labels=[1, 2, 3],
)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
disp.plot()
plt.title(y_val_non_basic.columns[10])
plt.show()

# %%
for i in range(0, 11):
    print(
        f1_score(
            y_val_non_basic[y_val_non_basic.columns[i]],
            predictions_non_basic[i],
            average="macro",
        )
    )

# %%
for i in range(0, 11):
    print(
        f1_score(
            y_val_non_basic[y_val_non_basic.columns[i]],
            predictions_non_basic[i],
            average="micro",
        )
    )
