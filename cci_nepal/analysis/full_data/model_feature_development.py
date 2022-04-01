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
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# %%
import pandas as pd
import numpy as np
import itertools

import warnings
import logging

from matplotlib import pyplot as plt
import seaborn as sns

import cci_nepal
from cci_nepal.getters.real_data import get_real_data as grd
from cci_nepal.pipeline.real_data import data_manipulation as dm
from cci_nepal.pipeline.real_data import nfri_list_file as nlf
from cci_nepal.pipeline.dummy_data import model_manipulation as mm

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from imblearn.over_sampling import RandomOverSampler


# %%
def nfri_preferences_to_numbers(df):
    """
    Takes in a dataframe a returns a dataframe with nfri categorical preferences transformed into numbers.
    """
    mapping = {
        "Essential": 3,
        "essential": 3,
        "Desirable": 2,
        "desirable": 2,
        "Unnecessary": 1,
        "unncessary": 1,
        "Essential (अति आवश्यक) ": 3,
        "Desirable (आवश्यक)": 2,
        "Unnecessary (अनावश्यक)": 1,
    }
    return df.applymap(lambda s: mapping.get(s) if s in mapping else s)


# %%
def transform_sets(df, column_names):
    """
    A series of transformations applied to both the train and test including:
    Dropping un-needed columns, renaming columns, remove new nfri items, update house materials,
    dropping 'other' columns, fill empty values with 0
    """
    columns_to_drop = [
        0,
        4,
        7,
        18,
        31,
        41,
        43,
        44,
        56,
        57,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
    ]
    df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
    df.columns = column_names
    # Update house materials column
    cemment_with_bricks = [
        "Bricks with ciment",
        "btickets with ciment",
        "brickets with ciment",
    ]
    df.loc[
        df.Material_Other.isin(cemment_with_bricks), "House_Material"
    ] = "Cement bonded bricks/stone"
    df.drop(["Material_Other", "Ethnicity_Others"], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df["Respondent_Age"] = df["Respondent_Age"].astype(str).map(lambda x: x.strip())


# %%
def feature_creation(df):
    df.insert(2, "household_size", df.iloc[:, 5:25].sum(axis=1))
    df.insert(3, "total_female", df.iloc[:, 14:24].sum(axis=1))
    df.insert(4, "percent_female", (df.total_female / df.household_size) * 100)
    df.drop(["total_female"], axis=1, inplace=True)
    df.insert(4, "children", df.iloc[:, [7, 8, 9, 17, 18, 19]].sum(axis=1))
    df["children"] = np.where(df.children > 0, 1, 0)
    df.insert(5, "children_under_5", df.iloc[:, [8, 18]].sum(axis=1))
    df["children_under_5"] = np.where(df.children_under_5 > 0, 1, 0)
    df.insert(
        6,
        "adults",
        df.iloc[:, [12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28]].sum(
            axis=1
        ),
    )
    df.insert(
        7,
        "income_gen_ratio",
        ((df.Income_Generating_Members / df.household_size) * 100),
    )
    df.insert(
        8, "income_gen_adults", ((df.Income_Generating_Members / df.adults) * 100)
    )
    df.insert(9, "health_difficulty", df.iloc[:, [33, 34, 35, 36, 37, 38]].sum(axis=1))
    df.health_difficulty = np.where(df.health_difficulty > 0, 1, 0)
    df["respondent_female"] = np.where(df.Respondent_Gender == "female", 1, 0)
    df["previous_nfri"] = np.where(df.Previous_NFRI == "yes", 1, 0)
    df["sindupalchowk"] = np.where(df.District == "Sindupalchowk", 1, 0)
    df.income_gen_ratio = df.income_gen_ratio.replace(np.inf, np.nan)
    df.income_gen_adults = df.income_gen_adults.replace(np.inf, np.nan)
    df.fillna(0, inplace=True)


# %%
# Set column width and figure parameters
pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = [10, 6]

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# Read data
train_hill = pd.read_csv(
    f"{project_dir}/outputs/data/data_for_modelling/train_hill.csv"
)
train_terai = pd.read_csv(
    f"{project_dir}/outputs/data/data_for_modelling/train_terai.csv"
)
# Split train - train/validation
train_hill, val_hill = train_test_split(train_hill, test_size=0.1, random_state=42)
train_terai, val_terai = train_test_split(train_terai, test_size=0.1, random_state=42)
# Group hill and plain
train = pd.concat([train_hill, train_terai], ignore_index=True)
val = pd.concat([val_hill, val_terai], ignore_index=True)

# %%
# More interpretable column names for analysis
column_names = [
    "District",
    "Latitude",
    "Longitude",
    "Respondent_Gender",
    "Respondent_Age",
    "0-5_Male",
    "6-12_Male",
    "13-17_Male",
    "18-29_Male",
    "30-39_Male",
    "40-49_Male",
    "50-59_Male",
    "60-69_Male",
    "70-79_Male",
    "80-above_Male",
    "0-5_Female",
    "6-12_Female",
    "13-17_Female",
    "18-29_Female",
    "30-39_Female",
    "40-49_Female",
    "50-59_Female",
    "60-69_Female",
    "70-79_Female",
    "80-above_Female",
    "Ethnicity",
    "Ethnicity_Others",
    "Difficulty_Seeing",
    "Difficulty_Hearing",
    "Difficulty_Walking",
    "Difficulty_Remembering",
    "Difficulty_Selfcare",
    "Difficulty_Communicating",
    "House_Material",
    "Material_Other",
    "Income_Generating_Members",
    "Previous_NFRI",
    "Plastic_Tarpaulin",
    "Blanket",
    "Sari",
    "Male_Dhoti",
    "Shouting_Cloth_Jeans",
    "Printed_Cloth",
    "Terry_Cloth",
    "Utensil_Set",
    "Water_Bucket",
    "Nylon_Rope",
    "Sack_Packing_Bag",
    "Cotton_Towel",
    "Bathing_Soap",
    "Laundry_Soap",
    "Tooth_Brush_and_Paste",
    "Sanitary_Pad",
    "Ladies_Underwear",
    "Torch_Light",
    "Whistle_Blow",
    "Nail_Cutter",
    "Hand_Sanitizer",
    "Liquid_Chlorine",
]

# %%
len(column_names)

# %%
# Items, basic and non-basic divide
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]

# %%
transform_sets(train, column_names)
transform_sets(val, column_names)

# %%
feature_creation(train)
feature_creation(val)

# %%
# Features we want to use in the model
select_features = [
    "Ethnicity",
    "House_Material",
    "household_size",
    "percent_female",
    "children",
    "children_under_5",
    "income_gen_ratio",
    "income_gen_adults",
    "health_difficulty",
    "previous_nfri",
    "sindupalchowk",
]

# %%
# X and Y split
X_train = train[select_features]
X_val = val[select_features]
y_train = train[nfri_items]
y_val = val[nfri_items]

# %%
y_train_cat = y_train.copy()
y_val_cat = y_val.copy()
y_train = nfri_preferences_to_numbers(y_train)
y_val = nfri_preferences_to_numbers(y_val)

# %%
# Looking at basic /non basic
y_train_basic = y_train[basic]
y_val_basic = y_val[basic]
y_train_non_basic = y_train[non_basic]
y_val_non_basic = y_val[non_basic]

# %% [markdown]
# ### Pre-processing / feature engineering

# %%
X_train_pre_proc = X_train.copy()

# %%
X_train_pre_proc.head(1)

# %%
X_train_pre_proc.shape

# %%
col_names = [
    "household_size",
    "percent_female",
    "income_gen_ratio",
    "income_gen_adults",
]
features_scaled = X_train_pre_proc[col_names]
scaler = RobustScaler().fit(features_scaled.values)
features_scaled = scaler.transform(features_scaled.values)

# %%
X_train_pre_proc[col_names] = features_scaled

# %%
ohe = OneHotEncoder(drop="first")
features_dummies = X_train_pre_proc[["Ethnicity", "House_Material"]]
ohe.fit(features_dummies)
codes = ohe.transform(features_dummies).toarray()
feature_names = ohe.get_feature_names_out(["Ethnicity", "House_Material"])

# %%
remaining_features = [
    "children",
    "children_under_5",
    "health_difficulty",
    "previous_nfri",
    "sindupalchowk",
]

# %%
X_train_pre_proc = pd.concat(
    [
        X_train_pre_proc[col_names],
        X_train_pre_proc[remaining_features],
        pd.DataFrame(codes, columns=feature_names).astype(int),
    ],
    axis=1,
)

# %%
X_train_pre_proc.shape

# %%
# Variance
X_train_pre_proc.var().sort_values()

# %%
# Correlated features

# Create correlation matrix
corr_matrix = X_train_pre_proc.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

# %%
# No features highly correlated
to_drop

# %%
corr_matrix

# %%
# Drop features
X_train_pre_proc.drop(to_drop, axis=1, inplace=True)

# %%
X_train_pre_proc.columns

# %% [markdown]
# ### Modelling pipeline

# %%
# Define the transformations
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
# Regression models
linear = MultiOutputRegressor(LinearRegression())
r_forest = MultiOutputRegressor(RandomForestRegressor(random_state=42))
d_tree = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
regr = MultiOutputRegressor(Ridge(random_state=42))
svr = MultiOutputRegressor(LinearSVR(random_state=42))

# %%
# Define pipelines for regression models
pipe_lr = Pipeline(steps=[("pre_proc", transformer), ("linear regression", linear)])
pipe_rf = Pipeline(steps=[("pre_proc", transformer), ("random forest", r_forest)])
pipe_dt = Pipeline(steps=[("pre_proc", transformer), ("decision tree", d_tree)])
pipe_rg = Pipeline(steps=[("pre_proc", transformer), ("decision tree", regr)])
pipe_svr = Pipeline(steps=[("pre_proc", transformer), ("svr", svr)])

# %%
# %%capture
# Pipelines fit basic
pipe_lr.fit(X_train, y_train_basic)
pipe_rf.fit(X_train, y_train_basic)
pipe_dt.fit(X_train, y_train_basic)
pipe_rg.fit(X_train, y_train_basic)
pipe_svr.fit(X_train, y_train_basic)

# %%
# R2 scores (basic)
print("r2 model scores basic")
print(pipe_lr.score(X_val, y_val_basic))
print(pipe_rf.score(X_val, y_val_basic))
print(pipe_dt.score(X_val, y_val_basic))
print(pipe_rg.score(X_val, y_val_basic))
print(pipe_svr.score(X_val, y_val_basic))
# Avg accuracy of items (basic)
print("Avg accuracy of items model scores basic")
print(
    mm.accuracy_per_item(pipe_lr, "nfri-basic", X_val, y_val_basic)["Accuracy"].mean()
)
print(
    mm.accuracy_per_item(pipe_rf, "nfri-basic", X_val, y_val_basic)["Accuracy"].mean()
)
print(
    mm.accuracy_per_item(pipe_dt, "nfri-basic", X_val, y_val_basic)["Accuracy"].mean()
)
print(
    mm.accuracy_per_item(pipe_rg, "nfri-basic", X_val, y_val_basic)["Accuracy"].mean()
)
print(
    mm.accuracy_per_item(pipe_svr, "nfri-basic", X_val, y_val_basic)["Accuracy"].mean()
)

# %%
# %%capture
# Pipelines fit non-basic
pipe_lr.fit(X_train, y_train_non_basic)
pipe_rf.fit(X_train, y_train_non_basic)
pipe_dt.fit(X_train, y_train_non_basic)
pipe_rg.fit(X_train, y_train_non_basic)
pipe_svr.fit(X_train, y_train_non_basic)

# %%
# R2 scores (basic)
print("r2 model scores basic")
print(pipe_lr.score(X_val, y_val_non_basic))
print(pipe_rf.score(X_val, y_val_non_basic))
print(pipe_dt.score(X_val, y_val_non_basic))
print(pipe_rg.score(X_val, y_val_non_basic))
print(pipe_svr.score(X_val, y_val_non_basic))
# Avg accuracy of items (basic)
print("Avg accuracy of items model scores basic")
print(
    mm.accuracy_per_item(pipe_lr, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rf, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_dt, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rg, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_svr, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)

# %% [markdown]
# ### Testing chained regression

# %%
from sklearn.multioutput import RegressorChain

# Regression models
linear = RegressorChain(LinearRegression())
r_forest = RegressorChain(RandomForestRegressor(random_state=42))
d_tree = RegressorChain(DecisionTreeRegressor(random_state=42))
regr = RegressorChain(Ridge(random_state=42))
svr = RegressorChain(LinearSVR(random_state=42))

# %%
# Define pipelines for regression models
pipe_lr_ch = Pipeline(steps=[("pre_proc", transformer), ("linear regression", linear)])
pipe_rf_ch = Pipeline(steps=[("pre_proc", transformer), ("random forest", r_forest)])
pipe_dt_ch = Pipeline(steps=[("pre_proc", transformer), ("decision tree", d_tree)])
pipe_rg_ch = Pipeline(steps=[("pre_proc", transformer), ("ridge", regr)])
pipe_svr_ch = Pipeline(steps=[("pre_proc", transformer), ("svr", svr)])

# %%
# %%capture
# Pipelines fit basic
pipe_lr_ch.fit(X_train, y_train_basic)
pipe_rf_ch.fit(X_train, y_train_basic)
pipe_dt_ch.fit(X_train, y_train_basic)
pipe_rg_ch.fit(X_train, y_train_basic)
pipe_svr_ch.fit(X_train, y_train_basic)

# %%
# R2 scores (basic)
print("r2 model scores basic")
print(pipe_lr_ch.score(X_val, y_val_basic))
print(pipe_rf_ch.score(X_val, y_val_basic))
print(pipe_dt_ch.score(X_val, y_val_basic))
print(pipe_rg_ch.score(X_val, y_val_basic))
print(pipe_svr_ch.score(X_val, y_val_basic))
# Avg accuracy of items (basic)
print("Avg accuracy of items model scores basic")
print(
    mm.accuracy_per_item(pipe_lr_ch, "nfri-basic", X_val, y_val_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rf_ch, "nfri-basic", X_val, y_val_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_dt_ch, "nfri-basic", X_val, y_val_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rg_ch, "nfri-basic", X_val, y_val_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_svr_ch, "nfri-basic", X_val, y_val_basic)[
        "Accuracy"
    ].mean()
)

# %%
# %%capture
# Pipelines fit non basic
pipe_lr_ch.fit(X_train, y_train_non_basic)
pipe_rf_ch.fit(X_train, y_train_non_basic)
pipe_dt_ch.fit(X_train, y_train_non_basic)
pipe_rg_ch.fit(X_train, y_train_non_basic)
pipe_svr_ch.fit(X_train, y_train_non_basic)

# %%
# R2 scores (basic)
print("r2 model scores basic")
print(pipe_lr_ch.score(X_val, y_val_non_basic))
print(pipe_rf_ch.score(X_val, y_val_non_basic))
print(pipe_dt_ch.score(X_val, y_val_non_basic))
print(pipe_rg_ch.score(X_val, y_val_non_basic))
print(pipe_svr_ch.score(X_val, y_val_non_basic))
# Avg accuracy of items (basic)
print("Avg accuracy of items model scores basic")
print(
    mm.accuracy_per_item(pipe_lr_ch, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rf_ch, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_dt_ch, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rg_ch, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_svr_ch, "nfri-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)

# %% [markdown]
# ### Important features for each item

# %%
rob_scal_feats = list(
    pipe_lr["pre_proc"].named_transformers_["rob_scaler"].get_feature_names_out()
)
one_hot_feats = list(
    pipe_lr["pre_proc"].named_transformers_["one_hot"].get_feature_names_out()
)
features = list(itertools.chain(rob_scal_feats, one_hot_feats, remaining_features))

# %%
importance = pipe_lr.named_steps["linear regression"].estimators_[0].coef_

for i, v in enumerate(importance):
    print(str(features[i]) + ": %0d, Score: %.5f" % (i, v))
# plot feature importance
plt.bar(features, importance)
plt.xticks(rotation=90)
plt.show()

# %%
# %%capture
# Pipelines fit non-basic
pipe_lr.fit(X_train, y_train_non_basic)
pipe_rf.fit(X_train, y_train_non_basic)
pipe_dt.fit(X_train, y_train_non_basic)

# %%
# R2 scores (non_basic)
print("r2 model scores non_basic")
print(pipe_lr.score(X_val, y_val_non_basic))
print(pipe_rf.score(X_val, y_val_non_basic))
print(pipe_dt.score(X_val, y_val_non_basic))
# Avg accuracy of items (non_basic)
print("Avg accuracy of items model scores non_basic")
print(
    mm.accuracy_per_item(pipe_lr, "nfri-non-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_rf, "nfri-non-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)
print(
    mm.accuracy_per_item(pipe_dt, "nfri-non-basic", X_val, y_val_non_basic)[
        "Accuracy"
    ].mean()
)

# %% [markdown]
# ### Cross validation

# %%
# Regression models
linear = MultiOutputRegressor(LinearRegression())
r_forest = MultiOutputRegressor(RandomForestRegressor(random_state=42))
d_tree = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
regr = MultiOutputRegressor(Ridge(random_state=42))
svr = MultiOutputRegressor(LinearSVR(random_state=42))
lso = MultiOutputRegressor(Lasso(random_state=42))

# %%
# Define pipelines for regression models
pipe_lr = Pipeline(steps=[("pre_proc", transformer), ("linear regression", linear)])
pipe_rf = Pipeline(steps=[("pre_proc", transformer), ("random forest", r_forest)])
pipe_dt = Pipeline(steps=[("pre_proc", transformer), ("decision tree", d_tree)])
pipe_rg = Pipeline(steps=[("pre_proc", transformer), ("ridge", regr)])
pipe_svr = Pipeline(steps=[("pre_proc", transformer), ("svr", svr)])
pipe_lso = Pipeline(steps=[("pre_proc", transformer), ("lso", lso)])

# %%
# define the model cross-validation configuration
cv = KFold(n_splits=10, shuffle=True, random_state=1)

# %%
# evaluate the pipeline using cross validation and calculate r2 + MAE (basic)
scores_lr_basic = cross_validate(
    pipe_lr,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_rf_basic = cross_validate(
    pipe_rf,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_dt_basic = cross_validate(
    pipe_dt,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_rg_basic = cross_validate(
    pipe_rg,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_svr_basic = cross_validate(
    pipe_svr,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_lso_basic = cross_validate(
    pipe_lso,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)

# %%
# evaluate the pipeline using cross validation and calculate r2 + MAE (non basic)
scores_lr_nb = cross_validate(
    pipe_lr,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_rf_nb = cross_validate(
    pipe_rf,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_dt_nb = cross_validate(
    pipe_dt,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_rg_nb = cross_validate(
    pipe_rg,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_svr_nb = cross_validate(
    pipe_svr,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_lso_nb = cross_validate(
    pipe_lso,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)

# %%
print(
    "Basic: Linear regression negative mse avg: ",
    scores_lr_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: Linear regression negative mse avg: ",
    scores_lr_nb["test_neg_mean_squared_error"].mean(),
)
print("")
print(
    "Basic: Random Forest negative mse avg: ",
    scores_rf_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: Random Forest negative mse avg: ",
    scores_rf_nb["test_neg_mean_squared_error"].mean(),
)
print("")
print(
    "Basic: Decision Tree negative mse avg: ",
    scores_dt_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: Decision Tree negative mse avg: ",
    scores_dt_nb["test_neg_mean_squared_error"].mean(),
)
print("")
print(
    "Basic: Ridge Regression negative mse avg: ",
    scores_rg_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: Ridge Regression negative mse avg: ",
    scores_rg_nb["test_neg_mean_squared_error"].mean(),
)
print("")
print(
    "Basic: SVR negative mse avg: ",
    scores_svr_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: SVR negative mse avg: ",
    scores_svr_nb["test_neg_mean_squared_error"].mean(),
)
print("")
print(
    "Basic: Lasso negative mse avg: ",
    scores_lso_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: Lasso negative mse avg: ",
    scores_lso_nb["test_neg_mean_squared_error"].mean(),
)

# %%
print("Basic: Linear regression r2 avg: ", scores_lr_basic["test_r2"].mean())
print("Non-basic: Linear regression r2 avg: ", scores_lr_nb["test_r2"].mean())
print("")
print("Basic: Random Forest r2 avg: ", scores_rf_basic["test_r2"].mean())
print("Non-basic: Random Forest r2 avg: ", scores_rf_nb["test_r2"].mean())
print("")
print("Basic: Decision Tree r2 avg: ", scores_dt_basic["test_r2"].mean())
print("Non-basic: Decision Tree r2 avg: ", scores_dt_nb["test_r2"].mean())
print("")
print("Basic: Ridge Regression r2 avg: ", scores_rg_basic["test_r2"].mean())
print("Non-basic: Ridge Regression r2 avg: ", scores_rg_nb["test_r2"].mean())
print("")
print("Basic: SVR r2 avg: ", scores_svr_basic["test_r2"].mean())
print("Non-basic: SVR r2 avg: ", scores_svr_nb["test_r2"].mean())
print("")
print("Basic: Lasso r2 avg: ", scores_lso_basic["test_r2"].mean())
print("Non-basic: Lasso r2 avg: ", scores_lso_nb["test_r2"].mean())

# %% [markdown]
# ### Tuning alpha for ridge regression

# %%
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

# %%
alphas = [0.01, 0.1, 1.0, 10.0, 100]

# %%
alphas

# %%
parameters = [{"ridge__estimator__alpha": alphas}]

# %%
scoring_func = make_scorer(mean_squared_error)

grid_search = GridSearchCV(
    estimator=pipe_rg,
    param_grid=parameters,
    scoring=scoring_func,  # <--- Use the scoring func defined above
    cv=10,
    n_jobs=-1,
)

# %%
# %%capture
grid_search.fit(X_train, y_train_basic)

# %%
best_parameters = grid_search.best_params_
print(best_parameters)

# %%
# %%capture
grid_search.fit(X_train, y_train_non_basic)

# %%
best_parameters = grid_search.best_params_
print(best_parameters)

# %% [markdown]
# #### Re-running with best alpha parameter

# %%
regr = MultiOutputRegressor(Ridge(random_state=42, alpha=100))
pipe_rg = Pipeline(steps=[("pre_proc", transformer), ("ridge", regr)])

# %%
scores_rg_basic = cross_validate(
    pipe_rg,
    X_train,
    y_train_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)
scores_rg_nb = cross_validate(
    pipe_rg,
    X_train,
    y_train_non_basic,
    cv=cv,
    scoring=("r2", "neg_mean_squared_error"),
    return_train_score=True,
)

# %%
print("")
print(
    "Basic: Ridge Regression negative mse avg: ",
    scores_rg_basic["test_neg_mean_squared_error"].mean(),
)
print(
    "Non-basic: Ridge Regression negative mse avg: ",
    scores_rg_nb["test_neg_mean_squared_error"].mean(),
)
print("")
print("Basic: Ridge Regression r2 avg: ", scores_rg_basic["test_r2"].mean())
print("Non-basic: Ridge Regression r2 avg: ", scores_rg_nb["test_r2"].mean())
print("")

# %% [markdown]
# ### Feature selection

# %%
X_train_pre_proc.shape

# %% [markdown]
# - Sequential Feature Selector: Selects features that maximizes the Cross Validation Score of model. (That is minimizes the mean error in our case.)
# - Recursive Feature Elimination: Eliminates features with least important coefficient scores until the desired number is reached.
# - Select From Model: Almost same as Recursive Feature Elimination, except all the coefficients lying below some threshold (default is mean) eliminated.

# %% [markdown]
# #### NFRI Basic

# %%
import time
from sklearn.feature_selection import SequentialFeatureSelector

start = time.time()

sfs_selector = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select=5,
    cv=2,
    direction="backward",
)
sfs_selector.fit(X_train_pre_proc, y_train_basic)

print("--- %s seconds ---" % (time.time() - start))
print(X_train_pre_proc.columns[sfs_selector.get_support()])

# %%
from sklearn.feature_selection import SelectFromModel

sfm_selector = SelectFromModel(estimator=LinearRegression())
sfm_selector.fit(X_train_pre_proc, y_train_basic)
X_train_pre_proc.columns[sfm_selector.get_support()]

# %%
from sklearn.feature_selection import RFE

rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)
rfe_selector.fit(X_train_pre_proc, y_train_basic)
X_train_pre_proc.columns[rfe_selector.get_support()]

# %% [markdown]
# #### NFRI Non Basic

# %%
start = time.time()

sfs_selector = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select=5,
    cv=2,
    direction="backward",
)
sfs_selector.fit(X_train_pre_proc, y_train_non_basic)

print("--- %s seconds ---" % (time.time() - start))
print(X_train_pre_proc.columns[sfs_selector.get_support()])

# %%
sfm_selector = SelectFromModel(estimator=LinearRegression())
sfm_selector.fit(X_train_pre_proc, y_train_non_basic)
X_train_pre_proc.columns[sfm_selector.get_support()]

# %%
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)
rfe_selector.fit(X_train_pre_proc, y_train_non_basic)
X_train_pre_proc.columns[rfe_selector.get_support()]
