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
# # Model run
# <br>
# <br>
# The purpose of this script is to run the x2 classification models (Shelter and Wash-Dignity NFRI items) on new data (defaulted to the test set) and save the results to excel files. The resulting files are:
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
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
import joblib

# Project libraries
import cci_nepal
from cci_nepal.getters import get_data as grd
from cci_nepal.pipeline import data_manipulation as dm
from cci_nepal.pipeline import model_tuning_report as mtr
from cci_nepal import config

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# Get parameters from config file
b_features = config["final_model"]["model_features"]

# Read data and feature names
test = grd.read_test_data()

# For dummy data prediction
# test = pd.read_csv(f"{project_dir}/outputs/data/data_for_modelling/test_dummy_data.csv")

column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")
select_features = grd.get_lists(f"{project_dir}/cci_nepal/config/select_features.csv")


# %%
# For bias audit
# subset = "female"
# test = test[test.iloc[:, 3] == subset]
# print(test.iloc[:, 3].value_counts())

# Lowercase values
test = test.applymap(lambda s: s.lower() if type(s) == str else s)


# %%
# Items, basic and non-basic divide
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]

# %%
# Data transformations and feature creation
dm.transform_sets(test, column_names)
dm.feature_creation(test)

# %%
# X and Y split
X_test = test[select_features]
y_test = test[nfri_items]

# %%
# Preferences to binary
y_test = dm.nfri_preferences_to_binary(y_test)

# %%
# Split basic /non basic
y_test_basic = y_test[basic]
y_test_non_basic = y_test[non_basic]

# %%
# Load transformer applied fitted to the training set
transformer = joblib.load(f"{project_dir}/outputs/models/transformer.pkl")

# %%
# Apply column transformer
X_test_transform = transformer.transform(X_test)

# %%
# Assign back to dataframes - to have feature names back
X_test_transform = pd.DataFrame(
    X_test_transform, columns=list(transformer.get_feature_names_out())
)


for feat in b_features:
    if feat not in X_test_transform.columns:
        X_test_transform[feat] = 0

# %%
# Reduce to just chosen features for basic and non-basic
X_test_basic = X_test_transform[b_features].copy()
X_test_non_basic = X_test_transform[b_features].copy()

# %%
# Add folder if not already created
Path(f"{project_dir}/outputs/models/").mkdir(parents=True, exist_ok=True)

# %%
# Loading models (best performing)
basic_model = pickle.load(
    open(f"{project_dir}/outputs/models/final_classification_model_shelter.sav", "rb")
)

non_basic_model = pickle.load(
    open(f"{project_dir}/outputs/models/final_classification_model_wash.sav", "rb")
)

# Predict on test set
y_pred_basic = basic_model.predict(X_test_basic)
y_pred_non_basic = non_basic_model.predict(X_test_non_basic)


# %%
# Predict probablity on test set
y_pred_basic_probability = basic_model.predict_proba(X_test_basic)
y_pred_non_basic_probability = non_basic_model.predict_proba(X_test_non_basic)

# %%
# Create dataframes with model inputs and predictions for basic and non basic
basic_preds = mtr.create_predictions_files(
    y_pred_basic_probability,
    basic,
    X_test,
    [
        "house_material",
        "household_size",
        "percent_non_male",
        "children_under_5",
        "income_gen_ratio",
        "health_difficulty",
        "sindupalchowk",
    ],
)

non_basic_preds = mtr.create_predictions_files(
    y_pred_non_basic_probability,
    non_basic,
    X_test,
    [
        "house_material",
        "household_size",
        "percent_female",
        "children_under_5",
        "income_gen_ratio",
        "health_difficulty",
        "sindupalchowk",
    ],
)


# f1_micro_overall_basic = f1_score(y_test_basic, y_pred_basic, average="micro")
# f1_micro_overall_non_basic = f1_score(y_test_non_basic, y_pred_non_basic, average="micro")

# overall_f1_score_basic = {'Subset': [subset], 'Overall_f1_score_micro': [f1_micro_overall_basic]}
# overall_f1_score_non_basic = {'Subset': [subset], 'Overall_f1_score_micro': [f1_micro_overall_non_basic]}

# overall_f1_micro_score_basic_df = pd.DataFrame(data=overall_f1_score_basic)
# overall_f1_micro_score_non_basic_df = pd.DataFrame(data=overall_f1_score_non_basic)

evaluation_metric_basic = mtr.accuracy_per_item(basic_model, X_test_basic, y_test_basic)
evaluation_metric_non_basic = mtr.accuracy_per_item(
    non_basic_model, X_test_non_basic, y_test_non_basic
)

# Add folder if not already created
Path(f"{project_dir}/outputs/data/test_evaluation_results/").mkdir(
    parents=True, exist_ok=True
)

# overall_f1_micro_score_basic_df.to_csv(f"{project_dir}/outputs/data/test_evaluation_results/bias_audit_f1_micro_shelter.csv", mode='a', index=False, header=False)
# overall_f1_micro_score_non_basic_df.to_csv(f"{project_dir}/outputs/data/test_evaluation_results/bias_audit_f1_micro_wash.csv", mode='a', index=False, header=False)

# overall_f1_micro_score_basic_df.to_csv('bias_audit_f1_micro.csv', mode='a', index=False, header=False)

# %%
# Save files

# overall_f1_micro_score_df.to_excel(
#    f"{project_dir}/outputs/data/test_evaluation_results/overall_f1_score_micro.xlsx", index=False
# )


basic_preds.to_excel(
    f"{project_dir}/outputs/data/test_evaluation_results/shelter_test_predictions.xlsx",
    index=False,
)
non_basic_preds.to_excel(
    f"{project_dir}/outputs/data/test_evaluation_results/wash_test_predictions.xlsx",
    index=False,
)


evaluation_metric_basic.to_excel(
    f"{project_dir}/outputs/data/test_evaluation_results/shelter_test_evaluation.xlsx",
    index=True,
)
evaluation_metric_non_basic.to_excel(
    f"{project_dir}/outputs/data/test_evaluation_results/wash_test_evaluation.xlsx",
    index=True,
)

# %%
# Create confusion matrix from the best performing model
cm_basic = multilabel_confusion_matrix(y_test_basic, y_pred_basic)
cm_non_basic = multilabel_confusion_matrix(y_test_non_basic, y_pred_non_basic)

# %%
# Get accuracy, sens and spec from confusion matrix per basic item
for cm, item in zip(cm_basic, basic):

    print(item.replace("_", " "))
    total1 = sum(sum(cm))
    # From confusion matrix calculate accuracy
    accuracy1 = (cm[0, 0] + cm[1, 1]) / total1
    print("Accuracy : ", accuracy1)

    sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print("Sensitivity : ", sensitivity1)

    specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("Specificity : ", specificity1)

    print("--")
    print(" ")

# %%
# Get accuracy, sens and spec from confusion matrix per non-basic item
for cm, item in zip(cm_non_basic, non_basic):

    print(item.replace("_", " "))
    total1 = sum(sum(cm))
    # From confusion matrix calculate accuracy
    accuracy1 = (cm[0, 0] + cm[1, 1]) / total1
    print("Accuracy : ", accuracy1)

    sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print("Sensitivity : ", sensitivity1)

    specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("Specificity : ", specificity1)

    print("--")
    print(" ")

# %%
print("Shelter items")
mtr.save_cm_plots(cm_basic, "basic", basic)

# %%
print("Wash and Dignity items")
mtr.save_cm_plots(cm_non_basic, "non_basic", non_basic)
