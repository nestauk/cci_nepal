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
# Import libraries
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

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
plt.rc("figure", max_open_warning=0)

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

# %% [markdown]
# ### Test best performing model / features from training

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
y_pred_basic = basic_model.predict(X_test_basic)
y_pred_non_basic = non_basic_model.predict(X_test_non_basic)

# %%
f1_scores = []
for i in range(0, 11):
    f1 = f1_score(
        y_test_basic[y_test_basic.columns[i]],
        pd.DataFrame(y_pred_basic)[pd.DataFrame(y_pred_basic).columns[i]],
        average="micro",
    )
    f1_scores.append(f1)

# %%
# %matplotlib inline
plt.style.use("ggplot")

x_pos = [i for i, _ in enumerate(basic)]
plt.ylim(ymin=0.5, ymax=1)

plt.bar(x_pos, f1_scores, color="green")
plt.xlabel("NFRI Item")
plt.ylabel("F1 scores")
plt.title("Basic items: F1 scores on test set", pad=20)
plt.xticks(rotation=90)

plt.xticks(x_pos, basic)

plt.show()

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
print("Basic items")
mtr.save_cm_plots(cm_basic, "basic", basic)

# %%
print("Non basic items")
mtr.save_cm_plots(cm_non_basic, "non_basic", non_basic)

# %%
