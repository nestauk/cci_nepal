# Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.utils import shuffle
import pickle
import logging

# Project libraries
import cci_nepal
from cci_nepal.getters.classification_model import get_real_data as grd
from cci_nepal.pipeline.classification_model import data_manipulation as dm

from cci_nepal.pipeline.classification_model import model_tuning_report as mtr


# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR
logging.info(project_dir)

column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")
select_features = grd.get_lists(f"{project_dir}/cci_nepal/config/select_features.csv")

# %%
# Items, basic and non-basic divide
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]
# %%

# Read test data of hill and terai and combine them
test_hill = grd.read_test_hill_data()
test_terai = grd.read_test_terai_data()
# Group hill and plain
test = pd.concat([test_hill, test_terai], ignore_index=True)
test = shuffle(test, random_state=1)
test = test.applymap(lambda s: s.lower() if type(s) == str else s)
dm.transform_sets(test, column_names)
dm.feature_creation(test)
X_test = test[select_features]
y_test = test[nfri_items]
y_test = dm.nfri_preferences_to_binary(y_test)
y_test_basic = y_test[basic]
y_test_non_basic = y_test[non_basic]

# Loading the already fitted and saved logistic regression models
model_basic = pickle.load(
    open(f"{project_dir}/outputs/data/model_results/best_estimator_basic.pkl", "rb")
)

model_non_basic = pickle.load(
    open(f"{project_dir}/outputs/data/model_results/best_estimator_non_basic.pkl", "rb")
)
logging.info(mtr.accuracy_per_item(model_basic, X_test, y_test_basic))
logging.info(mtr.accuracy_per_item(model_basic, X_test, y_test_non_basic))
