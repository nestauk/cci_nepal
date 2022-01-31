# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# +
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SequentialFeatureSelector

import time

import logging
import cci_nepal
from cci_nepal.getters.dummy_data import get_dummy_data as gdd
from cci_nepal.pipeline.dummy_data import data_manipulation as dm
from cci_nepal.pipeline.dummy_data import feature_manipulation as fm
from cci_nepal.pipeline.dummy_data import model_manipulation as mm
from cci_nepal.pipeline.dummy_data import nfri_list_file as nlf


# %load_ext nb_black

project_dir = cci_nepal.PROJECT_DIR
logging.info("Logging the project directory below:")
logging.info(project_dir)


# #### Reading the data ####

df = gdd.read_csv_file(
    f"{project_dir}/inputs/data/dummy_data/Dummy_Data_Full_Survey_Features.csv"
)


# Data Manipulation Section #

# #### Mapping Categories to Numbers ####

# The idea is to map categorical labels of NFRI Preferences (Essential, Desirable, Unnecessary) into numeric scores to be used in Model. That way, we will have a continuous scale for NFRI Importance Score.


df_clean = dm.nfri_preferences_to_numbers(df)

# #### Categorizing appropriate variables ####

# The following code is to first automatically collect all the categorical columns from the dataset, and then categorize the columns as a preparatory step for Linear Regression model.


dm.categorize_dataframe_variables(df_clean)


# ## NFRI Type Selection ##

# #### Here we selceted the type of NFRI that we want to predict below ####

# #### nfri-basic or nfri-non-basic, the two options that we choose from ####


nfri_type = "nfri-non-basic"  # Replace it with "nfri-basic" for basic items


# ### Feature Manipulation Section ###

# #### Input Feature Selection ####

# First, we select the input features (X) from our dataframe.

X = fm.input_feature_selection(
    df_clean, 21
)  # 21 is the input/output feature divider index in our case


# #### Feature Scaling

# In this step, we will form Feature Scaling of numerical input features using Min-Max scaler.

fm.feature_scaling(X)  # Inplace feature scaling happens at this step


# #### Feature Dummying ####

# In this step, we create dummy variables for all the categorical input features.

X = fm.feature_dummying(X)


# #### Output Features Section ####

# Depending on the NFRI type chosen above, we select output features (y) from our dataframe.

y = fm.output_feature_selection(df_clean, nfri_type)


# ### Modeling Section ###

# #### Fitting the chosen model for the chosen NFRI type below ####

# First we will create the train, validation and test dataset. And after that, we will fit on the train set and predict on test.

# #### Train-Validation-Set Separation ####

X_train, y_train, X_validation, y_validation, X_test, y_test = mm.train_test_validation(
    X, y, 0.8
)


# #### Model Fitting ####

model = mm.select_model(
    "linear-regression"
)  # Other model options are "k-nearest", "decision-tree", and "random-forest"
model.fit(X_train, y_train)


# #### Model Evaluation ####

logging.info(mm.accuracy_per_item(model, nfri_type, X_test, y_test))


# The same step can be repeated for the other NFRI type.

# ### Cross Validation ###

# In this section, we will perform K-fold cross validation and evaluate the Mean Absolute Error (MAE) and R Square for the chosen NFRI type model.


model_crossvalidation = mm.select_model("linear-regression")
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

n_scores = cross_val_score(
    model_crossvalidation,
    X_train,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
)
n_scores_absolute = np.absolute(n_scores)

logging.info(
    "MAE of Basic: %.3f (%.3f)" % (np.mean(n_scores_absolute), np.std(n_scores))
)

r2_score = cross_val_score(model_crossvalidation, X_train, y_train, cv=cv, scoring="r2")
logging.info("Average R2 Score: %.3f" % np.nanmean(r2_score))

# ### Feature Selection

# In this step, we will perform Feature Selection of our Model using the train dataset for the chosen NFRI type. For now, we have selected 5 as the number of features to select.
#
# In real datastep, this step will be peformed before fitting the final model.
#
# Since this step could take more computational time depending upon users machine, we have also added time for this section.

logging.info(
    "Feature selection step: could take a few minutes depending on the machine."
)

start = time.time()

sfs_selector = SequentialFeatureSelector(
    estimator=mm.select_model("linear-regression"),
    n_features_to_select=5,
    cv=2,
    direction="backward",
)
sfs_selector.fit(X_train, y_train)

logging.info("--- %s seconds ---" % (time.time() - start))
logging.info(X.columns[sfs_selector.get_support()])

# ### NFRI Calculator Function

# In the final section, we have created a function that predicts the most important NFRI for a household with certain input features. The calculator will display the most important NFRI along with importance score and labels of each NFRI.

# Below we will use our NFRI Calculator function by feeding in one dummy input feature. For now, we will use one of the observations from the test set as input.


feature_dummy = X_test.iloc[
    10,
].tolist()


logging.info(mm.nfri_calculator(model, nfri_type, feature_dummy))
