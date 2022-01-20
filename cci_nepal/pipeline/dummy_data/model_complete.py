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

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SequentialFeatureSelector


from matplotlib import pyplot as plt
import time
from operator import itemgetter

import logging
import cci_nepal
from cci_nepal.getters.dummy_data import get_dummy_data as gdd
from cci_nepal.utils.dummy_data import dummy_data_utils as ddu

# %load_ext nb_black

project_dir = cci_nepal.PROJECT_DIR
logging.info(project_dir)


df = gdd.read_csv_file(f"{project_dir}/inputs/data/Dummy_Data_Full_Survey_Features.csv")
df.head()

# #### Creating Additional Features ####

# In this section, we will create 3 additional features combined out of existing features to see how effectively they will be in Model. The 2 newly created features will be 'Total Members', 'Total Female' and 'Total Children'

df.insert(2, "Total_Members", df.iloc[:, 7:28].sum(axis=1))
df.insert(3, "Total_Female", df.iloc[:, 17:27].sum(axis=1))
# df.insert(4, "Total_Children", df.iloc[:, [7, 8, 9, 17, 18, 19]].sum(axis=1))

# #### Mapping Categories to Numbers ####

# The idea is to map categorical labels of NFRI Preferences (Essential, Desirable, Unnecessary) into numeric scores to be used in Model. That way, we will have a continuous scale for NFRI Importance Score.

mapping = {"Essential": 3, "Desirable": 2, "Unnecessary": 1}
df_clean = df.applymap(lambda s: mapping.get(s) if s in mapping else s)

df_clean.head(5)

# #### Test Section Only. To be excluded for real data ####

# Since the randomly created labels are somewhat evenly generated, the predicted label mostly averages out as 2 (Desirable). Thus, to ensure our Model can predict all three labels, we are feeding the score of 1 and 3 to all observations of some of the items here.

ddu.replace_column_with_number(
    df_clean, ["Blanket", "Printed Cloth", "Hand Sanitizer"], [3, 1, 3]
)

df_clean.head(3)

# #### Categorizing appropriate variables ####

# The following code is to first automatically collect all the categorical columns from the dataset, and then categorize the columns as a preparatory step for Linear Regression model.

# +
num_cols = df_clean._get_numeric_data().columns

columns_to_categorize = list(set(df_clean.columns) - set(num_cols))

df_clean[columns_to_categorize] = df_clean[columns_to_categorize].astype("category")
# -

model_basic = LinearRegression()
model_non_basic = LinearRegression()

# #### Collecting Input and Output features ####

input_features = df_clean.columns[:-21]
output_features = df_clean.columns[-21:]
input_numerical_features = [
    features for features in input_features if features not in columns_to_categorize
]

# #### Feature Scaling ####

# In this step, we will form Feature Scaling of numerical input features using Min-Max scaler.

scaler = MinMaxScaler()
df_clean[input_numerical_features] = scaler.fit_transform(
    df_clean[input_numerical_features]
)

# #### Feature Selection Step ####

# This is the section where we will manually select the Features to be kept into the Model. For real data, we will perform Feature Analysis, Selection and Engineering process before this step. For now, we will select all the Features.

X = df_clean[input_features]  # Collecting all features except the NFRI Item Preferences
X = pd.get_dummies(data=X, drop_first=True)

# Dividing NFRI into basic and non-basic in the step below. So as to prepare two separate Model for basic and non-basic items.

# +
nfri_basic = [
    "Plastic Tarpaulin",
    "Blanket",
    "Sari",
    "Male Dhoti",
    "Shouting Cloth / Jeans",
    "Printed Cloth",
    "Terry Cloth",
    "Utensil Set",
    "Water Bucket",
    "Nylon Rope",
    "Sack (Packing Bag)",
]

nfri_non_basic = [
    "Cotton Towel",
    "Laundry Soap",
    "Tooth Brush and Paste",
    "Sanitary Pad",
    "Ladies Underwear",
    "Torch Light",
    "Whistle Blow",
    "Nail Cutter",
    "Hand Sanitizer",
    "Liquid Chlorine",
]

y_basic = df_clean.loc[:, nfri_basic]


y_non_basic = df_clean.loc[:, nfri_non_basic]
# -

# #### Fitting the Linear Regression Model for both NFRI below ####

# First we will create the train and test dataset. And after that, we will fit on the train set and predict on test.

X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X, y_basic)
(
    X_train_non_basic,
    X_test_non_basic,
    y_train_non_basic,
    y_test_non_basic,
) = train_test_split(X, y_non_basic)


model_basic.fit(X_train_basic, y_train_basic)
model_non_basic.fit(X_train_non_basic, y_train_non_basic)

# #### Predicting the Model for Test Set so as to calculate accuracy below ####

test_predictions_basic = model_basic.predict(X_test_basic)
test_predictions_non_basic = model_non_basic.predict(X_test_basic)


# +
test_predictions_labels_basic = [
    [ddu.numeric_score_transformer(i) for i in nested]
    for nested in test_predictions_basic
]
test_predictions_items_wise_basic = list(
    map(list, zip(*test_predictions_labels_basic))
)  # Transforming the above list

test_predictions_labels_non_basic = [
    [ddu.numeric_score_transformer(i) for i in nested]
    for nested in test_predictions_non_basic
]
test_predictions_items_wise_non_basic = list(
    map(list, zip(*test_predictions_labels_non_basic))
)  # Transforming the above list
# -


# #### Accuracy of the Model for each NFRI item ####

# The below section will display the accuracy of the model (using the Categorical labels mapped from numerical score) for each NFRI item.

# +
accuracy_list_basic = []
for i in range(0, len(nfri_basic)):
    accuracy_list_basic.append(
        accuracy_score(test_predictions_items_wise_basic[i], y_test_basic.iloc[:, i])
    )

logging.info("Accuracy Per NFRI Basic Item")
logging.info(
    pd.DataFrame(accuracy_list_basic, y_test_basic.columns, columns=["Accuracy"])
)


# +
accuracy_list_non_basic = []
for i in range(0, len(nfri_non_basic)):
    accuracy_list_non_basic.append(
        accuracy_score(
            test_predictions_items_wise_non_basic[i], y_test_non_basic.iloc[:, i]
        )
    )

logging.info("Accuracy Per NFRI Non Basic Item")
logging.info(
    pd.DataFrame(
        accuracy_list_non_basic, y_test_non_basic.columns, columns=["Accuracy"]
    )
)

# -

# #### Analysing the Model Coefficients ####

# In this section, we will analyse the Model Coefficients of the fitted Linear Regression Model. For now, the Model here is fitted using the whole dataset.

model_basic.fit(X, y_basic)
model_non_basic.fit(X, y_non_basic)


coefficient_scores = model_basic.coef_[0]
plt.bar([x for x in range(len(coefficient_scores))], coefficient_scores)
plt.xlabel("Features")
plt.ylabel("Coefficient Scores")
# plt.show()

# The same step can be repeated for NFRI non-basic.

# #### Cross Validation ####

# In this section, we will perform K-fold cross validation and evaluate the Mean Absolute Error (MAE) and R Square for both of our models.

# +
model_test = LinearRegression()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# First performing the steps for NFRI basic

n_scores_basic = cross_val_score(
    model_test, X, y_basic, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)
n_scores_basic_absolute = np.absolute(n_scores_basic)


logging.info(
    "MAE of Basic: %.3f (%.3f)"
    % (np.mean(n_scores_basic_absolute), np.std(n_scores_basic))
)


r2_score_basic = cross_val_score(model_test, X, y_basic, cv=cv, scoring="r2")
logging.info("Average R2 Score for NFRI Basic: %.3f" % np.nanmean(r2_score_basic))


# Repeating the same steps for NFRI_non_basic

n_scores_non_basic = cross_val_score(
    model_test, X, y_non_basic, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)
n_scores_non_basic_absolute = np.absolute(n_scores_non_basic)
logging.info(
    "MAE of Non Basic: %.3f (%.3f)"
    % (np.mean(n_scores_non_basic_absolute), np.std(n_scores_non_basic))
)


r2_score_non_basic = cross_val_score(model_test, X, y_non_basic, cv=cv, scoring="r2")
logging.info(
    "Average R2 Score for NFRI Non Basic: %.3f" % np.nanmean(r2_score_non_basic)
)

# -

# ### Feature Selection

# In this step, we will perform Feature Selection of our Model using the entire dataset for both Basic and Non-basic NFRI. For now, we have selected 10 as the number of features to select.
#
# In real datastep, this step will be peformed before fitting the final model.
#
# Since this step could take more computational time depending upon users machine, we have also added time for this section.

# +

logging.info(
    "The following step is the Feature Selection step using Sequential Feature Selector."
)
logging.info(
    "Depending upon users machine, the computation time could vary, which will be calculated after the step ends."
)
start = time.time()

sfs_selector = SequentialFeatureSelector(
    estimator=LinearRegression(), n_features_to_select=5, cv=5, direction="backward"
)
sfs_selector.fit(X, y_basic)

logging.info("--- %s seconds ---" % (time.time() - start))
logging.info("The 5 selected features are:")
logging.info(X.columns[sfs_selector.get_support()])


# -

# ### NFRI Calculator Function

# In the final section, we have created a function that predicts the most important NFRI for a household with certain input features. The calculator will display the most important NFRI along with importance score and labels of each NFRI.


def nfri_calculator(model, nfri_type, input_feature):

    """

    Takes the input features and outputs the sorted NFRI items for the selected features.

    Parameters:

    model: Trained Machine Learning model
    nfri_type: Type of NFRI to predict (basic or non-basic)
    input_feature: A list of input features for which the output is to be predicted

    Returns:

    A pandas dataframe with NFRI Items, Importance Score, and Labels, sorted wrt Importance Score (desc order)

    """

    yhat = model.predict([input_feature])
    sorted_indices = np.flip(np.argsort(yhat[0]))
    sorted_items = itemgetter(*sorted_indices)(nfri_type)
    sorted_scores = np.sort(yhat[0])[::-1].tolist()
    sorted_labels = [ddu.label_transformer(d) for d in -np.sort(-yhat[0])]
    return pd.DataFrame(
        {
            "Sorted Items": sorted_items,
            "Sorted Scores": sorted_scores,
            "Sorted labels": sorted_labels,
        }
    )


# Below we will use our NFRI Calculator function by feeding in one dummy input feature. For now, we will use one of the observations from the dataset as input.

feature_dummy = X.iloc[
    10,
].tolist()

nfri_basic_prediction = nfri_calculator(model_basic, nfri_basic, feature_dummy)

nfri_non_basic_prediction = nfri_calculator(
    model_non_basic, nfri_non_basic, feature_dummy
)

logging.info(nfri_basic_prediction)

logging.info(nfri_non_basic_prediction)
