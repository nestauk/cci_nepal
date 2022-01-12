import pandas as pd
import numpy as np
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from matplotlib import pyplot
from operator import itemgetter

import logging
import cci_nepal
from cci_nepal.getters.dummy_data import get_dummy_data as gdd


def label_transformer(d):
    if d < 1.5:
        return "Unnecessary"
    elif d < 2.5:
        return "Desirable"
    else:
        return "Essential"


def numeric_score_transformer(d):
    if d < 1.5:
        return 1
    elif d < 2.5:
        return 2
    else:
        return 3


project_dir = cci_nepal.PROJECT_DIR
logging.info(project_dir)


df = gdd.read_csv_file(f"{project_dir}/inputs/data/Dummy_Data_Full_Survey_Features.csv")
df.head()


df.columns

# Mapping categories to numbers #

mapping = {"Essential": 3, "Desirable": 2, "Unnecessary": 1}
df_dummy = df.applymap(lambda s: mapping.get(s) if s in mapping else s)


df_dummy.head(5)

# This part is just to test if all 3 labels are predicted well in the model ###

# Since most of the predictions were otherwise averaged around 2 ###


def replace_with_number(df, columns, values):
    for one_column, one_value in zip(columns, values):
        df[one_column] = one_value

    return df


replace_with_number(df_dummy, ["Blanket", "Printed Cloth", "Hand Sanitizer"], [3, 1, 3])


# Columns to categorize for Linear Regression modeling #

num_cols = df_dummy._get_numeric_data().columns

columns_to_categorize = list(set(df_dummy.columns) - set(num_cols))

df_dummy[columns_to_categorize] = df_dummy[columns_to_categorize].astype("category")


model_basic = LinearRegression()
model_non_basic = LinearRegression()


df_dummy.shape
df_dummy.iloc[:, 0:33]

# Features list to be selected here #

X = df_dummy.iloc[:, 0:33]  # All features fed in for now
X = pd.get_dummies(data=X, drop_first=True)
logging.info(X.columns)
X


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

y_basic = df_dummy.loc[:, nfri_basic]
y_basic

y_non_basic = df_dummy.loc[:, nfri_non_basic]
y_non_basic

X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X, y_basic)
(
    X_train_non_basic,
    X_test_non_basic,
    y_train_non_basic,
    y_test_non_basic,
) = train_test_split(X, y_non_basic)


model_basic.fit(X_train_basic, y_train_basic)
model_non_basic.fit(X_train_non_basic, y_train_non_basic)


test_predictions_basic = model_basic.predict(X_test_basic)
test_predictions_non_basic = model_non_basic.predict(X_test_basic)

test_predictions_labels_basic = [
    [numeric_score_transformer(i) for i in nested] for nested in test_predictions_basic
]
test_predictions_items_wise_basic = list(
    map(list, zip(*test_predictions_labels_basic))
)  # Transforming the above list

test_predictions_labels_non_basic = [
    [numeric_score_transformer(i) for i in nested]
    for nested in test_predictions_non_basic
]
test_predictions_items_wise_non_basic = list(
    map(list, zip(*test_predictions_labels_non_basic))
)  # Transforming the above list


accuracy_list_basic = []
for i in range(0, len(nfri_basic)):
    accuracy_list_basic.append(
        accuracy_score(test_predictions_items_wise_basic[i], y_test_basic.iloc[:, i])
    )

logging.info(
    pd.DataFrame(accuracy_list_basic, y_test_basic.columns, columns=["Accuracy"])
)


accuracy_list_non_basic = []
for i in range(0, len(nfri_non_basic)):
    accuracy_list_non_basic.append(
        accuracy_score(
            test_predictions_items_wise_non_basic[i], y_test_non_basic.iloc[:, i]
        )
    )

logging.info(
    pd.DataFrame(
        accuracy_list_non_basic, y_test_non_basic.columns, columns=["Accuracy"]
    )
)


model_basic.fit(X, y_basic)
model_non_basic.fit(X, y_non_basic)

# Making prediction

row_basic = X.iloc[
    10,
].tolist()
yhat_basic = model_basic.predict([row_basic])

# Summarizing prediction

logging.info("Printing the scores for NFRI Basic")
logging.info(yhat_basic[0])


logging.info("Now printing the scores for NFRI Non Basic")


# Making prediction

row_non_basic = X.iloc[
    10,
].tolist()
yhat_non_basic = model_non_basic.predict([row_non_basic])

# Summarizing prediction

logging.info(yhat_non_basic[0])


# Plotting the Coefficient scores of the Model

coefficient_scores = model_basic.coef_[0]

for i, v in enumerate(coefficient_scores):
    logging.info("Feature: %0d, Score: %.5f" % (i, v))


logging.info(X.columns)
pyplot.bar([x for x in range(len(coefficient_scores))], coefficient_scores)
pyplot.show()

coeff_basic_df = pd.DataFrame(model_basic.coef_[0], X.columns, columns=["Coefficient"])
logging.info("The Coefficient for item", nfri_basic[0], "is:")
coeff_basic_df

# Getting the indices of items on the basis of importance in Descending order

sorted_indices_basic = np.flip(np.argsort(yhat_basic[0]))

sorted_indices_non_basic = np.flip(np.argsort(yhat_non_basic[0]))

logging.info("NFRI Basic sorted on the basis of importance score")
logging.info(itemgetter(*sorted_indices_basic)(nfri_basic))


logging.info("Now time for NFRI Non Basic")
logging.info(itemgetter(*sorted_indices_non_basic)(nfri_non_basic))

logging.info("The importance label of above NFRI Basic")
logging.info([label_transformer(d) for d in -np.sort(-yhat_basic[0])])

logging.info("Now time for NFRI Non Basic")
logging.info([label_transformer(d) for d in -np.sort(-yhat_non_basic[0])])

# Evaluating multioutput regression model with k-fold cross-validation
# Try one each for basic and non-basic respectively (in case of y that is)

# Defining Model
model_test = LinearRegression()
# Setting the cross-validation parameters
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate the model and collecting the scores
n_scores_basic = cross_val_score(
    model_test, X, y_basic, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

logging.info(n_scores_basic)
# Making the scores positive
n_scores_basic_absolute = absolute(n_scores_basic)
# Summarizing the performance
logging.info(
    "MAE of Basic: %.3f (%.3f)" % (mean(n_scores_basic_absolute), std(n_scores_basic))
)

logging.info("Time for some cross validation score")

r2_score_basic = cross_val_score(model_test, X, y_basic, cv=cv, scoring="r2")
logging.info(r2_score_basic)
logging.info("Average R2 Score")
logging.info(np.nanmean(r2_score_basic))

# Repeating the same steps for NFRI_non_basic

n_scores_non_basic = cross_val_score(
    model_test, X, y_non_basic, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)
logging.info(n_scores_non_basic)
n_scores_non_basic_absolute = absolute(n_scores_non_basic)
logging.info(
    "MAE of Non Basic: %.3f (%.3f)"
    % (mean(n_scores_non_basic_absolute), std(n_scores_non_basic))
)

sfs_selector = SequentialFeatureSelector(
    estimator=LinearRegression(), n_features_to_select=10, cv=10, direction="backward"
)
sfs_selector.fit(X, y_basic)
logging.info(X.columns[sfs_selector.get_support()])
