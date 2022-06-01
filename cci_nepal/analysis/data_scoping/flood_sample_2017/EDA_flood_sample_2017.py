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
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Import libraries
import cci_nepal
from cci_nepal.getters.data_scoping import get_sample_data as gsd
from cci_nepal.pipeline.data_scoping import mis_sample_pipeline as msp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# %%
# Set directory
project_directory = cci_nepal.PROJECT_DIR

# %%
feedback_2017 = gsd.read_excel_file(
    f"{project_directory}/inputs/data/wp2_data_scoping/nepal_2017_flood_sample.xlsx"
)

# %%
# Drop survey intro column
feedback_2017.drop(feedback_2017.columns[4], axis=1, inplace=True)
feedback_2017 = msp.clean_df_columns(feedback_2017)  # Calling cleaning function

# %%
feedback_2017.head(1)

# %%
feedback_2017.shape

# %% [markdown]
# ### Questions to answer:
#
# - Area distribution
#     - Distributions of households across districts and small areas
#     - Distribution of NFRI satisfaction scores across districts
#     - Distribution of items delivered
#     - Whats the distribution of NFRI feedback suggestions across districts?
# - A count of NFRI items
#     - A comparison between the distribution of NFRI unique items compared to the items in the ‘NFRI full set’
# - What is the size and spread of the answers to the question on suggested items (prefer to receive)?
#     - How many responses are there in total?
#     - How many can be actually used? - eg some are suggestions for cash or a house so harder to infer an item from
#
#
# - NFRI ratings relationships
#     - Correlation between variables
#     - What features are strongly correlated?

# %%
feedback_2017[
    "Overall are you satisfied with the NFRI distribution process "
].value_counts().plot(kind="bar")
plt.title("Overall are you satisfied with the NFRI distribution process")

# %%
feedback_2017[
    "To what extent has the relief distribution helped improve your current living condition "
].value_counts().plot(kind="bar")
plt.title(
    "To what extent has the relief distribution helped improve your current living condition"
)

# %%
list(feedback_2017.columns)

# %%
useful_cols = [
    "Date of interview daymonthyear ",
    "District ",
    "VDCMunicipality ",
    "Ward No",
    "GPSlatitude",
    "GPSlongitude",
    "GPSaltitude",
    "GPSprecision",
    "What is the gender of the respondent ",
    "Is respondent head of household",
    "What is the respondents relationship to the head of household",
    "If others specify ",
    "What is the age of the respondent ",
    "Is the respondent the person who received the relief items ",
    "What is the ethnicity of the household",
    "If others specify ",
    "If Male 0-5   - How many",
    "If Male 6-17   - How many",
    "If Male 18-59  - How many",
    "If Male 60     How many",
    "If Female 0-5  - How many",
    "If Female 6-17   - How many",
    "If Female 18-59  -  How many",
    "If Female 60    How many",
    "People with disability member in HHs ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudDo not know  ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudFemale head of household   ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudFamily member with chronic diseasedisability   ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudYoung children in house    ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudPregnant or Lactating women     ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudElderly household members    ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHouse was destroyed  ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHousehold members from low caste  ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHouse badly damaged    ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHousehold very poor    ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHousehold members unemployed   ",
    "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudOthers ",
    "If others please mention ",
    "Do you think all the people in your community who needed assistance were included in the beneficiary lists",
    "What has your household received from the Nepal Red Cross since the floods Select all that apply",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyKitchen Set  ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyShelter Tool Kit  ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyTarps  ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyBlankets ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyOther Household Items  ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyOther ",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  1",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyNFRI full set",
    "What has your household received from the Nepal Red Cross since the floods Select all that applyCash",
    "If NFRI full set",
    "If NFRI full setKitchen Set  ",
    "If NFRI full setTarps  ",
    "If NFRI full setBlankets ",
    "If NFRI full setSari",
    "If NFRI full setPlain clothes",
    "If NFRI full setPrinted clothes",
    "If NFRI full setBucket",
    "If NFRI full setSheeting clothes",
    "If NFRI full setDori ",
    "If NFRI full setSack",
    "If NFRI full setMale Dhoti",
    "If NFRI full setSuites",
    "If WASHHygiene Supplies  ",
    "If WASHHygiene Supplies  Bucket",
    "If WASHHygiene Supplies  Soap",
    "If WASHHygiene Supplies  ORS  ",
    "If WASHHygiene Supplies  Acitap",
    "If other specify",
    "How have you used the other items supplied to you by NRCS kitchen set blankets WASH supplies Multiple responses allowed",
    "How have you used the other items supplied to you by NRCS kitchen set blankets WASH supplies Multiple responses allowedUsed in household  ",
    "How have you used the other items supplied to you by NRCS kitchen set blankets WASH supplies Multiple responses allowedShared with familyfriends  ",
    "How have you used the other items supplied to you by NRCS kitchen set blankets WASH supplies Multiple responses allowedOther ",
    "If other please specify",
    "Has receiving these supplies caused conflict within your household ",
    "Are you facing any conflicts with other community members because of the relief items you received ",
    "If this support for flood would be given again what would you prefer to receiveCash ",
    "If this support for flood would be given again what would you prefer to receiveRelief Items     ",
    "If this support for flood would be given again what would you prefer to receiveOthers ",
    "If others please mention1",
    "Overall are you satisfied with the NFRI distribution process ",
    "If NOT AT ALL why",
    "To what extent has the relief distribution helped improve your current living condition ",
    "Please explain if answer is slightly or not at all",
    "What is your suggestions on how to improve our relief distribution in future",
]

# %%
fbk = feedback_2017[useful_cols].copy()

# %%
fbk["District "].value_counts()

# %%
((fbk["District "].value_counts() / 1092) * 100).plot(kind="bar")

# %%
fbk["VDCMunicipality "].value_counts()

# %%
fbk.head(1)

# %% [markdown]
# #### Area distribution

# %%
area_items = fbk[
    [
        "VDCMunicipality ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyKitchen Set  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyShelter Tool Kit  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyTarps  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyBlankets ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyOther Household Items  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyOther ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  1",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyNFRI full set",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyCash",
        "If NFRI full set",
        "If NFRI full setKitchen Set  ",
        "If NFRI full setTarps  ",
        "If NFRI full setBlankets ",
        "If NFRI full setSari",
        "If NFRI full setPlain clothes",
        "If NFRI full setPrinted clothes",
        "If NFRI full setBucket",
        "If NFRI full setSheeting clothes",
        "If NFRI full setDori ",
        "If NFRI full setSack",
        "If NFRI full setMale Dhoti",
        "If NFRI full setSuites",
        "If WASHHygiene Supplies  ",
        "If WASHHygiene Supplies  Bucket",
        "If WASHHygiene Supplies  Soap",
        "If WASHHygiene Supplies  ORS  ",
        "If WASHHygiene Supplies  Acitap",
    ]
]

# %%
area_items = area_items.fillna(0)

# %%
area_items.head(1)

# %%
area_items_grouped = area_items.groupby(by="VDCMunicipality ").sum()

# %%
area_items_grouped

# %% [markdown]
# #### Satisfaction correlation

# %%
satisfaction = fbk[
    [
        "Overall are you satisfied with the NFRI distribution process ",
        "To what extent has the relief distribution helped improve your current living condition ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyKitchen Set  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyShelter Tool Kit  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyTarps  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyBlankets ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyOther Household Items  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyOther ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  1",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyNFRI full set",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyCash",
        "If NFRI full setKitchen Set  ",
        "If NFRI full setTarps  ",
        "If NFRI full setBlankets ",
        "If NFRI full setSari",
        "If NFRI full setPlain clothes",
        "If NFRI full setPrinted clothes",
        "If NFRI full setBucket",
        "If NFRI full setSheeting clothes",
        "If NFRI full setDori ",
        "If NFRI full setSack",
        "If NFRI full setMale Dhoti",
        "If NFRI full setSuites",
        "If WASHHygiene Supplies  Bucket",
        "If WASHHygiene Supplies  Soap",
        "If WASHHygiene Supplies  ORS  ",
        "If WASHHygiene Supplies  Acitap",
    ]
]

# %%
satisfaction.columns = [
    "Satisfaction",
    "Extent relief has improved living condition",
    "Kitchen Set",
    "Shelter Tool Kit",
    "Tarps",
    "Blankets",
    "WASHHygiene Supplies",
    "Other Household Items",
    "Other",
    "WASHHygiene Supplies2",
    "NFRI full set",
    "Cash",
    "NFRI-full Kitchen Set",
    "NFRI-full Tarps",
    "NFRI-full Blankets",
    "NFRI-full Sari",
    "NFRI-full Plain clothes",
    "NFRI-full Printed clothes",
    "NFRI-full Bucket",
    "NFRI-full Sheeting clothes",
    "NFRI-full Dori ",
    "NFRI-full Sack",
    "NFRI-full Male Dhoti",
    "NFRI-full Suites",
    "WASHHygiene Bucket",
    "WASHHygiene Soap",
    "WASHHygiene ORS  ",
    "WASHHygiene Acitap",
]

# %%
satisfaction = satisfaction.fillna(0)

# %%
impr = satisfaction.groupby(by=["Extent relief has improved living condition"]).mean()
satis = satisfaction.groupby(by=["Satisfaction"]).mean()

# %%
impr.T.plot(figsize=(15, 5), kind="bar")

# %%
satis.T.plot(figsize=(15, 5), kind="bar")

# %%
satisfaction.head(1)

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(satisfaction.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %% [markdown]
# ### Predicting extent relief relieved living condition

# %%
X = satisfaction[
    satisfaction.columns.difference(
        ["Extent relief has improved living condition", "Satisfaction"]
    )
]
y = satisfaction[["Extent relief has improved living condition"]]

# %%
X, y = shuffle(X, y)

# %%
model = ExtraTreesClassifier()
model.fit(X, y.values.ravel())
print(
    model.feature_importances_
)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind="barh")
plt.show()

# %%
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# %%
clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(x_train, y_train)

# %%
clf

# %%
# Returns a NumPy Array
# Predict for One Observation (image)
clf.predict(x_test[0:10])

# %%
predictions = clf.predict(x_test)

# %%
# Use score method to get accuracy of model
score = clf.score(x_test, y_test)
print(score)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %%
y_test.value_counts()

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm

# %%
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Moderately", "Not at all", "Significantly", "Slightly"]
)

# %%
cm.plot()

# %%
smote = SMOTE(random_state=42)

# %%
x_train, y_train = smote.fit_resample(x_train, y_train)

# %%
clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(x_train, y_train)

# %%
predictions = clf.predict(x_test)

# %%
# Use score method to get accuracy of model
score = clf.score(x_test, y_test)
print(score)

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm

# %%
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Moderately", "Not at all", "Significantly", "Slightly"]
)

# %%
cm.plot()

# %% [markdown]
# ### Predicting satisfaction

# %%
X = satisfaction[
    satisfaction.columns.difference(
        ["Extent relief has improved living condition", "Satisfaction"]
    )
]
y = satisfaction[["Satisfaction"]]
X, y = shuffle(X, y)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# %%
model = ExtraTreesClassifier()
model.fit(X, y.values.ravel())
print(
    model.feature_importances_
)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind="barh")
plt.show()

# %%
clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(x_train, y_train)
print(clf.predict(x_test[0:10]))
predictions = clf.predict(x_test)
score = clf.score(x_test, y_test)
print(score)

# %%
y_test.value_counts()

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm

# %%
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Not at all   ", "Somewhat   ", "Yes  completely   "]
)

# %%
cm.plot()

# %%
clf.estimators_[0].coef_

# %% [markdown]
# ### Merging columns

# %%
satisfaction["WASHHygiene Supplies"] = (
    satisfaction["WASHHygiene Supplies"] + satisfaction["WASHHygiene Supplies2"]
)
satisfaction["Kitchen Set"] = (
    satisfaction["Kitchen Set"] + satisfaction["NFRI-full Kitchen Set"]
)
satisfaction["Tarps"] = satisfaction["Tarps"] + satisfaction["NFRI-full Tarps"]
satisfaction["Blankets"] = satisfaction["Blankets"] + satisfaction["NFRI-full Blankets"]
satisfaction["Bucket"] = (
    satisfaction["NFRI-full Bucket"] + satisfaction["WASHHygiene Bucket"]
)

satisfaction.drop(
    [
        "WASHHygiene Supplies2",
        "Cash",
        "NFRI-full Kitchen Set",
        "NFRI-full Tarps",
        "NFRI-full Blankets",
        "WASHHygiene Bucket",
        "NFRI-full Bucket",
    ],
    axis=1,
    inplace=True,
)


# %%
satisfaction.columns = [
    "Satisfaction",
    "Extent improved living",
    "Kitchen Set",
    "Shelter Tool Kit",
    "Tarps",
    "Blankets",
    "Hygiene Supplies",
    "Other Household Items",
    "Other",
    "NFRI full set",
    "Sari",
    "Plain clothes",
    "Printed clothes",
    "Sheeting clothes",
    "Dori ",
    "Sack",
    "Male Dhoti",
    "NFRI-full Suites",
    "Hygiene Soap",
    "Hygiene ORS  ",
    "WASHHygiene Acitap",
    "Bucket",
]

# %%
satisfaction.head(1)

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(satisfaction.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
X = satisfaction[
    satisfaction.columns.difference(["Extent improved living", "Satisfaction"])
]
y = satisfaction[["Satisfaction"]]

# %%
model = ExtraTreesClassifier()
model.fit(X, y.values.ravel())
print(
    model.feature_importances_
)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind="barh")
plt.show()

# %%
X, y = shuffle(X, y)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(x_train, y_train)
predictions = clf.predict(x_test)
score = clf.score(x_test, y_test)
print(score)

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Not at all   ", "Somewhat   ", "Yes  completely   "]
)
cm.plot()

# %% [markdown]
# #### Random forest

# %%
# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(
    n_estimators=10, criterion="entropy", random_state=42
)
classifier.fit(x_train, y_train.values.ravel())

# %%
predictions = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
print(score)

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Not at all   ", "Somewhat   ", "Yes  completely   "]
)
cm.plot()

# %%
X = satisfaction[
    satisfaction.columns.difference(["Extent improved living", "Satisfaction"])
]
y = satisfaction[["Extent improved living"]]

# %%
model = ExtraTreesClassifier()
model.fit(X, y.values.ravel())
print(
    model.feature_importances_
)  # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind="barh")
plt.show()

# %%
X, y = shuffle(X, y)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(x_train, y_train)
predictions = clf.predict(x_test)
score = clf.score(x_test, y_test)
print(score)

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Moderately", "Not at all", "Significantly", "Slightly"]
)
cm.plot()

# %% [markdown]
# #### Random forest

# %%
# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(
    n_estimators=10, criterion="entropy", random_state=42
)
classifier.fit(x_train, y_train.values.ravel())

# %%
predictions = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
print(score)

# %%
cm = metrics.confusion_matrix(y_test, predictions)
cm = ConfusionMatrixDisplay(
    cm, display_labels=["Moderately", "Not at all", "Significantly", "Slightly"]
)
cm.plot()

# %% [markdown]
# ### Cluster by household / area

# %%
fbk.head(2)

# %%
import geopandas as gpd
from shapely.geometry import Point, Polygon

# %%
crs = {"init": "epsg:4326"}

# %%
geometry = [Point(xy) for xy in zip(fbk["GPSlatitude"], fbk["GPSlongitude"])]

# %%
geometry[:3]

# %%
df = gpd.GeoDataFrame(fbk, crs=crs, geometry=geometry)

# %%
df.head(1)

# %% [markdown]
# ### 11 / Oct notes:
#
# - First step: demographic, item, satisfaction score
# - Combinations of items and demographic characteristics - different levels of satisfaction
# - Heatmap: demographic one side, items other side
# - Analysis of people not satisfied
# - Difference significantly to not at all?
#     - Does it just correlate with number of items
# - Do the RC already distribute items according to perceived need
#     - Eg different items for women in household compared to majority men
#     - Correspond to different levels of satisfaction

# %%
fbk.head(1)

# %%
demog = fbk[
    [
        "What is the ethnicity of the household",
        "If Male 0-5   - How many",
        "If Male 6-17   - How many",
        "If Male 18-59  - How many",
        "If Male 60     How many",
        "If Female 0-5  - How many",
        "If Female 6-17   - How many",
        "If Female 18-59  -  How many",
        "If Female 60    How many",
        "People with disability member in HHs ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudDo not know  ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudFemale head of household   ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudFamily member with chronic diseasedisability   ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudYoung children in house    ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudPregnant or Lactating women     ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudElderly household members    ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHouse was destroyed  ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHousehold members from low caste  ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHouse badly damaged    ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHousehold very poor    ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudHousehold members unemployed   ",
        "What were the selection criteria for receiving this assistance Select all that apply  do not read aloudOthers ",
        "To what extent has the relief distribution helped improve your current living condition ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyKitchen Set  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyShelter Tool Kit  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyTarps  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyBlankets ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyOther Household Items  ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyOther ",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyWASHHygiene Supplies  1",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyNFRI full set",
        "What has your household received from the Nepal Red Cross since the floods Select all that applyCash",
        "If NFRI full setKitchen Set  ",
        "If NFRI full setTarps  ",
        "If NFRI full setBlankets ",
        "If NFRI full setSari",
        "If NFRI full setPlain clothes",
        "If NFRI full setPrinted clothes",
        "If NFRI full setBucket",
        "If NFRI full setSheeting clothes",
        "If NFRI full setDori ",
        "If NFRI full setSack",
        "If NFRI full setMale Dhoti",
        "If NFRI full setSuites",
        "If WASHHygiene Supplies  Bucket",
        "If WASHHygiene Supplies  Soap",
        "If WASHHygiene Supplies  ORS  ",
        "If WASHHygiene Supplies  Acitap",
    ]
].copy()

# %%
demog.head(1)

# %%
demog.columns = [
    "Ethnicity",
    "Male 0-5",
    "Male 6-17",
    "Male 18-59",
    "Male 60",
    "Female 0-5",
    "Female 6-17",
    "Female 18-59",
    "Female 60",
    "Disabilities",
    "Selection Do not know ",
    "Selection Female head of household",
    "Selection Family member with chronic disease disability",
    "Selection Young children in house",
    "Selection Pregnant or Lactating women",
    "Selection Elderly household members",
    "Selection House was destroyed",
    "Selection Household members from low caste",
    "Selection House badly damaged",
    "Selection Household very poor",
    "Selection Household members unemployed",
    "Selection Others",
    "Extent relief has improved living condition",
    "Kitchen Set",
    "Shelter Tool Kit",
    "Tarps",
    "Blankets",
    "WASHHygiene Supplies",
    "Other Household Items",
    "Other",
    "WASHHygiene Supplies2",
    "NFRI full set",
    "Cash",
    "NFRI-full Kitchen Set",
    "NFRI-full Tarps",
    "NFRI-full Blankets",
    "NFRI-full Sari",
    "NFRI-full Plain clothes",
    "NFRI-full Printed clothes",
    "NFRI-full Bucket",
    "NFRI-full Sheeting clothes",
    "NFRI-full Dori ",
    "NFRI-full Sack",
    "NFRI-full Male Dhoti",
    "NFRI-full Suites",
    "WASHHygiene Bucket",
    "WASHHygiene Soap",
    "WASHHygiene ORS  ",
    "WASHHygiene Acitap",
]

# %%
demog = demog.fillna(0).copy()

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(demog.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
demog["WASHHygiene Supplies"] = (
    demog["WASHHygiene Supplies"] + demog["WASHHygiene Supplies2"]
)
demog["Kitchen Set"] = demog["Kitchen Set"] + demog["NFRI-full Kitchen Set"]
demog["Tarps"] = demog["Tarps"] + demog["NFRI-full Tarps"]
demog["Blankets"] = demog["Blankets"] + demog["NFRI-full Blankets"]
demog["Bucket"] = demog["NFRI-full Bucket"] + demog["WASHHygiene Bucket"]

demog.drop(
    [
        "WASHHygiene Supplies2",
        "Cash",
        "NFRI-full Kitchen Set",
        "NFRI-full Tarps",
        "NFRI-full Blankets",
        "WASHHygiene Bucket",
        "NFRI-full Bucket",
    ],
    axis=1,
    inplace=True,
)

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(demog.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
demog["Household size"] = demog.iloc[:, 1:9].sum(axis=1)

# %%
demog.iloc[:, 5:9].sum(axis=1)

# %%
demog["Total female"] = demog.iloc[:, 5:9].sum(axis=1)

# %%
demog["Percent female"] = demog["Total female"] / demog["Household size"]

# %%
demog["Majority female"] = np.where(demog["Percent female"] > 0.5, 1, 0)

# %%
demog = pd.get_dummies(demog, columns=["Ethnicity"])

# %%
demog.head(1)

# %%
print(demog.columns)

# %%
heat = demog[
    [
        "Majority female",
        "Ethnicity_AdibasiJanjatiNewar         ",
        "Ethnicity_Brahmin  Chettri  Sanyashi  Thakuri       ",
        "Ethnicity_Dalit  ",
        "Ethnicity_Madhesi ",
        "Ethnicity_Other specify    ",
        "Selection Female head of household",
        "Selection Family member with chronic disease disability",
        "Selection Young children in house",
        "Selection Pregnant or Lactating women",
        "Selection Elderly household members",
        "Selection House was destroyed",
        "Selection Household members from low caste",
        "Selection House badly damaged",
        "Selection Household very poor",
        "Selection Household members unemployed",
        "Extent relief has improved living condition",
        "Kitchen Set",
        "Shelter Tool Kit",
        "Tarps",
        "Blankets",
        "WASHHygiene Supplies",
        "Other Household Items",
        "Other",
        "NFRI full set",
        "NFRI-full Sari",
        "NFRI-full Plain clothes",
        "NFRI-full Printed clothes",
        "NFRI-full Sheeting clothes",
        "NFRI-full Dori ",
        "NFRI-full Sack",
        "NFRI-full Male Dhoti",
        "NFRI-full Suites",
        "WASHHygiene Soap",
        "WASHHygiene ORS  ",
        "WASHHygiene Acitap",
        "Bucket",
    ]
]

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(heat.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
