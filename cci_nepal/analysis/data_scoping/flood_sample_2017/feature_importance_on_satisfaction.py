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
#     display_name: Python 3 (ipykernel)
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# %%
# #!pip3 install scikit-learn
# #!pip install imbalanced-learn

# %%
# Set directory
project_directory = cci_nepal.PROJECT_DIR

# %%
feedback_2017 = gsd.read_excel_file(
    f"{project_directory}/inputs/data/Other_database.xlsx"
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
# ### Questions to answer
#
# What impact does the number of items distributed have on NFRI satisfaction / living improvement per household?
#
# Is there a strong correlation between number of items (or avg items per HH member) and satisfaction / living improvement score
# Recode satisfaction scores as ordinal (1 - low, 4 - high) and plot them against avg items per HH size
# To what extent do other factors relating to their distribution experience (eg timings, distance to travel) affect the NFRI satisfaction / living improvement per household?
#
# Look at feature importance across the whole dataset with either satisfaction or living improvement as the predictors
#
#   - What features are most important in predicting?

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
set(feedback_2017["When were you informed of the date of distribution"])

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
((fbk["District "].value_counts() / 1092) * 100).plot(kind="bar")

# %%
fbk["VDCMunicipality "].value_counts()

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

# %% [markdown]
# What impact does the number of items distributed have on NFRI satisfaction / living improvement per household?
#
# Is there a strong correlation between number of items (or avg items per HH member) and satisfaction / living improvement score Recode satisfaction scores as ordinal (1 - low, 4 - high) and plot them against avg items per HH size To what extent do other factors relating to their distribution experience (eg timings, distance to travel) affect the NFRI satisfaction / living improvement per household?
#
# Look at feature importance across the whole dataset with either satisfaction or living improvement as the predictors
#
# What features are most important in predicting?

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

# %%
# count of items received by each household.
satisfaction["item_counts"] = satisfaction[
    [
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
].sum(axis=1)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# %%
set(feedback_2017["When were you informed of the date of distribution"])

# %%
satisfaction["timing"] = feedback_2017[
    "When were you informed of the date of distribution"
].replace(
    {
        "1 month or more before       ": 30,
        "1 week before   ": 7,
        "1-2 days before     ": 2,
        "2 weeks before   ": 14,
        "2-3 weeks before     ": 21,
        "3-6 days before     ": 6,
        "On the day of distribution   ": 0,
    }
)

# %%
set(satisfaction.timing)

# %%
# normalize the days/timing
satisfaction.timing = (satisfaction.timing - satisfaction.timing.min()) / (
    satisfaction.timing.max() - satisfaction.timing.min()
)


# %%
satisfaction["recode_satisfaction"] = satisfaction.Satisfaction.replace(
    {"Not at all ": 1, "Somewhat   ": 2, "Yes  completely   ": 3}
)
# if package was received on time or not
satisfaction["received_on_time"] = feedback_2017[
    "Did you receive your assistance on time"
].replace({"Yes ": 1, "No ": 0})

# %%
# prepare data to use and extract feature importance.
y1 = satisfaction.recode_satisfaction
X1 = satisfaction[
    satisfaction.columns.difference(
        [
            "Extent relief has improved living condition",
            "Satisfaction",
            "recode_satisfaction",
            "timing",
        ]
    )
]

# %%
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.20, random_state=0
)

# %%

# %%
model = DecisionTreeClassifier()
# fit the model
model.fit(x_train1, y_train1)
# get importance
importance = model.feature_importances_

# %%
importance

# %%
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(15).plot(kind="barh")
plt.title("Feature importance for decision tree")
plt.show()

# %%
yp_dt = model.predict(x_test1)
accuracy_score(yp_dt, y_test1)

# %%
# random forest
rf = RandomForestClassifier()
rf.fit(x_train1, y_train1)
# get feature importance
rf_importance = rf.feature_importances_

# %%
lm = LinearRegression()
lm.fit(x_train1, y_train1)

# %%
feat_coefficients = pd.Series(lm.coef_, index=x_train1.columns)
feat_coefficients.plot(figsize=(15, 5), kind="bar")
plt.title("Feature coefficients for NFRI Satisfaction")
plt.show()

# %%
lm.coef_

# %%
x_train1.columns

# %%
satisfaction["Extent relief has improved living condition"] = satisfaction[
    "Extent relief has improved living condition"
].replace(
    {"Not at all    ": 0, "Slightly  ": 1, "Moderately   ": 2, "Significantly  ": 3}
)

# %%
# prepare data to use and extract feature importance.
y3 = satisfaction["Extent relief has improved living condition"]
X3 = satisfaction[
    satisfaction.columns.difference(
        [
            "Extent relief has improved living condition",
            "Satisfaction",
            "recode_satisfaction",
        ]
    )
]

# %%
lm2 = LinearRegression()
lm2.fit(X3, y3)

# %%
feat_coefficients = pd.Series(lm2.coef_, index=X3.columns)
feat_coefficients.plot(figsize=(15, 5), kind="bar")
plt.title("Feature coefficients with extent relief has improved  living condition")
plt.show()

# %%
lm2.coef_

# %%
X3.columns

# %%
yp_rf = rf.predict(x_test1)
accuracy_score(yp_rf, y_test1)

# %%
rf_importance

# %%
# plot graph of feature importances for better visualization
feat_importances = pd.Series(rf.feature_importances_, index=X1.columns)
feat_importances.nlargest(10).plot(kind="barh")
plt.title("Feature importance for random forest")
plt.show()

# %%
# prepare data to use and extract feature importance.
y2 = satisfaction.recode_satisfaction
X2 = satisfaction[
    satisfaction.columns.difference(
        [
            "Extent relief has improved living condition",
            "Satisfaction",
            "recode_satisfaction",
        ]
    )
]

# %%
x_train2, x_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.20, random_state=0
)

# %%
model2 = DecisionTreeClassifier()
# fit the model
model2.fit(x_train2, y_train2)

# %%
yp_dt2 = model2.predict(x_test2)

# %%
accuracy_score(yp_dt2, y_test2)

# %%
# random forest with timing data included
rf2 = RandomForestClassifier()
rf2.fit(x_train2, y_train2)
# get feature importance
rf2_importance = rf2.feature_importances_

# %%
yp_rf2 = rf2.predict(x_test2)
accuracy_score(yp_rf2, y_test2)

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(satisfaction.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
plt.scatter(satisfaction.recode_satisfaction, satisfaction.item_counts)
plt.show()

# %%
set(satisfaction.item_counts)

# %%
satisfaction[["item_counts", "Satisfaction"]].groupby("Satisfaction").mean().plot(
    kind="bar", figsize=(15, 10), fontsize=12
)
plt.title("Average number of items household received vs satisfaction")

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(satisfaction.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
impr = satisfaction.groupby(by=["Extent relief has improved living condition"]).mean()
satis = satisfaction.groupby(by=["Satisfaction"]).mean()

# %%

plt.plot(satisfaction["item_counts"], satisfaction["recode_satisfaction"])

# %%
plt.scatter(satisfaction["item_counts"], satisfaction["recode_satisfaction"])

# %%
sat1 = []
sat2 = []
sat3 = []
for item in set(satisfaction.item_counts):
    temp = satisfaction[(satisfaction["item_counts"] == item)]
    for val in range(1, 4):
        d = temp[(satisfaction["recode_satisfaction"] == val)]
        if val == 1:
            sat1.append(d.shape[0] / temp.shape[0])
        if val == 2:
            sat2.append(d.shape[0] / temp.shape[0])
        if val == 3:
            sat3.append(d.shape[0] / temp.shape[0])


# %%
set(satisfaction.Satisfaction)

# %%
proportion_df = pd.DataFrame(
    {"Not at all": sat1, "somewhat": sat2, "Yes Completely": sat3},
    index=set(satisfaction.item_counts),
)

# %%
proportion_df.plot(figsize=(15, 5))
plt.title(
    "Proportion of beneficiary who recorded satisfaction score vs number of items household received"
)

# %%
plt.figure(figsize=(14, 12))
sns.heatmap(satisfaction.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%
Xt = satisfaction[
    satisfaction.columns.difference(
        [
            "Extent relief has improved living condition",
            "Satisfaction",
            "recode_satisfaction",
        ]
    )
]
yt = satisfaction[["Extent relief has improved living condition"]]

# %%
Xt, yt = shuffle(Xt, yt)

# %%
clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(Xt, yt)

# %%
clf.estimators_[0].coef_

# %%
Xt.shape

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
y.values.ravel()

# %%
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# %%
y_test.value_counts()

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
plt.figure(figsize=(14, 12))
sns.heatmap(satisfaction.corr(), linewidths=0.1, cmap="YlGnBu")
plt.yticks(rotation=0)

# %%

# %%
useful_cols

# %% [markdown]
# ## Issue 12 handled from here onward

# %%
useful_col_df = feedback_2017[useful_cols]

# %%
df_comsatisfied = useful_col_df[
    useful_col_df["Overall are you satisfied with the NFRI distribution process "]
    == "Yes  completely   "
]
df_notsatisfied = useful_col_df[
    useful_col_df["Overall are you satisfied with the NFRI distribution process "]
    == "Not at all "
]

# %%

# %%
# plot of gender distribution on NFRI package satisfaction
fig, axes = plt.subplots(1, 2)
fig.set_figheight(7)
fig.set_figwidth(15)
df_notsatisfied["What is the gender of the respondent "].value_counts().plot(
    ax=axes[0], width=0.6, kind="bar"
)
axes[0].set_title("Not satisfied - by gender")
df_comsatisfied["What is the gender of the respondent "].value_counts().plot(
    ax=axes[1], kind="bar"
)
axes[1].set_title("Completely Satisfied")
plt.show()

# %%
# ethnicity distribution
fig, axes = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(15)
df_notsatisfied["What is the ethnicity of the household"].value_counts().plot(
    ax=axes[0], width=0.6, kind="bar"
)
axes[0].set_title("Not satisfied ethnicity")
df_comsatisfied["What is the ethnicity of the household"].value_counts().plot(
    ax=axes[1], kind="bar"
)
axes[1].set_title("Completely Satisfied")
plt.show()

# %%
# People with disability distribution
fig, axes = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(15)
df_notsatisfied["People with disability member in HHs "].value_counts().plot(
    ax=axes[0], width=0.6, kind="bar"
)
axes[0].set_title("People with disability not satisfied")
df_comsatisfied["People with disability member in HHs "].value_counts().plot(
    ax=axes[1], kind="bar"
)
axes[1].set_title("People with disability completely satisfied")
plt.show()

# %%
set(
    df_comsatisfied[
        "To what extent has the relief distribution helped improve your current living condition "
    ]
)

# %%
# splitting the data based on how live has been improved
significantly_improved_df = useful_col_df[
    useful_col_df[
        "To what extent has the relief distribution helped improve your current living condition "
    ]
    == "Significantly  "
]
df_notimproved = useful_col_df[
    useful_col_df[
        "To what extent has the relief distribution helped improve your current living condition "
    ]
    == "Not at all    "
]

# %%
# plot to show distribution of people by gender who whose lives were not changed at all or improved significantly
fig, axes = plt.subplots(1, 2)
fig.set_figheight(7)
fig.set_figwidth(15)
significantly_improved_df["What is the gender of the respondent "].value_counts().plot(
    ax=axes[0], width=0.6, kind="bar"
)
axes[0].set_title("Significantly improved - by gender")
df_notimproved["What is the gender of the respondent "].value_counts().plot(
    ax=axes[1], kind="bar"
)
axes[1].set_title("Not at all ")
plt.show()

# %%
# ethnicity distribution
fig, axes = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(15)
significantly_improved_df["What is the ethnicity of the household"].value_counts().plot(
    ax=axes[0], width=0.6, kind="bar"
)
axes[0].set_title("Significantly improved - by ethnicity")
df_notimproved["What is the ethnicity of the household"].value_counts().plot(
    ax=axes[1], kind="bar"
)
axes[1].set_title("Not at all ")
plt.show()

# %%
# People with disability distribution
fig, axes = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(15)
significantly_improved_df["People with disability member in HHs "].value_counts().plot(
    ax=axes[0], width=0.6, kind="bar"
)
axes[0].set_title("Significantly improved - with disability")
df_notimproved["People with disability member in HHs "].value_counts().plot(
    ax=axes[1], kind="bar"
)
axes[1].set_title("Not at all ")
plt.show()

# %%
new_df = pd.read_csv(f"{project_directory}/inputs/data/Bipad.csv")

# %%
new_df.head()

# %%
new_df.shape

# %%
list(new_df.columns)

# %%
new_df["House affected"].value_counts()

# %%
new_df["Unknown - People Injured"].value_counts().plot(kind="bar")

# %%
new_df["Incident on"].min()

# %%
new_df["Incident on"].max()

# %%

# %%
plt.figure(figsize=(10, 5))
new_df["Hazard"].value_counts().plot(kind="bar")
plt.title("Hazard distribution")

# %%
new_df[
    [
        "Total estimated loss (NPR)",
        "Agriculture economic loss (NPR)",
        "Infrastructur economic loss (NPR)",
    ]
].sum().plot(kind="barh")
plt.title("Losses incurred")


# %%
new_df[
    [
        "Total infrastructure destroyed",
        "House destroyed",
        "House affected",
        "Total livestock destroyed",
    ]
].sum().plot(kind="bar")
plt.title("Value of things destroyed")

# %%
new_df[
    [
        "Total livestock destroyed",
        "Total - People Death",
        "Male - People Death",
        "Female - People Death",
        "Unknown - People Death",
        "Disabled - People Death",
        "Total - People Missing",
        "Male - People Missing",
        "Female - People Missing",
        "Unknown - People Missing",
        "Disabled - People Missing",
        "Total - People Injured",
        "Male - People Injured",
        "Female - People Injured",
        "Unknown - People Injured",
        "Disabled - People Injured",
    ]
].sum().plot(kind="bar")
