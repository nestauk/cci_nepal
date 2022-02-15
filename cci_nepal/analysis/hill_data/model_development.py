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
# ### Things to test
#
# - Rebalancing
# - Classification versus regression models
# - Test performance of individual models of specific items
# - Trial different features

# %%
import pandas as pd
import numpy as np

import warnings
import logging

from matplotlib import pyplot as plt
import seaborn as sns

import cci_nepal
from cci_nepal.getters.real_data import get_real_data as grd
from cci_nepal.pipeline.real_data import data_manipulation as dm
from cci_nepal.pipeline.real_data import nfri_list_file as nlf

from cci_nepal.pipeline.dummy_data import model_manipulation as mm

# %%
pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = [10, 6]

# %%
project_dir = cci_nepal.PROJECT_DIR

# %%
df = grd.read_csv_file(f"{project_dir}/inputs/data/real_data/Full_Consent_Hill.csv")

# %%
df.head(1)

# %%
columns_to_drop = [
    0,
    1,
    2,
    3,
    4,
    7,
    18,
    31,
    41,
    43,
    44,
    56,
    57,
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

# %%
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

# %%
# Removing the new NFRI items section for now
df = df.iloc[:, :-4]

# %%
df.head(5)

# %%
features = [
    "gender_respondent",
    "age_respondent",
    "male_0_5",
    "male_6_12",
    "male_13_17",
    "male_18_29",
    "male_30_39",
    "male_40_49",
    "male_50_59",
    "male_60_69",
    "male_70_79",
    "male_80",
    "female_0_5",
    "female_6_12",
    "female_13_17",
    "female_18_29",
    "female_30_39",
    "female_40_49",
    "female_50_59",
    "female_60_69",
    "female_70_79",
    "female_80",
    "hh_ethnicity",
    "ethnicity_other",
    "hh_sight difficulty",
    "hh_hearing_difficulty",
    "hh_walking_difficulty",
    "hh_memory_difficulty",
    "hh_self_care_difficulty",
    "hh_communicating_difficulty",
    "house_material",
    "house_material_other",
    "income_generating_members",
    "previous_nfri",
]

nfri_items = [
    "plastic_tarpaulin",
    "blanket",
    "sari",
    "male_dhoti",
    "shouting_cloth_jeans",
    "printed_cloth",
    "terry_cloth",
    "utensil_set",
    "water_bucket",
    "nylon_rope",
    "sack_packing_bag",
    "cotton_towel",
    "bathing_soap",
    "laundry_soap",
    "toothbrush_paste",
    "sanitary_pad",
    "ladies_underwear",
    "torch_light",
    "whistle_blow",
    "nail_cutter",
    "hand_sanitizer",
    "liquid_chlorine",
]

# %%
basic = nfri_items[0:11]
non_basic = nfri_items[11:]

# %%
df.columns = features + nfri_items

# %%
df.head(5)

# %%
df.isnull().sum()[df.isnull().sum() > 0].sort_values().plot(kind="barh")

# %% [markdown]
# #### Ethnicity
# - Very low counts per group (other groups)
# - prefer not to say is a very low count

# %%
df.ethnicity_other.value_counts()

# %%
df.shape

# %%
df.hh_ethnicity.value_counts()

# %% [markdown]
# #### House materials
# - 3% Mato ghar
# - the rest a low count on house material
# - some duplicate values written slightly differently
# - some values the same as house material categories (so need to be recoded)

# %%
df.house_material.value_counts()

# %%
df.house_material_other.value_counts()

# %%
(62 / 1722) * 100

# %%
# Update house materials column
cemment_with_bricks = [
    "Bricks with ciment",
    "btickets with ciment",
    "brickets with ciment",
]
df.loc[
    df.house_material_other.isin(cemment_with_bricks), "house_material"
] = "Cement bonded bricks/stone"

# %%
df.drop(["house_material_other", "ethnicity_other"], axis=1, inplace=True)

# %%
# Fill missing with 0
df.fillna(0, inplace=True)

# %% [markdown]
# ### Features to build
# - Income
#     - generating adults (%)
#     - Ratio income generating to hh size
# - Age/gender
#     - % Male
#     - % Female
#     - Children Y/N (and/or Children < 5 Y/N?)
#     - Single adult hh (dropped as too small)
#     - Adult to child ratio?

# %%
# Household size

# %%
# Adding hh size column
df.insert(2, "household_size", df.iloc[:, 2:22].sum(axis=1))
df.household_size = df.household_size + 1  # To include the respondent

# %% [markdown]
# Percent male / female

# %%
df.insert(3, "total_male", df.iloc[:, 3:13].sum(axis=1))
df.insert(4, "total_female", df.iloc[:, 14:24].sum(axis=1))

# %%
df.total_female = np.where(
    df.gender_respondent == "Female", (df.total_female + 1), df.total_female
)
df.total_male = np.where(
    df.gender_respondent == "Male", (df.total_male + 1), df.total_male
)

# %%
df.insert(5, "percent_male", (df.total_male / df.household_size) * 100)
df.insert(6, "percent_female", (df.total_female / df.household_size) * 100)

# %%
df.drop(["total_male", "total_female"], axis=1, inplace=True)

# %%
# Show 4 different binwidths
for i, binwidth in enumerate([1, 5, 10, 15]):

    # Set up the plot
    ax = plt.subplot(2, 2, i + 1)

    # Draw the plot
    ax.hist(
        df["percent_female"], bins=int(180 / binwidth), color="blue", edgecolor="black"
    )

    # Title and labels
    ax.set_title("Histogram with Binwidth = %d" % binwidth, size=15)
    ax.set_xlabel("Percent female", size=10)
    ax.set_ylabel("Count households", size=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# Children Y/N

# %%
df.insert(5, "children", df.iloc[:, [5, 6, 7, 15, 16, 17]].sum(axis=1))
df.insert(6, "children_under_5", df.iloc[:, [6, 16]].sum(axis=1))

# %%
df.children = np.where(df.children > 0, 1, 0)
df.children_under_5 = np.where(df.children_under_5 > 0, 1, 0)

# %%
# Large proportion have children
df.children.value_counts().plot(kind="bar")

# %%
df.children_under_5.value_counts().plot(kind="bar")

# %% [markdown]
# Single adult hh

# %%
df.head(1)

# %%
df.insert(
    7,
    "adults",
    df.iloc[:, [10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26]].sum(axis=1),
)
df.adults = df.adults + 1  # To include the respondent

# %%
df.insert(8, "single_parent_hh", np.where(df.adults == 1, 1, 0))

# %%
df.single_parent_hh.value_counts()

# %%
# Drop as too small

# %%
df.drop(["single_parent_hh"], axis=1, inplace=True)

# %% [markdown]
# Ratio income generating to hh size

# %%
df.insert(
    8, "income_gen_ratio", ((df.income_generating_members / df.household_size) * 100)
)

# %%
# Show 4 different binwidths
for i, binwidth in enumerate([1, 5, 10, 15]):

    # Set up the plot
    ax = plt.subplot(2, 2, i + 1)

    # Draw the plot
    ax.hist(
        df["income_gen_ratio"],
        bins=int(180 / binwidth),
        color="blue",
        edgecolor="black",
    )

    # Title and labels
    ax.set_title("Histogram with Binwidth = %d" % binwidth, size=15)
    ax.set_xlabel("Percentage income generating hh members", size=10)
    ax.set_ylabel("Count households", size=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# Ratio income generating adults

# %%
df.insert(9, "income_gen_adults", ((df.income_generating_members / df.adults) * 100))

# %%
# Show 4 different binwidths
for i, binwidth in enumerate([1, 5, 10, 15]):

    # Set up the plot
    ax = plt.subplot(2, 2, i + 1)

    # Draw the plot
    ax.hist(
        df["income_gen_adults"],
        bins=int(180 / binwidth),
        color="blue",
        edgecolor="black",
    )

    # Title and labels
    ax.set_title("Histogram with Binwidth = %d" % binwidth, size=15)
    ax.set_xlabel("Percentage income generating adults", size=10)
    ax.set_ylabel("Count households", size=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Health difficulties

# %%
df.insert(10, "health_difficulty", df.iloc[:, [31, 32, 33, 34, 35, 36]].sum(axis=1))

# %%
df.head(1)

# %%
# 53 = outlier / error?
df.health_difficulty.plot()

# %%
# As counts get lower and lower will recode to Y/N health difficulty
df.health_difficulty.value_counts()

# %%
df.health_difficulty = np.where(df.health_difficulty > 0, 1, 0)

# %%
# Could consider adjusting this later? Percent of hh then reduce the feature (eg over ...% Y/N)

# %%
df.health_difficulty.value_counts()

# %% [markdown]
# ## Modelling

# %%
df_model = dm.nfri_preferences_to_numbers(df)

# %%
# Recode to binary on select features
df_model["respondent_female"] = np.where(df_model.gender_respondent == "Female", 1, 0)
df_model.previous_nfri = np.where(df_model.previous_nfri == "Yes", 1, 0)

# %%
dummy_features = ["age_respondent", "hh_ethnicity", "house_material"]

# %%
df_dummys = pd.get_dummies(df_model[dummy_features], drop_first=True)

# %%
df_model_select_features = [
    "respondent_female",
    "household_size",
    "percent_female",
    "children",
    "children_under_5",
    "income_gen_ratio",
    "income_gen_adults",
    "health_difficulty",
    "previous_nfri",
]

# %%
df_features = pd.merge(
    df_model[df_model_select_features], df_dummys, left_index=True, right_index=True
)

# %%
df_features.head(2)

# %%
X_train, y_train, X_validation, y_validation, X_test, y_test = mm.train_test_validation(
    df_features, df_model[non_basic], 0.8
)

# %%

# %%
scaler = MinMaxScaler()
numerical_df = df.select_dtypes(include="number")
df[numerical_df.columns] = scaler.fit_transform(numerical_df)

# %%
feature_scaling(X_train)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
df_variables = df_clean.iloc[:, [2, 3, 4, 5, 6, 37]]
variables_corr = df_variables.corr()

# %%

# %%
df_preferences_basic = (
    df_clean.loc[:, basic].apply(pd.Series.value_counts, normalize=True)
).T

# %%
df_preferences_basic.sort_values(by=[3], inplace=True, ascending=False)
df_preferences_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
df_preferences_basic
