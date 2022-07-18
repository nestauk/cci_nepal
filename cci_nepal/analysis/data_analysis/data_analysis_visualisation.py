# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# %% [markdown]
# #### Importing libraries #####

# %%
import logging
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# %matplotlib inline
# plt.rcParams["figure.figsize"] = [12, 8]


import seaborn as sns

import cci_nepal
from cci_nepal.getters import get_data as grd
from cci_nepal.pipeline import data_manipulation as dm
from cci_nepal.pipeline import model_tuning_report as mtr
from cci_nepal import config

# %%
project_dir = cci_nepal.PROJECT_DIR


# %% [markdown]
# The following analysis is of the NFRI Survey train dataset collected across two districts (Sindhupalchowk and Mahottari) of Nepal. The dataset consists of 2619 rows (observations) and 73 columns (variables) in total.
#
# The variables consist of demographic and geographic information at household level and their preference of NFRI items.
#
# Keeping in mind our project goal and further steps (like Modeling), the variables can be further divided into input and output variables. The demographic, geographic and other related variables of households can be treated as input variables and their NFRI preference as output variables.
#
# The analysis will be divided into four parts:
#
# - First Part: Data Pre-Processing
# - Second Part: Analysis of input variables / features.
# - Third Part: Analysis of output variables (i.e NFRI Preference)
# - Fourth Part: Analysis of output variables (NFRI Preferences) across different input variables.
#

# %%
df = grd.read_train_data()
column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")
selected_features = grd.get_lists(f"{project_dir}/cci_nepal/config/select_features.csv")

# %%
df.shape

# %% [markdown]
# ### First Part of Analysis: Data Preprocessing ####

# %% [markdown]
# - Removal of non-feature columns (like notes, introduction, etc.)
# - Renaming of columns to make them more interpretable.
# - Replacement of null values with 0 (as answers with 0 are left as null in our survey.)
# - Addition of new features (like total male, total female, total children, etc.)

# %%
columns_to_drop = [5, 16, 29, 39, 41, 42, 54, 55, 67, 68, 69, 70, 71]
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
df.columns = column_names
df.fillna(0, inplace=True)
df = df.applymap(lambda s: s.lower() if type(s) == str else s)

df.insert(5, "total_male", df.iloc[:, 5:15].sum(axis=1))
df.insert(6, "total_female", df.iloc[:, 16:26].sum(axis=1))
df.insert(7, "total_children", df.iloc[:, [7, 8, 9, 17, 18, 19]].sum(axis=1))
df.insert(8, "children_under_5", df.iloc[:, [8, 18]].sum(axis=1))
df.insert(9, "health_difficulty", df.iloc[:, 31:37].sum(axis=1))
df["household_size"] = df["total_male"] + df["total_female"]
df["percent_non_male"] = ((df.household_size - df.total_male) / df.household_size) * 100
df["income_gen_ratio"] = (df.income_generating_members / df.household_size) * 100
df["sindupalchowk"] = np.where(df.district == "sindupalchok", 1, 0)
df.previous_nfri = np.where(df.previous_nfri == "yes", 1, 0)
df.income_gen_ratio = df.income_gen_ratio.replace(np.inf, np.nan)

# %% [markdown]
# #### Dividing NFRI items into Basic and Non Basic items ####

# %%
nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]

# %%
df.head()

# %% [markdown]
# #### First thing first, let's set the colors and the visual settings right! ####

# %%
# sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc("axes", titlesize=20)  # fontsize of the axes title
plt.rc("axes", labelsize=16)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=16)  # fontsize of the tick labels
plt.rc("legend", fontsize=16)  # legend fontsize
# plt.rc('font', size=14)          # controls default text sizes

# Wanted palette details
enmax_palette = [
    "#0000FF",
    "#FF6E47",
    "#18A48C",
    "#EB003B",
    "#9A1BB3",
    "#FDB633",
    "#97D9E3",
]
color_codes_wanted = [
    "nesta_blue",
    "nesta_orange",
    "nesta_green",
    "nesta_red",
    "nesta_purple",
    "nesta_yellow",
    "nesta_agua",
]
c = lambda x: enmax_palette[color_codes_wanted.index(x)]
nesta_color = c("nesta_blue")
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)
sns.set_palette(sns.color_palette(enmax_palette))

# %% [markdown]
# ### Second Part: Analysis of Input Variables ###

# %% [markdown]
# #### Respondent Gender ####

# %%
ax = sns.barplot(
    x=df.respondent_gender.value_counts(normalize=True).index,
    y=df.respondent_gender.value_counts(normalize=True),
    color=nesta_color,
)

ax.set_title("Respondent Gender Percentage", pad=12)
ax.set_xlabel("Gender")
ax.set_ylabel("Percentage")
plt.xticks(rotation=90)
ax.set(xticklabels=["Male", "Female", "Other"])

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/respondent_gender.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/respondent_gender.svg",
    bbox_inches="tight",
)


# %% [markdown]
# As we see, the percentage of Respondent Gender is almost balanced, which was our plan before the survey also.

# %% [markdown]
# #### Respondent Age Breakdown ####

# %%
df["respondent_age"].value_counts(normalize=True)

# %%
plt.figure(figsize=(8, 6))

ax = sns.barplot(
    x=df.respondent_age.value_counts(normalize=True).index,
    y=df.respondent_age.value_counts(normalize=True),
    color=nesta_color,
)

ax.set_title("Respondent Age Percentage", pad=12)
ax.set_xlabel("Age")
ax.set_ylabel("Percentage")
plt.xticks(rotation=90)
ax.set(xticklabels=["30-39", "40-49", "18-29", "50-59", "60-69", "70-79", "80-above"])

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/respondent_age.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/respondent_age.svg",
    bbox_inches="tight",
)


# %% [markdown]
# #### Household Size, Total Male, Total Female

# %%
ax = sns.histplot(data=df, x="household_size", kde=True, binwidth=1, color=nesta_color)
ax.set_title("Household Size Distribution", pad=10)
ax.set_xlabel("Household Size")
ax.set_ylabel("Count")

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/household_size.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/household_size.svg",
    bbox_inches="tight",
)


# %%
ax = sns.histplot(data=df, x="total_male", kde=True, binwidth=1, color=nesta_color)
ax.set_title("Total Male Distribution", pad=10)
ax.set_xlabel("Male Number")
ax.set_ylabel("Count")

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/total_male.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/total_male.svg",
    bbox_inches="tight",
)


# %%
ax = sns.histplot(data=df, x="total_female", kde=True, binwidth=1, color=nesta_color)
ax.set_title("Total Female Count", fontdict={"fontsize": 20}, pad=10)
ax.set_xlabel("Female Number", fontsize=20)
ax.set_ylabel("Count", fontsize=20)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/total_female.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/total_female.svg",
    bbox_inches="tight",
)


# %% [markdown]
# #### Children #####

# %% [markdown]
# #### Total Children, Children Upto 5 ####

# %%
ax = sns.histplot(data=df, x="total_children", kde=True, binwidth=1, color=nesta_color)
ax.set_title("Total Children Distribution", pad=10)
ax.set_xlabel("Children Number")
ax.set_ylabel("Count")

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/total_children.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/total_children.svg",
    bbox_inches="tight",
)


# %%
ax = sns.histplot(
    data=df, x="children_under_5", kde=True, binwidth=1, color=nesta_color
)
ax.set_title("Total Children Under 5 Distribution", pad=10)
ax.set_xlabel("Children Number")
ax.set_ylabel("Count")

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/total_children_under_5.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/total_children_under_5.svg",
    bbox_inches="tight",
)


# %%
plt.figure(figsize=(8, 6))


ax = sns.barplot(
    x=df.children_under_5.value_counts(normalize=True).index,
    y=df.children_under_5.value_counts(normalize=True),
    color=nesta_color,
)
ax.set_title("Children Under 5 Percentage Representation", pad=12)
ax.set_xlabel("Children Under 5 Number")
ax.set_ylabel("Percentage")
ax.set(xticklabels=["0", "1", "2", "3", "4", "5", "6"])

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/children_under_5_percentage.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/children_under_5_percentage.svg",
    bbox_inches="tight",
)


# %% [markdown]
# #### Ethnicity ####

# %%
df["ethnicity"].value_counts(normalize=True)

# %%
plt.figure(figsize=(8, 6))


plt.figure(figsize=(8, 6))
ax = sns.barplot(
    x=df.ethnicity.value_counts(normalize=True).index,
    y=df.ethnicity.value_counts(normalize=True),
    color=nesta_color,
)
ax.set_title("Ethnicity Percentage", pad=12)
ax.set_xlabel("Ethnicity")
ax.set_ylabel("Percentage")
plt.xticks(rotation=90)

ax.set(
    xticklabels=[
        "Adibasi/Janjati/Newar",
        "Brahmin/Chhetri/Sanyashi/Thakuri",
        "Dalit",
        "Madhesi",
        "Other",
        "Prefer not to answer",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/ethnicity.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/ethnicity.svg",
    bbox_inches="tight",
)


# %% [markdown]
# - As we see, more than 95 percents of observations come from four majour ethnicities in the option.
#
# - All four major ethnicities almost evenly distributed, with this being highest at 28 and this being lowest at 19.

# %%
ax = sns.catplot(
    x="ethnicity", kind="count", hue="district", height=5.5, aspect=3, data=df
)
plt.xlabel("Ethnicity")
plt.ylabel("Count")
plt.xticks(rotation=90)
ax.set(
    xticklabels=[
        "Dalit",
        "Brahmin/Chhetri/Sanyashi/Thakuri",
        "Madhesi",
        "Adibasi/Janjati/Newar",
        "Prefer not to answer",
        "Other",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/ethnicity_district.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/ethnicity_district.svg",
    bbox_inches="tight",
)


# ax.set(xticklabels=["Adibasi/Janjati/Newar", "Brahmin/Chhetri/Sanyashi/Thakuri", "Dalit", "Madhesi", "Other", "Prefer not to answer"]);

# %% [markdown]
# But when we look at the district wise distribution,....

# %% [markdown]
# #### House Material ####

# %%
df["house_material"].value_counts(normalize=True)

# %%
plt.figure(figsize=(8, 6))

ax = sns.barplot(
    x=df.house_material.value_counts(normalize=True).index,
    y=df.house_material.value_counts(normalize=True),
    color=nesta_color,
)
ax.set_title("House Material Percentage Representation")
ax.set_xlabel("House Material")
ax.set_ylabel("Percentage")
plt.xticks(rotation=90)

ax.set(
    xticklabels=[
        "Mud bonded bricks stone",
        "Rcc with pillar",
        "Other",
        "Cement bonded bricks stone",
        "Wooden pillar",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/house_material.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/house_material.svg",
    bbox_inches="tight",
)


# %% [markdown]
# As we can see, the "other" category is significantly represented in the dataset (almost 20 percent of total observations. So, we will delve deeper to see what answers constitute the "other" category.

# %%
df[df["house_material"] == "other"].material_other.value_counts(normalize=True)[0:5]

# %% [markdown]
# As we can see, almost 2/3rd (65 percent) of the other categories have "mato ghar" as the answer, meaning clay house, followed by clay again at 6 percent. The other answers are also mostly clay related terms. Thus, the "other" category mostly refers to the "Clay" as house materials.

# %%
ax = sns.catplot(
    x="house_material", kind="count", hue="district", height=5.5, aspect=3, data=df
)
plt.xlabel("House Material")
plt.ylabel("Count")
ax.set(
    xticklabels=[
        "Mud bonded bricks stone",
        "Other",
        "Cement bonded bricks stone",
        "Rcc with pillar",
        "Wooden pillar",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/house_material_district.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/house_material_district.svg",
    bbox_inches="tight",
)


# ax.set(xticklabels=["Other", "Rcc with pillar", "Cement bonded bricks stone", "Wooden pillar", "Mud bonded bricks stone"]);


# %% [markdown]
# #### Total Income Generating Members ####

# %%
plt.figure(figsize=(8, 6))

ax = sns.barplot(
    x=df.income_generating_members.value_counts(normalize=True).index,
    y=df.income_generating_members.value_counts(normalize=True),
    color=nesta_color,
    order=[1, 2, 3, 0, 4, 5, 6, 7],
)
ax.set_title("Income Generating Members Percentage Representation")
ax.set_xlabel("Number of Members")
ax.set_ylabel("Percentage")
plt.xticks(rotation=90)

ax.set(xticklabels=["1", "2", "3", "0", "4", "5", "6", "7"])

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/income_generating_members.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/income_generating_members.svg",
    bbox_inches="tight",
)


# %% [markdown]
# As we see, almost 60 percent of the households have just 1 income generating member, with 26 percent having two income generating members.

# %% [markdown]
# #### Previous NFRI History ####

# %%
plt.figure(figsize=(8, 6))

ax = sns.barplot(
    x=df.previous_nfri.value_counts(normalize=True).index,
    y=df.previous_nfri.value_counts(normalize=True),
    color=nesta_color,
    order=[1, 0],
)
ax.set_title("Previous NFRI History Percentage Representation")
ax.set_xlabel("Label")
ax.set_ylabel("Percentage")
plt.xticks(rotation=90)

ax.set(xticklabels=["Yes", "No"])

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/previous_nfri.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/previous_nfri.svg",
    bbox_inches="tight",
)


# %% [markdown]
# #### Correlation Matrix of all Input Features ####

# %%
df_all_input_features = df.loc[
    :,
    [
        "house_material",
        "previous_nfri",
        "household_size",
        "percent_non_male",
        "children_under_5",
        "income_gen_ratio",
        "health_difficulty",
        "sindupalchowk",
    ],
]
df_all_input_features = pd.get_dummies(df_all_input_features)

plt.figure(figsize=(15, 6))
ax = sns.heatmap(
    df_all_input_features.corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG"
)  # cmap="BrBG" # pal = nesta pallet
ax.set_title("Correlation Heatmap of all input features", pad=12)

ax.set(
    xticklabels=[
        "Previous NFRI",
        "Household Size",
        "Percent Non Male",
        "Children Under 5",
        "Income Generating Ratio",
        "Health Difficulty",
        "Sindupalchowk",
        "House Material - Cement Bonded Bricks Stone",
        "House Material - Mud Bonded Bricks Stone",
        "House Material - Other",
        "House Material - RCC with Pillar",
        "House Material - Wooden Pillar",
    ]
)

ax.set(
    yticklabels=[
        "Previous NFRI",
        "Household Size",
        "Percent Non Male",
        "Children Under 5",
        "Income Generating Ratio",
        "Health Difficulty",
        "Sindupalchowk",
        "House Material - Cement Bonded Bricks Stone",
        "House Material - Mud Bonded Bricks Stone",
        "House Material - Other",
        "House Material - RCC with Pillar",
        "House Material - Wooden Pillar",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/correlation_matrix_all_input_features.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/correlation_matrix_all_input_features.svg",
    bbox_inches="tight",
)


# %% [markdown]
# #### Correlation Matrix of Numerical Input Features ####

# %%
plt.figure(figsize=(16, 6))
ax = sns.heatmap(
    df.loc[:, selected_features[1:]].corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG"
)  # cmap="BrBG" # pal = nesta pallet
ax.set_title(
    "Correlation Heatmap of Numerical Features", fontdict={"fontsize": 20}, pad=12
)
ax.set(
    xticklabels=[
        "Household Size",
        "Percent Non Male",
        "Children Under 5",
        "Income Generating Ratio",
        "Health Difficulty",
        "Sindupalchowk",
    ]
)
ax.set(
    yticklabels=[
        "Household Size",
        "Percent Non Male",
        "Children Under 5",
        "Income Generating Ratio",
        "Health Difficulty",
        "Sindupalchowk",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/correlation_matrix_numerical_input_features.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/correlation_matrix_numerical_input_features.svg",
    bbox_inches="tight",
)


# %% [markdown]
# ### Third Part of Analysis: NFRI Preferences ###

# %% [markdown]
# #### Class Proportion Shelter kit ####

# %%
df_preference_labels_basic = (
    df.loc[:, basic].apply(pd.Series.value_counts, normalize=True).T
)
df_preference_labels_basic = df_preference_labels_basic.reindex(
    ["essential", "desirable", "unnecessary"], axis=1
).sort_values(by="essential", ascending=False)


ax = sns.heatmap(df_preference_labels_basic, annot=True)
ax.set_title("Class Proportion Per Shelter Item", pad=12)
ax.set(xticklabels=["Essential", "Desirable", "Unnecessary"])
ax.set(
    yticklabels=[
        "Water Bucket",
        "Utensil Set",
        "Blanket",
        "Nylon Rope",
        "Plastic Tarpaulin",
        "Sack Packing Bag",
        "Sari",
        "Terry Cloth",
        "Shouting Cloth Jeans",
        "Printed Cloth",
        "Male Dhoti",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/class_proportion_shelter_items.png"
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/class_proportion_shelter_items.svg"
)


# %% [markdown]
# 9 out of 11 being deemed esssential by more than 50 percent, 3 as essential by more than 90 percent, and Male Dhoti interesting being called essential by almost 48 percent and unnecessary by 35 percent!

# %% [markdown]
# #### Class Proportion Wash-Dignity kit ####

# %%
df_preference_labels_non_basic = (
    df.loc[:, non_basic].apply(pd.Series.value_counts, normalize=True).T
)
df_preference_labels_non_basic = df_preference_labels_non_basic.reindex(
    ["essential", "desirable", "unnecessary"], axis=1
).sort_values(by="essential", ascending=False)
ax = sns.heatmap(df_preference_labels_non_basic, annot=True)
ax.set_title("Class Proportion Per Wash-Dignity Item", pad=12)
ax.set(xticklabels=["Essential", "Desirable", "Unnecessary"])
ax.set(
    yticklabels=[
        "Laundry Soap",
        "Bathing Soap",
        "Tooth Brush and Paste",
        "Cotton Towel",
        "Sanitary Pad",
        "Torch Light",
        "Ladies Underwear",
        "Hand Sanitizer",
        "Nail Cutter",
        "Liquid Chlorine",
        "Whistle Blow",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/class_proportion_wash_items.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/class_proportion_wash_items.svg",
    bbox_inches="tight",
)


# %% [markdown]
# - 10 out of 11 items being deemed essential by more than 50 percent, 8 out of 11 items deemed essential by more than 80 percent and 3 items above 90 percent.
#
# - Whistle Blow contrasting deemed essential by only 34 percent.

# %% [markdown]
# #### Output Preferences Numerized ####

# %%
df_numeric = dm.nfri_preferences_to_numbers(df)

# %%
shelter_scores = df_numeric.loc[:, basic].agg(["mean", np.std])
shelter_scores_sorted = shelter_scores.T
shelter_scores_sorted.sort_values(by=["mean"], inplace=True)
shelter_scores_sorted

# %%
ax = sns.heatmap(
    shelter_scores_sorted, vmin=0, vmax=3, annot=True
)  # cmap = pal for nesta pallet
ax.set_title("Heatmap of Mean and Variance of Preference Score of Shelter kit", pad=12)
ax.set(xticklabels=["Mean Score", "Variance Score"])
ax.set(
    yticklabels=[
        "Male Dhoti",
        "Printed Cloth",
        "Sari",
        "Shouting Cloth Jeans",
        "Terry Cloth",
        "Sack Packing Bag",
        "Plastic Tarpaulin",
        "Nylon Rope",
        "Blanket",
        "Utensil Set",
        "Water Bucket",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/preference_score_shelter_items.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/preference_score_shelter_items.svg",
    bbox_inches="tight",
)


# %%
wash_scores = df_numeric.loc[:, non_basic].agg(["mean", np.std])
wash_scores
wash_scores_sorted = wash_scores.T
wash_scores_sorted.sort_values(by=["mean"], inplace=True)
wash_scores_sorted

# %%
ax = sns.heatmap(wash_scores_sorted, vmin=0, vmax=3, annot=True)  # cmap="BrBG"
ax.set_title("Heatmap of Mean and Variance of Preference Score of Wash kit", pad=12)
ax.set(xticklabels=["Mean Score", "Variance Score"])
ax.set(
    yticklabels=[
        "Whistle Blow",
        "Liquid Chlorine",
        "Nail Cutter",
        "Ladies Underwear",
        "Hand Sanitizer",
        "Sanitary Pad",
        "Torch Light",
        "Cotton Towel",
        "Tooth Brush and Paste",
        "Bathing Soap",
        "Laundry Soap",
    ]
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/preference_score_wash_items.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/preference_score_wash_items.svg",
    bbox_inches="tight",
)


# %% [markdown]
# #### Correlation of Shelter kit items ####

# %%
plt.figure(figsize=(16, 6))
ax = sns.heatmap(
    df_numeric.loc[:, basic].corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG"
)  # cmap="BrBG"
ax.set_title("Correlation Heatmap for Preference Scores of Shelter Items", pad=12)
ax.set(
    xticklabels=[
        "Plastic Tarpaulin",
        "Blanket",
        "Sari",
        "Male Dhoti",
        "Shouting Cloth Jeans",
        "Printed Cloth",
        "Terry Cloth",
        "Utensil Set",
        "Water Bucket",
        "Nylon Rope",
        "Sack Packing Bag",
    ]
)
ax.set(
    yticklabels=[
        "Plastic Tarpaulin",
        "Blanket",
        "Sari",
        "Male Dhoti",
        "Shouting Cloth Jeans",
        "Printed Cloth",
        "Terry Cloth",
        "Utensil Set",
        "Water Bucket",
        "Nylon Rope",
        "Sack Packing Bag",
    ]
)
# plt.savefig('Correlation Heatmap for Shelter Kit.svg')

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/correlation_matrix_shelter_items.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/correlation_matrix_shelter_items.svg",
    bbox_inches="tight",
)


# %% [markdown]
# All clothing related items are highly correlated in terms of preferences, maybe exhibiting same grouping in terms of respondents' perception towards them.
#
# Interetingly, sacking bag and nylon rope also have relatively high correlation. Same for Water Bucket and Utensil Set.

# %% [markdown]
# #### Correlation of Wash-Dignity items ####

# %%
plt.figure(figsize=(16, 6))
ax = sns.heatmap(
    df_numeric.loc[:, non_basic].corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG"
)  # cmap = "Blues
ax.set_title("Correlation Heatmap for Preference Scores of Wash Items", pad=12)

ax.set(
    xticklabels=[
        "Cotton Towel",
        "Bathing Soap",
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
)
ax.set(
    yticklabels=[
        "Cotton Towel",
        "Bathing Soap",
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
)

plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/correlation_matrix_wash_items.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/correlation_matrix_wash_items.svg",
    bbox_inches="tight",
)


# %% [markdown]
# Here also, we can see the high correlation score between items that exhibt similar grouping. Like Laundry Soap and Bathing Soap being very highly correlated, and Tooth Brush and Paste also being highly correlated to both.
#
# Similarly, Ladies Underwear and Sanitary Pad are also highly correlated. Same can be observed for Hand Sanitizer and Liquid Chlorine.

# %% [markdown]
# ### Fourth Part of Analysis: Analysing NFRI Preferences across different Input Variables ####

# %% [markdown]
# #### NFRI Preferences District Wise ####

df_numeric.groupby("district")[basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]

df_numeric.groupby("district")[non_basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]

# %% [markdown]
# As we can see, the preference scores are fairly same for both the districts for wash items. Whereas for shelter items, the clothing items have noticeably low scores for Sindupalchowk compared to Mahottari.

# %% [markdown]
# #### NFRI Preferences Respondent Gender Wise ####

# %%
df_numeric.groupby("respondent_gender")[basic].apply(
    lambda x: x.astype(int).mean()
).iloc[
    0:4,
]

# %% [markdown]
# As we can see, for both basic and non basic, the NFRI preference scores are almost identical for all the items. That way, the preference scores of items are similar regardless of respondent gender.

# %%
df_numeric.groupby("respondent_gender")[non_basic].apply(
    lambda x: x.astype(int).mean()
).iloc[
    0:4,
]

# %% [markdown]
# #### NFRI Preferences Ethnicity Wise ####

# %% [markdown]
# Comparing across the four major ethnicities only as they make up more than 95 percent
#

# %%
df_numeric.groupby("ethnicity")[basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]

# %%
df_numeric.groupby("ethnicity")[non_basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]

# %% [markdown]
# As we can see, most of the items have similar scores across all ethnicities. And for the ones that are different, the scores are consistently higher for Dalit and Madhesi ethnicity.
#
# (Dalit and Madhesi ethnicity are also the ones that are mostly present in Mahottari district in our dataset.)

# %% [markdown]
# #### NFRI Preference House Material Wise ####

# %%
df_numeric.groupby("house_material")[basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]

# %%
df_numeric.groupby("house_material")[non_basic].apply(
    lambda x: x.astype(int).mean()
).iloc[
    0:4,
]

# %% [markdown]
# As we see, the category "other" has signicantly higher score for both basis and non-basic, specially for items that have lower scores for other categories. (Like for Sari, Male Dhoti, Clothing items, Whistle Blow.)
#
# 70 percent of category "other" comprise of house material Clay, and most of these households are present in Mahottari district (as highlighted above in the section of House Material.)

# %% [markdown]
# So the feature District, Ethnicity and House Material are related in our dataset, with Madhesi and Dalit ethnicity and House Material "other" presently mostly in one district Mahottari only.

# %% [markdown]
# #### NFRI Preferences previous NFRI history wise ####

# %%
df_numeric.groupby("previous_nfri")[basic].apply(lambda x: x.astype(int).mean())

# %%
df_numeric.groupby("previous_nfri")[non_basic].apply(lambda x: x.astype(int).mean())

# %% [markdown]
# As we can see, for most of the items in basic, the preference score is higher for the No category (households that haven't received NFRI in past, coded as 0 in our script.) As for non basic, the scores are fairly similar except for Whistle Blow, which also also higher score for the No category.

# %% [markdown]
# #### Creating a map of Nepal  ####

# %% [markdown]
# This is an entirely option section where we visualise the points of survey responses.
# Running the script below requires the library geopandas which is not part of the requirements.txt file.
#


import geopandas as gpd
from shapely.geometry import *
from geopandas import GeoDataFrame

# Read Nepal map file
nepal_shapefile = gpd.read_file("New_Local_Level_Map.shp")

geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
shapefile_geo = gpd.GeoDataFrame(df, geometry=geometry)

fig, ax = plt.subplots(figsize=(15, 15))
nepal_shapefile.plot(ax=ax, alpha=0.9, color=c("nesta_green"))
shapefile_geo[shapefile_geo["district"] == "Sindupalchok"].plot(
    ax=ax, markersize=5, marker="o", color=c("nesta_blue"), label="Sindhupalchowk"
)
shapefile_geo[shapefile_geo["district"] == "Mahottari"].plot(
    ax=ax, markersize=5, marker="o", color=c("nesta_red"), label="Mahottari"
)

plt.title("Location Points of Survey Responses")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(prop={"size": 16})
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/png/nepal_map.png",
    bbox_inches="tight",
)
plt.savefig(
    f"{project_dir}/outputs/figures/data_analysis/svg/nepal_map.svg",
    bbox_inches="tight",
)
