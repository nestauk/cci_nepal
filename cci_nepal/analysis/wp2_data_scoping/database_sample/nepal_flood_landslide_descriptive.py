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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cci_nepal
import logging
import re

# %matplotlib inline


# %%
project_dir = cci_nepal.PROJECT_DIR
logging.info(project_dir)

# %%
# to recognize the used characterset
# #!pip install openpyxl

# %%
# read sample nepal flood data for 2020
data_df = pd.read_excel(f"{project_dir}/inputs/data/PDM_ Datasheet.xlsx")

# %%
# data on nepal's population by district in 2011 -could be explored to gain insight about the population distribution
# of the affected areas.
population_df = pd.read_csv(f"{project_dir}/inputs/data/data.csv")

# %%
# extract population only
population = population_df[population_df["Category "] == "Population"]

# %%
# removing nepali from responses
data_df = data_df.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)


# %%
def clean_df_columns(df):
    df.columns = df.columns.str.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)
    df.columns = df.columns.str.lstrip()
    logging.info(df.columns)
    # return df


# %%
clean_df_columns(data_df)


# %%
# renaming age group columns by removing  html tags. We use a combination of columns in the dataframe to establish
# meaningful column names for the age brackets.
data_df.rename(
    columns={
        "span styledisplaynonerow-  malespan": "male0_5",
        "span styledisplaynonerow- femalespan": "female0_5",
        "span styledisplaynonerow1-  malespan": "male6_17",
        "span styledisplaynonerow1- femalespan": "female6_17",
        "span styledisplaynonerow2-  malespan": "male18_59",
        "span styledisplaynonerow2- femalespan": "female18_59",
        "span styledisplaynonerow3-  malespan": "male60+",
        "span styledisplaynonerow3- femalespan": "female60+",
    },
    inplace=True,
)

# %%
# compute missing values per column in the dataframe.
percent_missing = data_df.isnull().sum() * 100 / len(data_df)
missing_value_df = pd.DataFrame(
    {"column_name": data_df.columns, "percent_missing": percent_missing}
)


# %%
# visualizing the composition of missing values in the data
missing_value_df.sort_values("percent_missing", inplace=True)
plt.figure(figsize=(15, 10))
plt.ylabel("missing value %", fontsize=16)
plt.xlabel("Fields", fontsize=16)
plt.title("Percentage of missing values by columns", fontsize=20)
missing_value_df["percent_missing"].plot(kind="bar")
plt.xticks([])

# %%
# missing values greater than 82%
plt.figure(figsize=(15, 10))
missing_value_df["percent_missing"].tail(45).plot(kind="bar")
plt.title("columns with missing values > 82%", fontsize=18)
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/missing_values_greater80.png"
)
plt.show()

# %%
# attributes with few missing values(<15%).
plt.figure(figsize=(15, 10))
plt.tick_params(axis="both", labelsize=16)
missing_value_df["percent_missing"].head(25)[6:].plot(kind="barh")
plt.title("columns with missing values <15%", fontsize=20)
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/missing_values_less15.png"
)
plt.show()

# %%
plt.figure(figsize=(15, 10))
missing_value_df["percent_missing"].plot(kind="hist")
plt.title("Distribution of columns with missing values", fontsize=20)
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/missing_values_greater80b.png"
)
plt.show()

# %%
# remove columns which all have null values
data_df.dropna(axis=1, how="all", inplace=True)

# %%
# number of columns that were completely null
146 - data_df.shape[1]

# %%
# replace the null values for the age groups for onward processing. The assumption here is that missing values
# imply situations of non-applicability. For example, the absence of male0_5 means no children of this age in houshold
# columns[22:30] of current dataframe hold data on the composition of households by age
data_df[data_df.columns[22:30]] = data_df[data_df.columns[22:30]].fillna(0, axis=1)

# %%
# male/female populations by age groups for the different households.
beneficiaries_by_age_group = {}
for col in data_df.columns[22:30]:
    beneficiaries_by_age_group[col] = sum(data_df[col])

# %%
# stacked bar chart representing distribution of household members by age groups.
labels = ["Age0_5", "Age6_17", "Age18_59", "Age60_above"]
men_means = [87.0, 232, 541, 63]
women_means = [72, 226, 550, 70]
width = 0.35  # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax.bar(labels, men_means, width, label="Male")
ax.bar(labels, women_means, width, bottom=men_means, label="Female")
ax.set_ylabel("Number of beneficiaries")
ax.set_title("Beneficiaries by age group and gender")
ax.legend()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/beneficiaries_by_age_groups.png"
)
plt.show()

# %%
# % of household members age 60 and above
age_60 = (
    beneficiaries_by_age_group["female60+"] + beneficiaries_by_age_group["male60+"]
) / sum(beneficiaries_by_age_group.values())
logging.info(age_60)

# %%
# % of beneficiaries age under 5
age_under5 = (
    beneficiaries_by_age_group["female0_5"] + beneficiaries_by_age_group["male0_5"]
) / sum(beneficiaries_by_age_group.values())
logging.info(age_under5)

# %%
# visualizing how informants were notified about NRCS package distribution
data_df[
    "How were you notified about the relief delivery date"
].value_counts().sort_values().plot(kind="barh")
plt.title("How were you notified about the relief delivery date")
plt.xlabel("Frequency")
plt.savefig(f"{project_dir}/outputs/figures/nepal_descriptive/notification_means.png")


# %%
# free text attribute found in the dataset
len(
    data_df[
        "Are there any suggestion you want to give to Nepal Red Cross society to improve the relief distribution program in the future Mention if any"
    ].unique()
)

# %% [markdown]
# # Some thoughts arising from the data
# 1. Which communication means adopted by nepal RC proved more effective for the affected communities?
# 2. Of the financial aid provided, how can one prioritize the needs of the communities based on how the aid was used?
#
# 3. based on the time taken to arrive at the relief distribution centers, how convenient is it to reduce the number of people who travelled between 3-4 hours to get the aid? Can new centers that would reduce this travel time be created?
#
# 4. For the communication modes used, were resources equitably distributed to the different channels?
#

# %%
# replace the 9 null values with the mode for the specific column on mode of communication.
data_df["How were you notified about the relief delivery date"] = data_df[
    "How were you notified about the relief delivery date"
].fillna(
    "        From community representatives TeachersCommunity leaders and peoples representatives"
)
communication_modes = data_df[
    "How were you notified about the relief delivery date"
].unique()

# %%
# check for duplicate records in the data - none found
data_df.duplicated().sum()

# %%
# skewness of data
data_df.skew(axis=0, skipna=True)

# %%
right_skew = (42 / 50) * 100
right_skew

# %%
left_skew = (4 / 50) * 100
left_skew

# %%
data_df["Are you satisfied with the relief distribution process"].isnull().sum()

# %%
# feedback on how people felt about the distribution process.
data_df["Are you satisfied with the relief distribution process"].value_counts()

# %% [markdown]
# ## What accounts for the fact that some districts are more represented than others?

# %%
district_counts = data_df["District name"].value_counts()

# %%
district_counts

# %%
population.District.unique()

# %%
# extract population of distrricts in the the sample data
district_population = {}
for name in data_df["District name"].unique():
    if name in population.District.unique():
        district_population[name] = population.Value[population.District == name]
        print(population.Value[population.District == name])

# %%
district_population

# %%
district_counts[:14]

# %%
# district names and their frequencies in the sample data
names = [
    "Lamjung",
    "Achham",
    "Kailali",
    "Sindhupalchok",
    "Myagdi",
    "Dhading",
    "Dolakha",
    "Sankhuwasabha",
    "Gulmi",
    "Jajarkot",
    "Bajura",
    "Baglung",
    "Darchula",
    "Arghakhanchi",
    "8 others",
]
counts = [50, 37, 32, 28, 28, 26, 25, 18, 16, 15, 11, 9, 7, 7, 14]


# %%
# plotting proportions of informants from the different districts.
fig2, ax2 = plt.subplots()
ax2.pie(counts, labels=names, autopct="%1.1f%%", shadow=True, startangle=90)
ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/district_representation.png"
)
plt.title("Representation of districts")
plt.show()

# %%
data_df[
    "Are there any suggestion you want to give to Nepal Red Cross society to improve the relief distribution program in the future Mention if any"
].unique()


# %%
data_df.columns[22:30]

# %%
# Motives for receiving relief packages
data_df["Why did you choose to receive relief supplies"].value_counts()

# %%
plt.figure(figsize=(15, 10))
plt.tight_layout()
plt.title("Why did you choose to receive relief supplies", fontsize=20)
data_df["Why did you choose to receive relief supplies"].value_counts().plot(
    kind="barh"
)
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/motive_for_receiving_id.png"
)

# %%
data_df["Is the informant the person receiving the relief materials"].value_counts()

# %%
# why receive relief package by gender
plt.figure(figsize=(15, 10))
plt.tight_layout()
plt.title("Why did you choose to receive relief supplies", fontsize=20)
data_df["Why did you choose to receive relief supplies"][
    data_df["Gender of the informant"] == "  male"
].value_counts().plot(kind="barh")
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/motive_for_receiving_id_male.png"
)

# %%
# motive for receiving relief by gender -male
data_df["Why did you choose to receive relief supplies"][
    data_df["Gender of the informant"] == "  male"
].value_counts()

# %%
# motive for receiving relief by gender -female
data_df["Why did you choose to receive relief supplies"][
    data_df["Gender of the informant"] == " female"
].value_counts()

# %%
# use stacked bar to represent the motives by gender. Figures are manually recorded from above to ensure matching classes
labels1 = [
    "vulnerable family",
    "dont know",
    "because of the family crisis",
    "others",
    "Being displaced",
    "because the house was partially damaged",
    "because the house was completely damaged",
]
male_num = [1, 2, 6, 22, 41, 67, 81]
female_num = [3, 1, 3, 5, 16, 33, 42]
width = 0.35  # the width of the bars
fig, ax2 = plt.subplots()
plt.tick_params(axis="both", labelsize=16)
plt.figure(figsize=(15, 15))
ax2.barh(labels1, male_num, width, label="Male")
ax2.barh(labels1, female_num, width, label="Female")
ax2.set_xlabel("Number of beneficiaries")
ax2.set_title("Motive for receiving relief by gender", fontsize=20)
ax2.legend()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/beneficiaries_by_gender.png"
)
plt.show()

# %%
labels1 = [
    "vulnerable family",
    "dont know",
    "because of the family crisis",
    "others",
    "Being displaced",
    "because the house was partially damaged",
    "because the house was completely damaged",
]
male_num = [1, 2, 6, 22, 41, 67, 81]
female_num = [3, 1, 3, 5, 16, 33, 42]
width = 0.35  # the width of the bars: can also be len(x) sequence
relief_df_gender = pd.DataFrame({"male": male_num, "female": female_num}, index=labels1)
plt.figure(figsize=(15, 10))
relief_df_gender.plot.barh(stacked=True)
plt.title("Motive for receiving relief by gender", fontsize=20)
plt.xlabel("Frequency")
plt.legend()

# %%
labels1 = [
    "vulnerable family",
    "dont know",
    "because of the family crisis",
    "others",
    "Being displaced",
    "because the house was partially damaged",
    "because the house was completely damaged",
]
male_num = [1, 2, 6, 22, 41, 67, 81]
female_num = [3, 1, 3, 5, 16, 33, 42]
# the width of the bars: can also be len(x) sequence
relief_df_gender = pd.DataFrame({"male": male_num, "female": female_num}, index=labels1)
relief_df_gender.plot.barh(stacked=True, width=0.75)
plt.title("Motive for receiving relief by gender", fontsize=20)
# plt.figure(figsize=(20,20))
df_total = relief_df_gender["male"] + relief_df_gender["female"]
df_rel = relief_df_gender[relief_df_gender.columns[:]].div(df_total, 0) * 100
for n in df_rel:
    for i, (cs, ab, pc) in enumerate(
        zip(relief_df_gender.iloc[:, :].cumsum(1)[n], relief_df_gender[n], df_rel[n])
    ):
        plt.text(
            cs - ab / 2,
            i,
            str(np.round(pc, 1)) + "%",
            va="center",
            ha="center",
            rotation=20,
            fontsize=10,
        )

# %%
# why receive relief package by gender
plt.figure(figsize=(15, 10))
plt.tight_layout()
plt.title("Why did you choose to receive relief supplies", fontsize=20)
data_df["Why did you choose to receive relief supplies"][
    data_df["Gender of the informant"] == " female"
].value_counts().plot(kind="barh")
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/motive_for_receiving_id_male.png"
)

# %%
data_df.loc[:, "Why did you choose to receive relief supplies":]

# %%
cash_spending_df = data_df.loc[
    :, "For what did you spend the money you received from the Red Cross":
]

# %%
# renaming columns
for col in cash_spending_df.loc[
    :, "For what did you spend the money you received from the Red Cross":
].columns[1:10]:
    cash_spending_df.rename(
        columns={
            col: col.replace(
                "For what did you spend the money you received from the Red Cross", ""
            ).lstrip()
        },
        inplace=True,
    )


# %%
cash_spending_df.loc[
    :, "For what did you spend the money you received from the Red Cross":
].columns[1:11]

# %%
# cash spending by informant
plt.figure(figsize=(15, 10))
plt.xlabel("Frequency", fontsize=16)
plt.ylabel("Spending activity", fontsize=16)
plt.title("Cash spending", fontsize=20)
cash_spending_df[
    cash_spending_df.loc[
        :, "For what did you spend the money you received from the Red Cross":
    ].columns[1:10]
].sum(axis=0).sort_values().plot(kind="barh")

# %%
plt.figure(figsize=(15, 10))
plt.xlabel("Frequency", fontsize=16)
plt.ylabel("Spending activity", fontsize=16)
plt.title("Cash spending", fontsize=20)
cash_spending_df[
    cash_spending_df.loc[
        :, "For what did you spend the money you received from the Red Cross":
    ].columns[1:10]
].sum(axis=0).sort_values().plot(kind="barh")

# %%
# cash_spending_df holds only the data on cash spending. The motive is to allow renaming of fields including others
# without conflict. - others would appear many times in the main df
cash_spending_df = cash_spending_df[
    cash_spending_df.loc[
        :, "For what did you spend the money you received from the Red Cross":
    ].columns[1:10]
]

# %%
cash_spending_df["Gender of the informant"] = data_df["Gender of the informant"]
cash_spending_df["age of the informant"] = data_df["age of the informant"]


# %%
female_spending = cash_spending_df[
    cash_spending_df["Gender of the informant"] == " female"
]
male_spending = cash_spending_df[
    cash_spending_df["Gender of the informant"] == "  male"
]
# cash_spending_df[cash_spending_df.columns[:10]][cash_spending_df['Gender of the informant']==' female'].sum(axis=0).sort_values().plot(kind='barh')

# %%
female_spending[female_spending.columns[:9]].sum(axis=0).sort_values()

# %%
male_spending[male_spending.columns[:9]].sum(axis=0).sort_values()

# %%
cash_spent_by_gender_labels = [
    "paid the loan taken from neighbourrelatives",
    "spent unnecessarily",
    "purchased agricultural realted materials",
    "spent  on childrens education",
    "bought new clothes",
    "others",
    "spent to celebrate festival",
    "spent on house repairs",
    "bought daily necessities",
]
cash_spent_by_gender_female = [1, 2, 4, 4, 9, 14, 15, 29, 47]
cash_spent_by_gender_male = [0, 1, 10, 9, 23, 39, 24, 54, 107]
cash_spent_df_gender = pd.DataFrame(
    {"male": cash_spent_by_gender_male, "female": cash_spent_by_gender_female},
    index=cash_spent_by_gender_labels,
)
cash_spent_df_gender.plot.barh(stacked=True, width=0.75)
plt.title("Cash spent by gender", fontsize=20)

# %%
cash_spent_by_gender_labels = [
    "paid the loan taken from neighbourrelatives",
    "spent unnecessarily",
    "purchased agricultural realted materials",
    "spent  on childrens education",
    "bought new clothes",
    "others",
    "spent to celebrate festival",
    "spent on house repairs",
    "bought daily necessities",
]
cash_spent_by_gender_female = [1, 2, 4, 4, 9, 14, 15, 29, 47]
cash_spent_by_gender_male = [0, 1, 10, 9, 23, 39, 24, 54, 107]
cash_spent_df_gender = pd.DataFrame(
    {"male": cash_spent_by_gender_male, "female": cash_spent_by_gender_female},
    index=cash_spent_by_gender_labels,
)
cash_spent_df_gender.plot.barh(stacked=True, width=0.75)
plt.title("Cash spent by gender", fontsize=20)
df_total = cash_spent_df_gender["male"] + cash_spent_df_gender["female"]
df_rel = cash_spent_df_gender[cash_spent_df_gender.columns[:]].div(df_total, 0) * 100
# add percentages to the plot
for n in df_rel:
    for i, (cs, ab, pc) in enumerate(
        zip(
            cash_spent_df_gender.iloc[:, :].cumsum(1)[n],
            cash_spent_df_gender[n],
            df_rel[n],
        )
    ):
        plt.text(
            cs - ab / 2,
            i,
            str(np.round(pc, 1)) + "%",
            va="center",
            ha="center",
            rotation=20,
            fontsize=10,
        )

# %%
female_spending[female_spending.columns[:9]].sum(axis=0).sort_values().plot(kind="barh")

# %%
male_spending[male_spending.columns[:9]].sum(axis=0).sort_values().plot(kind="barh")

# %%
cash_spending_df.rename(columns={"others": "others spending"}, inplace=True)

# %%
# this holds only the list on which cash was spent to ease slicing
cash_spending_columns = cash_spending_df.columns[:9]
cash_spending_columns

# %%
# add the age brackets to the cash spending df and get statistics based on ages.
cash_spending_df[data_df.columns[22:30]] = data_df[data_df.columns[22:30]]

# %%
# cash spending for families with children less than 18 years.
plt.figure(figsize=(15, 10))
plt.title("Cash spending for families with children age less than 18", fontsize=20)
plt.xlabel("Frequency")
cash_spending_df[cash_spending_columns][
    (cash_spending_df.male0_5 > 0)
    | (cash_spending_df.female0_5 > 0)
    | (cash_spending_df.male6_17 > 0)
    | (cash_spending_df.female6_17 > 0)
].sum(axis=0).sort_values().plot(kind="barh")

# %%
# cash spending for families with children less than 18 years.
plt.figure(figsize=(15, 10))
plt.title(
    "Cash spending for families without children of age less than 18", fontsize=20
)
plt.xlabel("Frequency")
cash_spending_df[cash_spending_columns][
    (cash_spending_df.male0_5 == 0)
    & (cash_spending_df.female0_5 == 0)
    & (cash_spending_df.male6_17 == 0)
    & (cash_spending_df.female6_17 == 0)
].sum(axis=0).sort_values().plot(kind="barh")

# %%
# cash spending for families with children less than 18 years.
plt.figure(figsize=(15, 10))
plt.title(
    "Cash spending for families having atleast a person age 60+ but without children of age less than 18",
    fontsize=20,
)
plt.xlabel("Frequency")
cash_spending_df[cash_spending_columns][
    (cash_spending_df.male0_5 == 0)
    & (cash_spending_df.female0_5 == 0)
    & (cash_spending_df.male6_17 == 0)
    & (cash_spending_df.female6_17 == 0)
    | (cash_spending_df["male60+"] > 0)
    | (cash_spending_df["female60+"] > 0)
].sum(axis=0).sort_values().plot(kind="barh")

# %%
cash_spending_df[data_df.columns[22:30]]

# %%
plt.figure(figsize=(15, 10))
plt.title("Did you receive relief on time", fontsize=20)
plt.ylabel("Frequency", fontsize=18)
data_df["Did you receive relief on time"].value_counts().sort_values().plot(kind="bar")

# %%
df = data_df

# %%
data_df.loc[:, "Did you receive relief on time":].columns[2:10]

# %%
# we maintain the motives as the column names.
for col in df.loc[:, "Did you receive relief on time":].columns[2:10]:
    df.rename(
        columns={
            col: col.replace("Why you didnt recieve relief items on time", "").lstrip()
        },
        inplace=True,
    )


# %%
plt.tick_params(axis="both", labelsize=18)
df[df.loc[:, "Did you receive relief on time":].columns[2:10]].sum().sort_values().plot(
    kind="barh"
)

# %%
df_late_relief = df[df["Did you receive relief on time"] == "  no"].loc[
    :,
    "Did you receive relief on time":"Did you find the relief cash and otherdistributin area safe",
]

# %%
df_late_relief.rename(columns={"othersplease specify2": "others2"}, inplace=True)

# %%
len(df_late_relief["others"])

# %%
df_late_relief.columns

# %%
motive_late_relief = {}
for col in df_late_relief.columns[2:9]:
    val = sum(df_late_relief[col])
    if val != 0:
        motive_late_relief[col] = val
motive_late_relief["others"] = 6


# %%
plt.title("Motive for receiving relief late")
plt.xlabel("Frequency")
pos = np.arange(len(motive_late_relief.values())) + 0.5
plt.barh(pos, motive_late_relief.values(), align="center", color="green")
plt.yticks(pos, motive_late_relief.keys())


# %%
plt.figure(figsize=(10, 10))
plt.title("Why receive supplies?", fontsize=20)
plt.xlabel("Freqnency", fontsize=16)
data_df["Why did you choose to receive relief supplies"].value_counts().plot(
    kind="barh"
)

# %%
plt.figure(figsize=(15, 10))
plt.title("Suggested items to be included in future", fontsize=20)
plt.xlabel("Frequency")
data_df[
    "What do you suggest should be included in the relief of Nepal Red Cross in the future"
].value_counts().plot(kind="barh")

# %%
plt.figure(figsize=(15, 10))
plt.title("Suggested items to be included in futureage 59", fontsize=20)
plt.xlabel("Frequency")
data_df[
    "What do you suggest should be included in the relief of Nepal Red Cross in the future"
][data_df["age of the informant"] > 59].value_counts().plot(kind="barh")

# %%
data_df["Gender of the informant"].unique()

# %%
# extracting responses on suggested items by gender of the respondent.
data_df[
    "What do you suggest should be included in the relief of Nepal Red Cross in the future"
][data_df["Gender of the informant"] == "  male"].value_counts()

# %%
data_df[
    "What do you suggest should be included in the relief of Nepal Red Cross in the future"
][data_df["Gender of the informant"] == " female"].value_counts()

# %%
index_suggestions = [
    "food and non-food materials",
    "Cash",
    "others",
    "Cash      food and non-food materials",
    "food and non-food materials    others",
    "Cash    others",
    "Cash      food and non-food materials    others",
]
female_suggestions = [
    102,
    36,
    19,
    18,
    9,
    1,
    1,
]  # these are manually extracted to ensure that corresponding entries are matched
male_suggestions = [46, 23, 8, 10, 2, 0, 0]
suggested_items_df_gender = pd.DataFrame(
    {"male": male_suggestions, "female": female_suggestions}, index=index_suggestions
)
suggested_items_df_gender.plot.barh(stacked=True, width=0.75)
plt.title("Suggested Items by gender", fontsize=20)
df_total = suggested_items_df_gender["male"] + suggested_items_df_gender["female"]
df_rel = (
    suggested_items_df_gender[suggested_items_df_gender.columns[:]].div(df_total, 0)
    * 100
)
# add percentages to the plot
for n in df_rel:
    for i, (cs, ab, pc) in enumerate(
        zip(
            suggested_items_df_gender.iloc[:, :].cumsum(1)[n],
            suggested_items_df_gender[n],
            df_rel[n],
        )
    ):
        plt.text(
            cs - ab / 2,
            i,
            str(np.round(pc, 1)) + "%",
            va="center",
            ha="center",
            rotation=20,
            fontsize=10,
        )

# %%
index_suggestions = [
    "food and non-food materials",
    "Cash",
    "others",
    "Cash      food and non-food materials",
    "food and non-food materials    others",
    "Cash    others",
    "Cash      food and non-food materials    others",
]
female_suggestions = [102, 36, 19, 18, 9, 1, 1]
male_suggestions = [46, 23, 8, 10, 2, 0, 0]
suggested_items_df_gender = pd.DataFrame(
    {"male": male_suggestions, "female": female_suggestions}, index=index_suggestions
)
suggested_items_df_gender.plot.barh(width=0.65)
plt.title("Items suggested to be added to packages by gender", fontsize=20)
plt.figure(figsize=(15, 10))

# %%
plt.figure(figsize=(15, 10))
plt.title("Suggested items to be included in future", fontsize=20)
plt.xlabel("Frequency")
data_df[
    "What do you suggest should be included in the relief of Nepal Red Cross in the future"
].value_counts().plot(kind="barh")

# %%
data_df["age of the informant"]

# %%
# to rename columns corresponnding to what a family got from the NRCS package
column_dic = {}

# %%
for col in data_df.loc[
    :,
    "What did your family get from Nepal Red Cross Society after the flood and landslide":,
].columns[1:17]:
    new_column = col.replace(
        "What did your family get from Nepal Red Cross Society after the flood and landslide",
        "",
    )
    column_dic[col] = new_column.lstrip()

# %%
# renaming the columns
data_df.rename(columns=column_dic, inplace=True)


# %%
column_dic.values()

# %%
df_family_recieved = data_df.loc[
    :,
    "tent":"While distributing relief materialswere arrangements made to distribute relief items by avoiding COVID-19",
]
df_family_recieved.rename(columns={"othersplease specify": "other Items"}, inplace=True)

# %%
df_family_recieved.rename(
    columns={
        "What did your family get from Nepal Red Cross Society after the flood and landslide  others": "other things"
    },
    inplace=True,
)

# %%
df_family_recieved.drop(
    "While distributing relief materialswere arrangements made to distribute relief items by avoiding COVID-19",
    axis=1,
    inplace=True,
)

# %%
# this bar graph shows other items which were bought but do not belong to the previous categories.
plt.figure(figsize=(16, 20))
plt.title("Other Items bought using the cash", fontsize=32)
plt.tick_params(axis="both", labelsize=26)
df_family_recieved["other Items"].value_counts().plot(kind="barh")

# %%
# we clean the data belonging to the column other items to have more insight on the actual items purchased.
df_family_recieved["other Items"].unique()

# %%
column_dic.values()

# %%
# plotting the counts of different items received by affected population
plt.figure(figsize=(10, 10))
plt.title("Items received", fontsize=20)
plt.xlabel("Frequency")
plt.tight_layout()
data_df[column_dic.values()].sum(axis=0).sort_values().plot(kind="barh")

# %%
data_df[column_dic.values()][data_df["age of the informant"] < 30].sum(
    axis=0
).sort_values().plot(kind="barh")

# %%
data_df[column_dic.values()][data_df["age of the informant"] <= 30].sum(
    axis=0
).sort_values()

# %%
data_df["age of the informant"].unique()

# %%
plt.figure(figsize=(15, 10))
plt.title("How respondents age 30 to 49 spent cash received", fontsize=20)
data_df[column_dic.values()][
    (data_df["age of the informant"] > 30) & (data_df["age of the informant"] < 50)
].sum(axis=0).sort_values().plot(kind="barh")

# %%
data_df[column_dic.values()][(data_df["age of the informant"] > 30)].sum(
    axis=0
).sort_values()

# %%
packages_recieved_by_age_groups = [
    "Aquatabs",
    "Informationeducation and communication materials",
    "Net",
    "Kishori Kit",
    "Family Tent",
    "Rope",
    "Children Kit",
    "Mattress",
    "Dignity Kit",
    "non food relief item set",
    "Kitchen set",
    "blanket",
    "soap for handwash",
    "Hygiene Kit",
    "Bucket",
    "tent",
]
age_below30 = [0, 0, 2, 2, 4, 4, 5, 8, 11, 19, 21, 23, 35, 35, 38, 39]
age30_59 = [1, 1, 4, 1, 2, 8, 8, 20, 21, 36, 45, 54, 69, 77, 83, 96]
age60_above = [0, 0, 0, 0, 1, 3, 0, 1, 2, 8, 2, 5, 7, 4, 10, 12]
width = 0.75  # the width of the bars: can also be len(x) sequence
fig, ax2 = plt.subplots()
plt.tick_params(axis="both", labelsize=16)
plt.figure(figsize=(30, 20))
ax2.barh(packages_recieved_by_age_groups, age_below30, width, label="Age below 30")
ax2.barh(packages_recieved_by_age_groups, age30_59, width, label="Age 30 to 59")
ax2.barh(packages_recieved_by_age_groups, age60_above, width, label="Age 60 plus")
ax2.set_xlabel("Frequency")
ax2.set_title("Package relief received by age groups", fontsize=20)
ax2.legend()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/relief_received_by_age_groups.png"
)
plt.show()

# %%
data_df[column_dic.values()][
    (data_df["age of the informant"] > 29) & (data_df["age of the informant"] < 59)
].sum(axis=0).sort_values().plot(kind="barh")

# %%
data_df[
    "Did you get cash relief from Nepal Red Cross Society along with non-food items"
].value_counts()

# %%
312 / 332

# %%
# check  if further info was captured for those who said the  cash didn't help -nothing
data_df.loc[
    :, "Did you get cash relief from Nepal Red Cross Society along with non-food items":
].columns

# %%
plt.figure(figsize=(15, 10))
plt.title(
    "How did the NRCS and volunteers treat you at relief distribution center",
    fontsize=20,
)
plt.ylabel("Frequency", fontsize=16)
data_df[
    "How did the Nepal Red Cross Society and voluneers treat you at the place where the relief was distributed"
].value_counts().sort_values().plot(kind="bar")
