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
data_df = pd.read_excel(f"{project_dir}/inputs/data/PDM_ Datasheet.xlsx")

# %%
data_df.shape


# %%
def clean_df_columns(df):
    # df.columns = (re.sub(r"[^\x00-\x7f]", r"", col) for col in df.columns)
    # df.columns = df.columns.str.replace("[^a-zA-Z\s]+", "", regex=True)
    df.columns = df.columns.str.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)
    df.columns = df.columns.str.lstrip()
    logging.info(df.columns)
    return df


# %%
clean_df_columns(data_df)

# %%
# renaming age group columns
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
# clean the data by removing unwanted characters
data_df = data_df.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)

# %%
percent_missing = data_df.isnull().sum() * 100 / len(data_df)
missing_value_df = pd.DataFrame(
    {"column_name": data_df.columns, "percent_missing": percent_missing}
)


# %%
missing_value_df.sort_values("percent_missing", inplace=True)

missing_value_df["percent_missing"].plot(kind="bar")
plt.xticks([])


# %%
plt.figure(figsize=(15, 10))
missing_value_df["percent_missing"].tail(45).plot(kind="bar")
plt.title("columns with missing values > 82%")
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/missing_values_greater80.png"
)
plt.show()

# %%
plt.figure(figsize=(20, 15))
missing_value_df["percent_missing"].plot(kind="hist")
plt.title("Distribution of columns with missing values")
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/missing_values_greater80b.png"
)
plt.show()

# %%
len(missing_value_df[missing_value_df["percent_missing"] > 82].column_name.unique())

# %%
# remove columns which all have null values
data_df.dropna(axis=1, how="all", inplace=True)

# %%
data_df.shape

# %%
# number of columns that were completely null
146 - 129

# %%
# replace the null values for the age groups for onward processing. The assumption here is that missing values
# imply situations of non-applicability. For example, the absence of male0_5 means no children of this age in houshold
for col in data_df.columns[22:30]:
    data_df[col] = data_df[col].fillna(0)

# %%

# %%
beneficiaries_by_age_group = {}
for col in data_df.columns[22:30]:
    beneficiaries_by_age_group[col] = sum(data_df[col])

# %%
print(beneficiaries_by_age_group)

# %%
labels = ["Age0_5", "Age6_17", "Age18_59", "Age60_above"]
men_means = [87.0, 232, 541, 63]
women_means = [72, 226, 550, 70]
men_std = [2, 3, 4, 1, 2]
women_std = [3, 5, 2, 3, 3]
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
plt.bar(
    beneficiaries_by_age_group.keys(),
    beneficiaries_by_age_group.values(),
    color=["cyan", "green", "cyan", "green", "cyan", "green", "cyan", "green"],
)
plt.xticks(rotation=45)
plt.title("distribution of beneficiaries by age bracket and gender")
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/beneficiaries_by_age_groups.png"
)
plt.show()

# %%
beneficiaries_by_age_group

# %%
age_60 = (70 + 63) / (541 + 550 + 87 + 72 + 232 + 226 + 63 + 70)
logging.info(age_60)

# %%
age_under5 = (87 + 72) / (541 + 550 + 87 + 72 + 232 + 226 + 63 + 70)
logging.info(age_under5)

# %%
# drop columns with more than 82% null values.
for col in missing_value_df[missing_value_df["percent_missing"] > 82].column_name:
    if col in data_df.columns:
        data_df.drop(col, axis=1, inplace=True)

# %% [markdown]
# ## Which communication means adopted by nepal RC proved more effective for the affected communities?
# ## Of the financial aid provided, how can one prioritize the needs of the communities based on how the aid was used?
#
# ## based on the time taken to arrive at the relief distribution centers, how convenient is it to reduce the number of people who travelled between 3-4 hours to get the aid? Can new centers that would reduce this travel time be created?
#
# # For the communication modes used, were resources equitably distributed to the different channels?
#

# %%
# replace the 9 null values with the mode for the specific column
data_df["How were you notified about the relief delivery date"] = data_df[
    "How were you notified about the relief delivery date"
].fillna(
    "        From community representatives TeachersCommunity leaders and peoples representatives"
)
communication_modes = data_df[
    "How were you notified about the relief delivery date"
].unique()

# %%
communication_modes_frequency = {}
for m in communication_modes:
    freq = data_df["How were you notified about the relief delivery date"][
        data_df["How were you notified about the relief delivery date"] == m
    ].count()
    communication_modes_frequency[m] = freq
    print(m, freq)

# %%
explode = (0.1, 0, 0, 0, 0)  # only "explode" the 1st slice -ICT

fig2, ax2 = plt.subplots()
ax2.pie(
    communication_modes_frequency.values(),
    explode=explode,
    labels=communication_modes_frequency.keys(),
    autopct="%1.1f%%",
    shadow=True,
    startangle=90,
)
ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig(f"{project_dir}/outputs/figures/nepal_descriptive/communication_modes.png")
plt.show()


# %%

# %%
# check for duplicates
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
data_df["Are you satisfied with the relief distribution process"].value_counts()

# %% [markdown]
# ## What accounts for the fact that some districts are more represented than others?

# %%
district_counts = data_df["District name"].value_counts()

# %%
district_counts

# %%
district_counts[:14]

# %%
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
data_df.columns[20:30]

# %%
data_df["Why did you choose to receive relief supplies"].value_counts()

# %%
data_df["Why did you choose to receive relief supplies"].value_counts().sum()

# %%
data_df["Why did you choose to receive relief supplies"].value_counts().plot(kind="bar")
plt.figure(figsize=(15, 20))
plt.tight_layout()
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/motive_for_receiving_id.png"
)

# %%
data_df["Why did you choose to receive relief supplies"].value_counts()

# %%
(123 + 100) / (123 + 100 + 57 + 27 + 9 + 4 + 3)

# %%
52 / 146

# %%
