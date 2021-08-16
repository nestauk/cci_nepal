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
data_df.columns[20:40]

# %%
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
percent_missing = data_df.isnull().sum() * 100 / len(db)
missing_value_df = pd.DataFrame(
    {"column_name": db.columns, "percent_missing": percent_missing}
)

# %%
# remove columns which all have null values
data_df.dropna(axis=1, how="all", inplace=True)

# %%
data_df.shape

# %%
data_df.columns[22:30]

# %%
# replace the null values for the age groups for onward processing. The assumption here is that missing values
# imply situations of non-applicability. For example, the absence of male0_5 means no children of this age in houshold
for col in data_df.columns[22:30]:
    data_df[col] = data_df[col].fillna(0)

# %%
beneficiaries_by_age_group = {}
for col in data_df.columns[22:30]:
    beneficiaries_by_age_group[col] = sum(data_df[col])

# %%
plt.bar(beneficiaries_by_age_group.keys(), beneficiaries_by_age_group.values())
plt.xticks(rotation=45)
plt.title("distribution of beneficiaries by age bracket and gender")
plt.savefig(
    f"{project_dir}/outputs/figures/nepal_descriptive/beneficiaries_by_age_groups.png"
)
plt.show()

# %%

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
data_df["How were you notified about the relief delivery date"].unique()

# %%
data_df["How were you notified about the relief delivery date"].isnull().sum()

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
plt.bar(
    communication_modes_frequency.keys(),
    communication_modes_frequency.values(),
    color=["green", "gray", "blue", "black", "red"],
)
plt.ylabel("No of people reached")
plt.xticks(rotation=45)
plt.show()

# %%
