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
import nltk
import re
import logging

# %%
# nltk.download("words")

# %%
project_directory = cci_nepal.PROJECT_DIR
logging.info(project_directory)

# %%
db = pd.read_excel(
    f"{project_directory}/inputs/data/wp2_data_scoping/PDM_ Datasheet.xlsx"
)

# %%
db.shape

# %%
db.drop(db.columns[2], axis=1, inplace=True)


# %%
def clean_df_columns(df):
    df.columns = df.columns.str.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)
    df.columns = df.columns.str.lstrip()


# %%
clean_df_columns(db)

# %%
db = db.replace("[^a-zA-Z0-9\-\s]+", "", regex=True)

# %%
db.head(5)

# %%
percent_missing = db.isnull().sum() * 100 / len(db)
missing_value_df = pd.DataFrame(
    {"column_name": db.columns, "percent_missing": percent_missing}
)
missing_value_df.sort_values("percent_missing", inplace=True)

# %%
missing_value_df["percent_missing"].plot(kind="bar")
plt.xticks([])

# %%
plt.figure(figsize=(15, 10))
missing_value_df["percent_missing"].tail(60).plot(kind="bar")

# %%
# db.to_excel('clean-temp.xlsx', index=False)

# %%
db["Ethnicity of the informant"] = (
    db["Ethnicity of the informant"].str.replace("\d+", "").copy()
)

# %%
fig = plt.figure()
db["Ethnicity of the informant"].value_counts().sort_values().plot(kind="barh")
fig.suptitle("Ethnicity of the informant", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Ethnicity", fontsize=12)

# %%
nfri_items = {}
nfri_df = db.iloc[:, 52:69]
nfri_df.columns = nfri_df.columns.str.replace(
    "What did your family get from Nepal Red Cross Society after the flood and landslide",
    "",
)
nfri_df.columns = nfri_df.columns.str.strip()

# %%
for col in nfri_df.columns:
    nfri_df[col] = nfri_df[col].fillna(0)
    nfri_items[col] = sum(nfri_df[col])

# %%
fig = plt.figure()

plt.bar(nfri_items.keys(), nfri_items.values())
plt.xticks(rotation=90)
plt.tight_layout()

fig.suptitle(
    "What did your family get from Nepal Red Cross Society after the flood and landslide?",
    fontsize=14,
)
plt.xlabel("Items", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.show()

# %%
fig = plt.figure(figsize=(10, 15))
db.iloc[:, 69].value_counts().sort_values().plot(kind="barh")
fig.suptitle("Other relief items specified", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Items", fontsize=12)

# %%
cash_relief = {}
cr_df = db.iloc[:, 90:99]
cr_df.columns = cr_df.columns.str.replace(
    "For what did you spend the money you received from the Red Cross", ""
)
cr_df.columns = cr_df.columns.str.strip()

# %%
for col in cr_df.columns:
    cr_df[col] = cr_df[col].fillna(0)
    cash_relief[col] = sum(cr_df[col])

# %%
fig = plt.figure()

plt.bar(cash_relief.keys(), cash_relief.values())
plt.xticks(rotation=90)

fig.suptitle("How cash relief was spent", fontsize=14)
plt.xlabel("Spending activity", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.show()

# %%
fig = plt.figure()
db[
    "Did the cash relief provided by the Red Cross met your immediate needs"
].value_counts().sort_values().plot(kind="barh")
plt.xticks(rotation=90)
fig.suptitle(
    "Did the cash relief provided by the Red Cross met your immediate needs?",
    fontsize=14,
)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Answer", fontsize=12)

plt.show()


# %%
db[
    "Has the relief materials you received caused jealousy in the community"
].value_counts()

# %%
db[
    "What do you suggest should be included in the relief of Nepal Red Cross in the future"
]

# %%
db.shape

# %%
relief_sugg = {}
rs_df = db.iloc[:, 132:135]
# rs_df.columns = rs_df.columns.str.replace('For what did you spend the money you received from the Red Cross', '')
# rs_df.columns = rs_df.columns.str.strip()

# %%
rs_df.head(1)

# %%
