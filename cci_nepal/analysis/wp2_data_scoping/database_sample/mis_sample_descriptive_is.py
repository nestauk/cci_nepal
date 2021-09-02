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

# %%
# Set directory
project_directory = cci_nepal.PROJECT_DIR

# %%
db = gsd.read_excel_file(
    f"{project_directory}/inputs/data/wp2_data_scoping/PDM_ Datasheet.xlsx"
)

# %%
db.shape  # Shape of dataset

# %%
# Drop survey intro column
db.drop(db.columns[2], axis=1, inplace=True)

# %%
db = msp.clean_df_columns(db)  # Calling function

# %%
# Create table to show percent missing across columns
percent_missing = db.isnull().sum() * 100 / len(db)
missing_value_df = pd.DataFrame(
    {"column_name": db.columns, "percent_missing": percent_missing}
)
missing_value_df.sort_values("percent_missing", inplace=True)

# %%
# Plot figure
fig = plt.figure()
missing_value_df["percent_missing"].plot(kind="bar")
fig.suptitle("Percent missing across columns", fontsize=14)
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.xticks([])
plt.savefig(
    f"{project_directory}/outputs/figures/data_scoping/mis_sample/percent_missing_all_columns.png"
)

# %%
# Plot figure
fig = plt.figure(figsize=(15, 10))
missing_value_df["percent_missing"].tail(60).plot(kind="bar")
fig.suptitle("Percent missing - largest 60", fontsize=14)
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Percentage", fontsize=12)

# %%
# Clean ethnicity column
db["Ethnicity of the informant"] = (
    db["Ethnicity of the informant"].str.replace("\d+", "").copy()
)

# %%
# Plot figure
fig = plt.figure()
db["Ethnicity of the informant"].value_counts().sort_values().plot(kind="barh")
fig.suptitle("Ethnicity of the informant", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Ethnicity", fontsize=12)

plt.savefig(
    f"{project_directory}/outputs/figures/data_scoping/mis_sample/ethnicity_of_informant.png"
)

# %%
nfri_items = msp.group_counts(
    52,
    69,
    "What did your family get from Nepal Red Cross Society after the flood and landslide",
    db,
)

# %%
# Plot figure
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
# Plot figure
fig = plt.figure(figsize=(10, 15))
db.iloc[:, 69].value_counts().sort_values().plot(kind="barh")
fig.suptitle("Other relief items specified", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Items", fontsize=12)

plt.tight_layout()

plt.savefig(
    f"{project_directory}/outputs/figures/data_scoping/mis_sample/other_relief_items.png"
)

# %%
cash_relief = msp.group_counts(
    90, 99, "For what did you spend the money you received from the Red Cross", db
)

# %%
# Plot figure
fig = plt.figure()

plt.bar(cash_relief.keys(), cash_relief.values())
plt.xticks(rotation=90)

fig.suptitle("How cash relief was spent", fontsize=14)
plt.xlabel("Spending activity", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.savefig(
    f"{project_directory}/outputs/figures/data_scoping/mis_sample/how_relief_spent.png"
)

plt.show()

# %%
# Plot figure
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

plt.savefig(
    f"{project_directory}/outputs/figures/data_scoping/mis_sample/did_relief_meet_needs.png"
)

plt.show()


# %%
# Save cleaned file
db.to_excel(
    f"{project_directory}/outputs/data/data_scoping/mis_sample_cleaned.xlsx",
    index=False,
)
