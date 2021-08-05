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

# %%
nltk.download("words")

# %%
project_directory = cci_nepal.PROJECT_DIR

# %%
db = pd.read_excel(
    f"{project_directory}/inputs/data/wp2_data_scoping/PDM_ Datasheet.xlsx"
)

# %%
db.shape

# %%
db.drop(db.columns[2], axis=1, inplace=True)

# %%
db.head(1)

# %%
cols_update = []
for col in db.columns:
    result = re.sub(r"[^\x00-\x7f]", r"", col)
    cols_update.append(result)
db.columns = cols_update

# %%
db.columns = db.columns.str.replace("b", "", 1, regex=False)
db.columns = db.columns.str.replace("[^A-Za-z\s]+", "", regex=True)

# %%
db.head(1)

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
