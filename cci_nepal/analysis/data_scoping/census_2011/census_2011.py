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
from cci_nepal.getters.data_scoping import get_sample_data as gsd
from cci_nepal.pipeline.data_scoping import mis_sample_pipeline as msp
import cci_nepal
import matplotlib.pyplot as plt


# %%
from opencage.geocoder import OpenCageGeocode

# %%
# Set directory
project_directory = cci_nepal.PROJECT_DIR

# %% [markdown]
# ### Read in and clean 2017 flood data

# %%
feedback_2017 = gsd.read_excel_file(
    f"{project_directory}/inputs/data/wp2_data_scoping/nepal_2017_flood_sample.xlsx"
)

# %%
# Drop survey intro column
feedback_2017.drop(feedback_2017.columns[4], axis=1, inplace=True)
feedback_2017 = msp.clean_df_columns(feedback_2017)  # Calling cleaning function

# %%
feedback_2017.head(2)

# %% [markdown]
# ### 2011 Census data

# %%
taplejung = pd.read_excel(
    f"{project_directory}/inputs/data/wp2_data_scoping/district_profile_census_2011/01 Census 2011 District Taplejung.xlsx",
    sheet_name="Ind01P",
    skiprows=3,
)

# %%
taplejung.columns = [
    "remove1",
    "municipality",
    "household",
    "population",
    "male",
    "female",
    "avg_hsehld_size",
    "gender_ratio",
]
taplejung.drop(taplejung.index[:3], inplace=True)

# %%
taplejung.drop(taplejung.columns[0], axis=1, inplace=True)

# %%
taplejung.head(5)

# %%
key = ""
geocoder = OpenCageGeocode(key)

# %%
results = geocoder.reverse_geocode(26.8181799, 85.5206578)

# %%
results

# %%
# results[0]['components']['municipality']

# %%
municipalities = []

for i in feedback_2017.index:
    results = geocoder.reverse_geocode(
        feedback_2017["GPSlatitude"][i], feedback_2017["GPSlongitude"][i]
    )
    munic = results[0]["components"]["municipality"]
    municipalities.append(munic)

# %%
municipalities

# %%
