#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cci_nepal
import pandas as pd
import numpy as np
import random
import logging

from cci_nepal.getters import get_data as grd

# Set project directory
project_dir = cci_nepal.PROJECT_DIR

column_names = grd.get_lists(f"{project_dir}/cci_nepal/config/dummy_column_names.csv")
df_dummy = pd.DataFrame(np.nan, index=range(0, 3211), columns=range(0, 72))


district_list = ["Sindupalchok", "Mahottari"]
district = [district_list[random.randrange(len(district_list))] for i in range(3211)]
df_dummy.iloc[:, 0] = district


df_dummy.iloc[:, 6:16] = np.random.randint(0, 3, size=(3211, 10))

df_dummy.iloc[:, 17:27] = np.random.randint(0, 3, size=(3211, 10))


ethnicity_list = [
    "adibasi___janjati___newar",
    "brahmin___chettri___sanyashi___thakuri",
    "dalit",
    "madhesi",
    "other",
    "prefer_not_to_answer",
]
ethnicity = [ethnicity_list[random.randrange(len(ethnicity_list))] for i in range(3211)]

df_dummy.iloc[:, 27] = ethnicity


df_dummy.iloc[:, 30:36] = np.random.randint(0, 1, size=(3211, 6))


house_material_list = [
    "mud_bonded_bricks_stone",
    "rcc_with_pillar",
    "other",
    "cement_bonded_bricks_stone",
    "wooden_pillar",
]
house_material = [
    house_material_list[random.randrange(len(house_material_list))] for i in range(3211)
]


df_dummy.iloc[:, 36] = house_material


house_material_other_list = [
    "clay",
    "rcc_with_pillar",
    "other",
    "cement_bonded_bricks_stone",
    "wooden_pillar",
]
house_material = [
    house_material_list[random.randrange(len(house_material_list))] for i in range(3211)
]


df_dummy.iloc[:, 37] = [
    item for sublist in [["clay"] * 450, ["other"] * 2761] for item in sublist
]


df_dummy.iloc[:, 38] = np.random.randint(1, 2, size=(3211, 1))


previous_nfri_list = ["yes", "no"]
previous_nfri = [
    previous_nfri_list[random.randrange(len(previous_nfri_list))] for i in range(3211)
]


df_dummy.iloc[:, 40] = previous_nfri

NFRI_importance = ["Essential", "Desirable", "Unnecessary"]

nfri_preference_list = []
for i in range(0, 11):
    nfri_preference_list.append(
        [NFRI_importance[random.randrange(len(NFRI_importance))] for i in range(3211)]
    )

df_dummy.iloc[:, 43:54] = np.transpose(nfri_preference_list)
df_dummy.iloc[:, 56:67] = np.transpose(nfri_preference_list)
df_dummy.columns = column_names

df_dummy.to_csv(
    f"{project_dir}/outputs/data/data_for_modelling/dummy_data.csv", index=False
)
logging.info("Dummy data is created and stored in the outputs folder.")
