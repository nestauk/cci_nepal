#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [10, 6]

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.options.display.max_rows = 4000
# get_ipython().run_line_magic('load_ext', 'nb_black')


project_dir = cci_nepal.PROJECT_DIR
logging.info("Logging the project directory below:")
logging.info(project_dir)


# #### Reading the data ####

df = grd.read_csv_file(f"{project_dir}/inputs/data/real_data/Full_Consent_Hill.csv")


# #### Dropping columns that are not related to Features (like introduction, notes, etc.)

# In[5]:


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


# In[6]:


df.drop(df.columns[columns_to_drop], axis=1, inplace=True)


# #### Missing Data ####

# In[7]:


df.isnull().sum().sort_values(ascending=False)


# Mostly non voluntary questions where 0 was the expected answer are the missing value features. Thus, filling in the missing values with 0 in the step below.

# In[8]:


df.fillna(0, inplace=True)


# #### Feature Creation ####

# In[9]:


df.insert(2, "Total_Members", df.iloc[:, 2:22].sum(axis=1))


# In[10]:


df.insert(3, "Total_Male", df.iloc[:, 3:13].sum(axis=1))


# In[11]:


df.insert(4, "Total_Female", df.iloc[:, 14:24].sum(axis=1))


# In[12]:


df.insert(5, "Total_Children", df.iloc[:, [5, 6, 7, 15, 16, 17]].sum(axis=1))


# In[13]:


df.insert(6, "Total_Health_Difficulty", df.iloc[:, 28:34].sum(axis=1))


# #### Respondent Gender ####

# In[14]:


print(df.iloc[:, 0].value_counts(normalize=True))
df.iloc[:, 0].value_counts(normalize=False).plot(kind="bar")


# Almost 50 percent balance between the Gender Respondents, with Female slighly higher which is preferable to our need.

# #### Respondent Age ####

# In[15]:


print(df.iloc[:, 1].value_counts(normalize=True))


# In[16]:


df.iloc[:, 1].value_counts(normalize=True).plot(kind="bar")


# Respondents in 30s and 40s are the most frequent ones, which is quite expected also.

# #### Total Members ####

# In[17]:


# print(df.iloc[:, 2].value_counts(normalize=False))
print(df.iloc[:, 2].mean())
df.iloc[:, 2].value_counts(normalize=False).plot(kind="bar")


# Although the average household size is nearly 6, it is influenced heavily by the Outlier counts (like 49, 32 etc.) As confirmed by the bar above, size 4 is the most frequent one, followed by 5 and 6 closely, and 3 and 7 after that.

# #### Total Male ####

# In[18]:


# print(df.iloc[:, 3].value_counts(normalize=False))
df.iloc[:, 3].value_counts(normalize=False).plot(kind="bar")


# #### Outlier Total Male Observation #####

# In[19]:


df.loc[
    df["Total_Male"] == 45,
]


# #### Total Female ####

# In[20]:


# print(df.iloc[:, 4].value_counts(normalize=False))
df.iloc[:, 4].value_counts(normalize=False).plot(kind="bar")


# #### Outlier Total Female Observation ####

# In[21]:


df.loc[df["Total_Female"] == 34]


# #### Total Children ####

# In[22]:


# print(df.iloc[:, 5].value_counts(normalize=False))
df.iloc[:, 5].value_counts(normalize=False).plot(kind="bar")


# #### Outlier Total Children Observation ####

# In[23]:


df.loc[df["Total_Children"] == 34]


# #### Health Difficulty ####

# In[24]:


# print(df.iloc[:, 6].value_counts(normalize=False))
df.iloc[:, 6].value_counts(normalize=False).plot(kind="bar")


# As we can see, most of the households have no health difficulty.

# #### Most Frequent Health Difficulty ####

# In[25]:


df.iloc[:, 29:35].apply(pd.Series.value_counts, normalize=False).T


# As we can seem the most frequent Health Difficulty is seeing related, followed to hearing and then walking/climbing.

# #### Outlier Health Diffculty Observation ####

# In[26]:


df.loc[
    df[
        "Using.usual.language..how.many.members.of.your.household.have.difficulty.communicating..for.example.understanding.or.being.understood."
    ]
    == 53
]


# #### Ethnicity ####

# In[27]:


print(df.iloc[:, 27].value_counts(normalize=True))
df.iloc[:, 27].value_counts(normalize=True).plot(kind="bar")


# 8 percent of people belong to Madhesi ethinicity, which is main ethnicity of the other district Mahottari. So, should be interesting comparing the preferences of same ethnicity people across two different geographies later!

# #### House Material ####

# In[28]:


print(df.iloc[:, 35].value_counts(normalize=True))
df.iloc[:, 35].value_counts(normalize=True).plot(kind="bar")


# #### Total Income Generating Members ####

# In[29]:


# print(df.iloc[:, 37].value_counts(normalize=True))
df.iloc[:, 37].value_counts(normalize=True).plot(kind="bar")


# Nothing abnormal detected in case of total income generating members.

# #### Previous NFRI Distribution history ####

# In[30]:


print(df.iloc[:, 38].value_counts(normalize=True))
df.iloc[:, 38].value_counts(normalize=True).plot(kind="bar")


# Almost 75 percent of the respondents have already received NFRI in the past. That way, we can surmise most of the respondents have experience of previous flood and NFRI distribution. Later, we will compare the NFRI Preferences of respondents across previous experience categories.

# #### NFRI Preferences #####

# First, turning the categorical labels into numeric importance scores below.

# In[31]:


df_clean = dm.nfri_preferences_to_numbers(df)


# #### NFRI Basic ####

# In[32]:


df_preferences_basic = (
    df_clean.loc[:, nlf.nfri_basic].apply(pd.Series.value_counts, normalize=True)
).T


# In[33]:


df_preferences_basic.sort_values(by=[3], inplace=True, ascending=False)
df_preferences_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
df_preferences_basic


# 3 (almost 4) out of 11 items are deemed Essential by more than 80 percent of respondents.

# The Clothing related items are deemed less Essential by the respondents (all less than 50 percent), with Male Dhoti, an attire worn mostly in Terai / Plain region, deemed least Essential of all.

# #### NFRI Non-Basic ####

# In[34]:


df_preferences_non_basic = (
    df_clean.loc[:, nlf.nfri_non_basic].apply(pd.Series.value_counts, normalize=True)
).T


# In[35]:


df_preferences_non_basic.sort_values(by=[3], inplace=True, ascending=False)
df_preferences_non_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
df_preferences_non_basic


# 7 out of 10 items are deemed Essential by more than 80 percent of the respondents.

# The Whistle is the item deemed least essential, with only 20 percent finding the item Essential and almost 33 deeming it Unnecessary.

# #### Correlation Matrix ####

# In[36]:


def magnify():
    return [
        dict(selector="th", props=[("font-size", "7pt")]),
        dict(selector="td", props=[("padding", "0em 0em")]),
        dict(selector="th:hover", props=[("font-size", "12pt")]),
        dict(
            selector="tr:hover td:hover",
            props=[("max-width", "200px"), ("font-size", "12pt")],
        ),
    ]


# In[37]:


df_nfri_basic = df_clean.iloc[:, 39:50]
nfri_basic_corr = df_nfri_basic.corr()

cmap = sns.diverging_palette(5, 250, as_cmap=True)

nfri_basic_corr.style.background_gradient(cmap, axis=1).set_properties(
    **{"max-width": "80px", "font-size": "10pt"}
).set_caption("Hover to magify").set_precision(2).set_table_styles(magnify())


# All clothing related items are highly correlated in terms of preferences, maybe exhibiting same grouping in terms of respondents' perception towards them.

# Interetingly, sacking bag and nylon rope also have relatively high correlation.

# In[38]:


df_nfri_non_basic = df_clean.iloc[:, 50:61]
nfri_non_basic_corr = df_nfri_non_basic.corr()

cmap = sns.diverging_palette(5, 250, as_cmap=True)

nfri_non_basic_corr.style.background_gradient(cmap, axis=1).set_properties(
    **{"max-width": "80px", "font-size": "10pt"}
).set_caption("Hover to magify").set_precision(2).set_table_styles(magnify())


# Here also, we can see the high correlation score between items that exhibt similar grouping. Like Laundry Soap and Bathing Soap being very highly correlated, and Tooth Brush and Paste also being highly correlated to both.

# Similarly, Ladies Underwear and Sanitary Pad are also highly correlated.

# #### Multi Variable Analysis ####

# In[39]:


df_clean.groupby("What.is.the.Ethnicity.of.your.household.")[
    ["How.many.members.of.your.house.contribute.to.the.household.income."]
].mean().sort_values(
    ["How.many.members.of.your.house.contribute.to.the.household.income."],
    ascending=False,
).head()


# In[40]:


df_variables = df_clean.iloc[:, [2, 3, 4, 5, 6, 37]]
variables_corr = df_variables.corr()

cmap = sns.diverging_palette(5, 250, as_cmap=True)

variables_corr.style.background_gradient(cmap, axis=1).set_properties(
    **{"max-width": "80px", "font-size": "10pt"}
).set_caption("Hover to magify").set_precision(2).set_table_styles(magnify())


# #### Gender Wise Exploration ####

# #### First, diving into respondent gender ####

# In[41]:


df_gender_respondent_grouped = (
    df_clean.loc[df_clean["What.is.the.Gender.of.the.respondent."] != "Other"]
    .groupby(
        "What.is.the.Gender.of.the.respondent.",
        as_index=False,
    )
    .apply(np.mean)
)
df_gender_respondent_grouped.loc[:, nlf.nfri_basic]

# Female is 0 and Male is 1 below.


# In[42]:


df_gender_respondent_grouped = (
    df_clean.loc[df_clean["What.is.the.Gender.of.the.respondent."] != "Other"]
    .groupby(
        "What.is.the.Gender.of.the.respondent.",
        as_index=False,
    )
    .apply(np.mean)
)
df_gender_respondent_grouped.loc[:, nlf.nfri_non_basic]

# Female is 0 and Male is 1 below.


# Across all the items (both basic and non-basic), the importance scores are almost consistently same across both the gender respondents. Gender specific items are slighly rated higher by the respective gender, which is very understandable too.

# #### Second, diving into gender ratio (Male/Female) ####

# In[43]:


df_clean["Gender.Ratio"] = df_clean["Total_Male"] / df_clean["Total_Members"]


# In[44]:


df_clean["Gender.Ratio.Label"] = np.where(
    df_clean["Gender.Ratio"] > 0.5,
    "Male_Majority",
    np.where(
        df_clean["Gender.Ratio"] < 0.5,
        "Female_Majority",
        "Balanced",
    ),
)


# In[45]:


df_gender_ratio_grouped = (
    df_clean.loc[df_clean["Gender.Ratio.Label"] != "Balanced"]
    .groupby(
        "Gender.Ratio.Label",
        as_index=False,
    )
    .apply(np.mean)
)
df_gender_ratio_grouped.loc[:, nlf.nfri_basic]

# 0 = Female Majority below


# In[46]:


df_gender_ratio_grouped = (
    df_clean.loc[df_clean["Gender.Ratio.Label"] != "Balanced"]
    .groupby(
        "Gender.Ratio.Label",
        as_index=False,
    )
    .apply(np.mean)
)
df_gender_ratio_grouped.loc[:, nlf.nfri_non_basic]

# 0 = Female Majority below


# Just as with Gender Respondent variable above, all the items (both basic and non-basic), the importance scores are almost consistently same across both the gender respondents. Gender specific items once again are slighly rated higher by the respective gender.

# #### Comparing across previous NFRI experience ####
#

# In[47]:


df_experience_grouped = df_clean.groupby(
    "Has.your.family.received.any.Non.Food.Related.Item.from.the.Red.Cross.in.the.past.",
    as_index=False,
).apply(np.mean)

df_experience_grouped.loc[:, nlf.nfri_basic]


# The Clothing items, which are otherwise also deemed less essential, are deemed less essential by those who have previous NFRI experience. The other items have almost similar scores across both categories.

# In[48]:


df_experience_grouped.loc[:, nlf.nfri_non_basic]


# The Clothing items, which are otherwise also deemed less essential, are deemed less essential by those who have previous NFRI experience. The other items have almost similar scores across both categories.

# #### Diving into extreme observations ####

# Here, we will look into observations that have found even Male Dhoti and Whistle Blow, two items deemed least essential overall as essential.

# #### Male Dhoti i.e Lower #####

# In[49]:


lower_df = df_clean.loc[df_clean["Male.Dhoti"] == 3]


# In[50]:


lower_df["What.is.the.Ethnicity.of.your.household."].value_counts(normalize=True)


# In[51]:


df_clean["What.is.the.Ethnicity.of.your.household."].value_counts(normalize=True)


# Compared to 8 percent representation of total respondents number, Madhesi ethnicity has 25 percent representation of respondents deeming Male Dhoti as highly essential. So, a good remark in terms of validating the quality of survey response.

# In[52]:


lower_preferences_basic = (
    lower_df.loc[:, nlf.nfri_basic].apply(pd.Series.value_counts, normalize=True)
).T
lower_preferences_basic.sort_values(by=[3], inplace=True, ascending=False)
lower_preferences_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
lower_preferences_basic


# In[53]:


lower_preferences_non_basic = (
    lower_df.loc[:, nlf.nfri_non_basic].apply(pd.Series.value_counts, normalize=True)
).T
lower_preferences_non_basic.sort_values(by=[3], inplace=True, ascending=False)
lower_preferences_non_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
lower_preferences_non_basic


# Understandably, those who deem even Male Dhoti essential deem most of other items as essential too.

# #### Whistle Blow ####

# In[54]:


whistle_df = df_clean.loc[df_clean["Whistle.Blow"] == 3]


# In[55]:


whistle_df["What.is.the.Ethnicity.of.your.household."].value_counts(normalize=True)


# In[56]:


whistle_df["What.is.the.Gender.of.the.respondent."].value_counts(normalize=True)


# Interestingly, here also the Madhesi ethnicity is hugely represented (more than 4 times in total data). It will be interesting to compare this preference when we receive Mahottari data.

# In[57]:


whistle_preferences_basic = (
    whistle_df.loc[:, nlf.nfri_basic].apply(pd.Series.value_counts, normalize=True)
).T
whistle_preferences_basic.sort_values(by=[3], inplace=True, ascending=False)
whistle_preferences_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
whistle_preferences_basic


# In[58]:


whistle_preferences_non_basic = (
    whistle_df.loc[:, nlf.nfri_non_basic].apply(pd.Series.value_counts, normalize=True)
).T
whistle_preferences_non_basic.sort_values(by=[3], inplace=True, ascending=False)
whistle_preferences_non_basic.rename(
    columns={1: "Unnecessary", 2: "Desirable", 3: "Essential"}, inplace=True
)
whistle_preferences_non_basic


# As above, those who deem even Whistle Blow essential deem most of other items as essential too.
