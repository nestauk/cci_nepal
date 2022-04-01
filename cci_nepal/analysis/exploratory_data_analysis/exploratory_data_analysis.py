#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import libraries
import logging
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

# Project libraries
import cci_nepal
from cci_nepal.getters.exploratory_data_analysis import get_train_data as gtd
from cci_nepal.pipeline.exploratory_data_analysis import data_manipulation as dm

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR


# The following analysis is of the NFRI Survey dataset collected across two districts (Sindhupalchowk and Mahottari) of Nepal. The dataset consists of 2338 rows (observations) and 73 columns (variables) in total.
#
# The variables consist of demographic and geographic information at household level and their preference of NFRI items.
#
# Keeping in mind our project goal and further steps (like Modeling), the variables can be further divided into input and output variables. The demographic, geographic and other related variables of households can be treated as input variables and their NFRI preference as output variables.

# The analysis will be divided into four parts:
# - First Part: Data Pre-Processing
# - Second Part: Analysis of input variables / features.
# - Third Part: Analysis of output variables (i.e NFRI Preference)
# - Fourth Part: Analysis of output variables (NFRI Preferences) across different input variables.

# In[2]:

# #### Read the dataset ####

# In[3]:


df = gtd.read_train_data()


# In[4]:

# Read column names to make existing column names more interpretable
column_names = gtd.get_lists(f"{project_dir}/cci_nepal/config/column_names.csv")

# More interpretable column names for analysis


# ### First Part: Data Pre-Processing ###

# - Removal of non-feature columns (like notes, introduction, etc.)
# - Renaming of columns to make them more interpretable.
# - Replacement of null values with 0 (as answers with 0 are left as null in our survey.)
# - Addition of new features (like total male, total female, total children, etc.)
#
# nB: The train version of data already has all null values mapped as zero, as the survey had option to leave 0 answers as null.

# In[5]:


columns_to_drop = [5, 16, 29, 39, 41, 42, 54, 55, 67, 68, 69, 70, 71]
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
df.columns = column_names
df.fillna(0, inplace=True)
df = df.applymap(lambda s: s.lower() if type(s) == str else s)
df.insert(5, "Total_Male", df.iloc[:, 5:15].sum(axis=1))
df.insert(6, "Total_Female", df.iloc[:, 16:26].sum(axis=1))
df.insert(7, "Total_Children", df.iloc[:, [7, 8, 9, 17, 18, 19]].sum(axis=1))
df.insert(8, "Total_Health_Difficulty", df.iloc[:, 30:36].sum(axis=1))


# #### Dividing NFRI items into Basic and Non Basic items ####

# Since the NFRI items belong to two different package categories (Shelter package and Wash and Dignity package), we will divide the items into Basic (meaning Shelter) and Non-Basic (Wash and Dignity) categories for the ease for analysis.

# In[6]:


nfri_items = column_names[37:]
basic = nfri_items[0:11]
non_basic = nfri_items[11:]


# ### Second Part: Analysis of Input Variables ###

# #### Respondent Gender ####

# In[7]:


ax = df.loc[:, "Respondent_Gender"].value_counts(normalize=True).plot.bar()
ax.bar_label(ax.containers[0])


# As we see, the percentage of Respondent Gender is almost balanced, which was our plan before the survey also.

# #### Respondent Age Breakdown ####

# In[8]:


ax = df.loc[:, "Respondent_Age"].value_counts(normalize=True).plot(kind="bar")
ax.bar_label(ax.containers[0])


# Half of the Respondent Age come from 30-39 and 40-49 age group, the normal household head size in Nepal.

# #### Total Male ####

# In[9]:


df["Total_Male"].describe()


# In[10]:


ax = df.loc[:, "Total_Male"].value_counts(normalize=True).plot(kind="bar")


# #### Total Female ####

# In[11]:


df["Total_Female"].describe()


# In[12]:


df.loc[:, "Total_Female"].value_counts(normalize=True).plot(kind="bar")


# #### Total Members ####

# In[13]:


df["Total_Members"] = df["Total_Male"] + df["Total_Female"]
df["Total_Members"].describe()


# In[14]:


df.loc[:, "Total_Members"].value_counts(normalize=True).plot(kind="bar")


# While the mean household size is 5.66 (with 5 being the median), the household size 5 and 4 are both almost equally present (both having 19 percent representation out of total).

# #### Total Children ####

# In[15]:


df.loc[:, "Total_Children"].value_counts(normalize=True).plot(kind="bar")


# Children size of 2 (at 25 percent) is the most frequent for the dataset.
#
# Interesting, more than 20 percent of the households have no children!

# #### Health Difficulty ####

# In[16]:


ax = df.loc[:, "Total_Health_Difficulty"].value_counts(normalize=True).plot(kind="bar")
ax.bar_label(ax.containers[0])


# Nearly 70 percent of the households have no one with health difficulty.
#
# Now onto exploring which type of health difficulty is the most prevalent one.

# #### Most Common Health Difficulty ####

# In[17]:


ax = (
    pd.DataFrame(df.iloc[:, 31:37].apply(np.sum))
    .sort_values(by=0, ascending=False)
    .plot(kind="bar")
)
ax.bar_label(ax.containers[0])


# As we can see, the most prevalent health difficulty is Difficulty in Seeing, followed by Difficulty in Walking and Hearing.

# #### Ethnicity ####

# In[18]:


pd.DataFrame([df.loc[:, "Ethnicity"].value_counts(normalize=True)]).T


# In[19]:


ax = df.loc[:, "Ethnicity"].value_counts(normalize=True).plot(kind="bar")
ax.bar_label(ax.containers[0])


# As we see, more than 95 percents of observations come from four majour ethnicities in the option.

# All four major ethnicities almost evenly distributed, with this being highest at 28 and this being lowest at 19.

# #### House Material ####

# In[20]:


ax = df.loc[:, "House_Material"].value_counts(normalize=True).plot(kind="bar")
ax.bar_label(ax.containers[0])


# As we can see, the "other" category is significantly represented in the dataset (almost 20 percent of total observations. So, we will delve deeper to see what answers constitute the "other" category.

# In[21]:


df_other = df[df["House_Material"] == "other"]
df_other.loc[:, "Material_Other"].value_counts(normalize=True).plot(kind="bar")


# As we can see, almost 2/3rd (65 percent) of the other categories have "mato ghar" as the answer, meaning clay house, followed by clay again at 6 percent. The other answers are also mostly clay related terms. Thus, the "other" category mostly refers to the "Clay" as house materials.

# In the third part of analysis, we will delve deeper into the individual categories and see if the "other" category differ from others or not.

# #### Total Income Generating Members ####

# In[22]:


ax = (
    df.loc[:, "Income_Generating_Members"].value_counts(normalize=True).plot(kind="bar")
)
ax.bar_label(ax.containers[0])


# As we see, almost 60 percent of the households have just 1 income generating member, with 26 percent having two income generating members.

# #### Previous NFRI History ####

# In[23]:


ax = df.loc[:, "Previous_NFRI"].value_counts(normalize=True).plot(kind="bar")
ax.bar_label(ax.containers[0])


# As we see, the percentage of households that have not previously reveived NFRI is also significantly higher at 43 percent.
# (This goes with our plan of having both sets of households fairly represented in dataset.)

# ### Third Part: NFRI Preferences ###

# #### Basic ####

# In[24]:


df_preference_labels_basic = (
    df.loc[:, basic].apply(pd.Series.value_counts, normalize=True).T
)
df_preference_labels_basic = df_preference_labels_basic.reindex(
    ["essential", "desirable", "unnecessary"], axis=1
).sort_values(by="essential", ascending=False)
ax = sns.heatmap(df_preference_labels_basic, annot=True)


# 9 out of 11 being deemed esssential by more than 50 percent, 3 as essential by more than 90 percent, and Male Dhoti interesting being called essential by almost 48 percent and unnecessary by 35 percent!

# #### Non Basic ####

# In[25]:


df_preference_labels_non_basic = (
    df.loc[:, non_basic].apply(pd.Series.value_counts, normalize=True).T
)
df_preference_labels_non_basic = df_preference_labels_non_basic.reindex(
    ["essential", "desirable", "unnecessary"], axis=1
).sort_values(by="essential", ascending=False)
ax = sns.heatmap(df_preference_labels_non_basic, annot=True)


# 10 out of 11 items being deemed essential by more than 50 percent, 8 out of 11 items deemed essential by more than 80 percent and 3 items above 90 percent.
#
# Whistle Blow contrasting deemed essential by only 34 percent.

# #### Analysing NFRI Preferences as Numeric Scores ####

# In[26]:


df_numeric = dm.nfri_preferences_to_numbers(df)


# #### NFRI Basic ####

# In[27]:


df_numeric.loc[:, basic].agg(["mean", np.std])


# #### NFRI Non Basic ####

# In[28]:


df_numeric.loc[:, non_basic].agg(["mean", np.std])


# #### Correlation Matrix ####

# In[29]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(
    df_numeric.loc[:, basic].corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG"
)
heatmap.set_title("Correlation Heatmap for Basic", fontdict={"fontsize": 18}, pad=12)


# All clothing related items are highly correlated in terms of preferences, maybe exhibiting same grouping in terms of respondents' perception towards them.
#
# Interetingly, sacking bag and nylon rope also have relatively high correlation. Same for Water Bucket and Utensil Set.

# In[30]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(
    df_numeric.loc[:, non_basic].corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG"
)
heatmap.set_title(
    "Correlation Heatmap for Non Basic", fontdict={"fontsize": 18}, pad=12
)


# Here also, we can see the high correlation score between items that exhibt similar grouping. Like Laundry Soap and Bathing Soap being very highly correlated, and Tooth Brush and Paste also being highly correlated to both.
#
# Similarly, Ladies Underwear and Sanitary Pad are also highly correlated. Same can be observed for Hand Sanitizer and Liquid Chlorine.

# ### Fourth Part: Analysing NFRI Preferences across different Input Variables ###

# #### NFRI Preferences Respondent Gender Wise ####

# #### NFRI Basic ####

# In[31]:


df_numeric.groupby("Respondent_Gender", as_index=True).apply(np.mean).loc[:, basic]


# #### NFRI Non Basic ####

# In[32]:


df_numeric.groupby("Respondent_Gender", as_index=True).apply(np.mean).loc[:, non_basic]


# As we can see, for both basic and non basic, the NFRI preference scores are almost identical for all the items. That way, the preference scores of items are similar regardless of respondent gender.

# #### NFRI Preferences Gender Ratio Wise ####

# To see if there could be change in preferences scores for households with different gender ration, we will create a new variable called Gender Ratio and compare for households with less than and more than 50 percent male. (Households with equal ratio are omitted for this analysis.)

# In[33]:


df_numeric["Gender_Ratio"] = df_numeric["Total_Male"] / df_numeric["Total_Members"]
df_numeric["Gender_Ratio_Label"] = np.where(
    df_numeric["Gender_Ratio"] > 0.5,
    "Male_Majority",
    np.where(
        df_numeric["Gender_Ratio"] < 0.5,
        "Female_Majority",
        "Balanced",
    ),
)


# In[34]:


df_numeric.loc[df_numeric["Gender_Ratio_Label"] != "Balanced"].groupby(
    "Gender_Ratio_Label",
    as_index=True,
).apply(np.mean).loc[:, basic]


# Seriously man, so so so identical! Even for Sari the score for both is very similar (in fact slighly higher for Male Majority).

# In[35]:


df_numeric.loc[df_numeric["Gender_Ratio_Label"] != "Balanced"].groupby(
    "Gender_Ratio_Label",
    as_index=True,
).apply(np.mean).loc[:, non_basic]


# The scores once again are almost identical for both gender majority households. Here the female related items (Sanitary Pad and Ladies Underwear) have slighly higher scores for Female Majority households.

# #### NFRI Preferences Ethnicity Wise ####

# In[36]:


df_numeric.groupby("Ethnicity")[basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]  # Comparing across the four major ethnicities only as they make up more than 95 percent


# In[56]:


df_numeric.groupby("Ethnicity", as_index=False, sort=True)[
    "Income_Generating_Members"
].apply(lambda x: x.astype(int).mean()).iloc[0:4]


# In[39]:


df_numeric.groupby("Ethnicity")[non_basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]


# Looks like the scores are almost identical for most of the items. And for the ones that are different, the scores are consistently higher for Dalit and Madhesi ethnicity.

# #### NFRI Preferences House Material wise ####

# In[40]:


df_numeric.groupby("House_Material")[basic].apply(lambda x: x.astype(int).mean()).iloc[
    0:4,
]


# In[41]:


df_numeric.groupby("House_Material")[non_basic].apply(
    lambda x: x.astype(int).mean()
).iloc[
    0:4,
]


# As we see, the category other has signicantly higher score for both basis and non-basic, specially for items that have lower scores for other categories. (Like for Sari, Male Dhoti, Clothing items, Whistle Blow.)

# Since more than 70 percent of Other category is made up of Clay related words, we will analyse for observations where Clay is the household material, just to see how the NFRI Preference scores vary for Clay houses.

# In[119]:


df_clay = df_numeric[
    df_numeric["Material_Other"].str.contains("Mato ghar|clay", case=False)
]


# In[115]:


df_clay[basic].apply(lambda x: x.astype(int).mean())


# In[116]:


df_clay[non_basic].apply(lambda x: x.astype(int).mean())


# As we see, for the obsevations with Clay as house material (almost 15 percent of the total observation), the NFRI scores are siginificantly higher for almost all the items. This highlights that the households with house material as clay rate most items as essential.

# #### Delving deeper into House Material and Ethnicity : Two influential features ####

# Since NFRI score was found to be higher for Madhesi and Dalit ethnicity, and also house material category "Other" (most of which is Clay), we will see how these two variables are related to each other.

# In[125]:


pd.DataFrame(
    df_numeric.groupby("Ethnicity", as_index=True, sort=True)["House_Material"].apply(
        lambda x: x.value_counts(normalize=True)
    )
).iloc[0:-8:]


# As we can see, the "Other" house material is significantly present in only two Ethnicities Dalit and Madhesi, both ethnicities that had significantly higher importance scores than others.
#
# Specially for Dalit ethnicity, the Other category is the most frequent category, making almost 50 percent.

# #### NFRI Preferences previous NFRI history wise ####

# In[42]:


df_numeric.groupby("Previous_NFRI", as_index=True,).apply(
    np.mean
).loc[:, basic]


# In[43]:


df_numeric.groupby("Previous_NFRI", as_index=True,).apply(
    np.mean
).loc[:, non_basic]


# As we can see, for most of the items in basic, the preference score is higher for No category (households that haven't received NFRI in past.) As for non basic, the scores are fairly similar except for Whistle Blow, which has also higher score for No category.

logging.info("And this is where our analysis ends .... for now!")
