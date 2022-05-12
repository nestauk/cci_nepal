# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: cci_nepal
#     language: python
#     name: cci_nepal
# ---

# %%
# Import libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Project libraries
import cci_nepal

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# %%
# basic_results_train = pd.read_pickle(
#    f"{project_dir}/outputs/data/model_results/features_params_scores_basic.pkl"
# )
# non_basic_results_train = pd.read_pickle(
# f"{project_dir}/outputs/data/model_results/features_params_scores_non_basic.pkl"
# )

# %%
basic_results_train = pd.read_pickle(
    f"{project_dir}/outputs/data/model_results/features_params_scores_basic_filtered_features.pkl"
)
non_basic_results_train = pd.read_pickle(
    f"{project_dir}/outputs/data/model_results/features_params_scores_non_basic_filtered_features.pkl"
)

# %% [markdown]
# ### Basic

# %%
# %%

# n_features_use = [17, 17, 2, 2, 2, 17]

# n_features_use = [5, 5, 5, 5, 2, 5]

# %%
models = []
f1_scores = []
for model in basic_results_train:
    models.append(model)
    score = basic_results_train[model][0]
    f1_scores.append(score)

# %%
# %matplotlib inline
plt.style.use("ggplot")

x_pos = [i for i, _ in enumerate(models)]
plt.ylim(ymin=0.8, ymax=0.9)

plt.bar(x_pos, f1_scores, color="green")
plt.xlabel("Models")
plt.ylabel("F1 scores")
plt.title(
    "Basic model: F1 scores for each model tested (with optimum feature number)", pad=20
)

plt.xticks(x_pos, models)

plt.show()

# %%
# %matplotlib inline
# plt.style.use("ggplot")

# x_pos = [i for i, _ in enumerate(models)]

# plt.bar(x_pos, color="green")
# plt.xlabel("Models")
# plt.ylabel("Number of features")
# plt.title("Basic model: Optimum number of features to use per model", pad=20)

# plt.xticks(x_pos, models)

# plt.show()

# %% [markdown]
# ### Non-basic

# %%
# %%
# n_features_use = [5, 5, 5, 2, 2, 5]

# %%
models = []
f1_scores = []
for model in non_basic_results_train:
    models.append(model)
    score = non_basic_results_train[model][0]
    f1_scores.append(score)

# %%
# %matplotlib inline
plt.style.use("ggplot")

x_pos = [i for i, _ in enumerate(models)]
plt.ylim(ymin=0.9, ymax=0.91)

plt.bar(x_pos, f1_scores, color="green")
plt.xlabel("Models")
plt.ylabel("F1 scores")
plt.title(
    "Non-basic model: F1 scores for each model tested (with optimum feature number)",
    pad=20,
)

plt.xticks(x_pos, models)

plt.show()

# %%
# %matplotlib inline
# plt.style.use("ggplot")

# x_pos = [i for i, _ in enumerate(models)]

# plt.bar(x_pos, color="green")
# plt.xlabel("Models")
# plt.ylabel("Number of features")
# plt.title("Non-basic model: Optimum number of features to use per model", pad=20)

# plt.xticks(x_pos, models)

# plt.show()

# %%
