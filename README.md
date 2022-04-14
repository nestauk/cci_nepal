# cci_nepal

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt` and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>

### NFRI Predict Classification Model

The following scripts are related to the Classification Model trained on the NFRI Survey dataset.

#### Introduction

Our dataset comes from a survey conducted with the goal of identifying how essential a certain NFRI (Non Food Related Item) is for a particular household given the household features (like demographics and geographic location.)

#### Dataset

Our dataset consists 2338 observations and 73 features.

The features can be divided into input and output features, with input features related to various demograhics and geography and output features related to preference labels (Essential, Desirable and Non Essential) given by the respondents for each NFRI.

The dataset is further divided into train, validation and test set.

#### Classification Model

Our goal is to calculate the probability of each NFRI being Essential given its features. For the purpose of focusing on Essential, we have reduced the Desirable and Non Essential into one single category, making our problem a binary classification problem.

Since we want to predict the probability of each NFRI, we have multiple output too, making our model a multi-output classification problem too.

Lastly, as we have two different sets of NFRI (Shelter related and Hygine related, both of which are distributed in separate packages), we will have two separate classification model for each NFRI type.

In short, we will have two separate 'Multi Output Binary Classification' models.

#### Scripts: The story of 3 halves

We have a total of four scripts related to the Classification model, which can be better understood as a code flow of 3 halves.

**First Half**: feature_selection_model_tuning_classification.py

In this script, we perform feature selection, model tuning and testing of various classification algorithms (using Grid Search Cross Validation) and save the results (features selected and evaluation metrics) of each model.

**Second Half**: feature_selection_model_tuning_logistic_regression.py

Before running this script, we first select the 'best' model from the various models saved in the first script. While selecting the best model, other factors are also weighed in (like explainability of model) along with machine learning related evaluation features (like accuracy and different F1 scores.)

We then load the selected model in this script, and then fit the model using the chosen parameters from the earlier script. We then first fit and then save the model.

**Third Half**: classification_model_run.py

In this script, we first read the fitted model (saved in the second script) and evaluate it using the test dataset.

**Additional Half**: feature_selection_model_tuning_metrics.py

Just to make the flow of code easier, we have one more script where we code for several model tuning metrics.

**Pipeline Functions**: model_tuning_report.py

There is one more script in the pipeline folder named model_tuning_report.py where we have several functions necessary for the running of earlier four scripts. (Functions like testing several models, calculating accuracy/f1 scores for each NFRI, etc.)

#### Code Flow: To run the code

Just to evalulate the final chosen model, one can just run the classification_model_run.py script, as the final chosen model is already stored in the outputs directory.

Whereas, if one were to start from the scratch, one should first run the feature_selection_model_tuning_classification.py script, followed by feature_selection_model_tuning_logistic_regression.py script and finally classification_model_run.py script.
