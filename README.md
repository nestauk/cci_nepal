# Collective Crisis Intelligence Project for The Nepal Red Cross

**_Public repository for hosting the technical outputs of the CCI Nepal project._**

## Welcome!

This repository contains the code and documentation for a collective crisis intellegence project with the Nepal Red Cross that looks at finding the optimum Non-food-related-items (NFRI) for different households.

### Background

After a crisis strikes, people can be left without important supplies that support them to stay safe, healthy and comfortable. As part of coordinated efforts with other governmental and non-governmental organisations, the Nepal Red Cross Society (NRCS) provides Non-Food Related Items (NFRI) packages to affected communities.

A typical family NFRI package includes:

- Tarpaulin
- Blanket
- Sari, Male Dhoti
- Shouting cloth, printed cloth, plain cloth, teri cotten cloth
- Utensil set
- Water bucket
- Rope

These packages are often distributed in the aftermath of a crisis based on an Initial Rapid Assessment. During the project scoping phase, we interviewed NRCS team members to assess their needs and found that the team was interested in:

- Identifying a better way of knowing what NFRI items to distribute and when
- Exploring what new NFRI items community members are interested in

In line with these needs, the project uses new data collected from surveying two districts in Nepal - Sindhupalchok and Mahottari.

<b>Project aims:</b>

1. Measure the extent to which different households perceive existing items as essential (also referred to as item “essentialness” henceforth), and
2. Collect and summarise suggestions on new NFRI items to include in packages in case of a flood crisis.

The analysis provided in this repository generates two outputs:

1. A model that predicts item essentialness based on provided household information
2. A stand alone piece of analysis showing what new items are suggested by households across different demographic features

Find out more about the project in our report 'title of the report' [insert link].

## Contents

`Published on July xx, 2022`

- [**Model workflow**](https://github.com/nestauk/cci_nepal/tree/15_model_pipeline/cci_nepal/pipeline/classification_model/model_save_run): Python code for train the models and then running on the test set.
- [**Free text analysis**](https://github.com/nestauk/cci_nepal/tree/15_model_pipeline/cci_nepal/analysis): Analysis of the survey questions asking participants for suggestions of new NFRI items.

## Data

This project uses new survey data collected from two districts in Nepal - Sindhupalchok and Mahottari. These districts were selected as they capture information from both the Hill and Plain regions in Nepal. Both districts are among the most flood affected districts of Nepal and have received NFRI from the NRC in the past. To design the survey we worked with members of the Red Cross and IFRC teams to develop the questions and ensure they were accessible.

3,265 responses were collected from a 50/50 split across male and female respondents. Random sampling was then used for the respondents collected within each gender.

<b>The survey was split into three sections</b>

1. <b>Demographic information about the household</b>: The first section asks the respondent a series of questions to understand the demographic characteristics of their household. In addition this section also asks for the households lat/long location and if they have previously received NFRI items.
2. <b>NFRI preferences of the household in a flood crisis</b>: The second section asks the respondent to imagine a new flood crisis and asks them to state how important they see each NFRI item already distributed by the Red Cross.
3. <b>New NFRI items</b>: The final section asks the respondent to suggest new NFRI items their household might need in a flood crisis that are not listed in section 2.

The first two sections are used for modelling where the first is our X features and the second our Y output variables. This last section is not used for modelling but for a one-off piece of analysis for the Red Cross to help them understand what new items are requested by different groups.

### Data dictionary of final features

The below table depicts the final features used by the model with their data type and a brief description.

| Column name        | Description | Type  |
| :----------------- | :---------- | :---- |
| household_size     | Description | int   |
| percent_non_male   | Description | float |
| children_under_5   | Description | int   |
| income_gen_ratio   | Description | float |
| health_difficulty  | Description | int   |
| sindupalchowk      | Description | int   |
| household_material | Description | str   |

## Installation

Unfortunately the dataset is not publicly available due to it being a survey collected directly by the Red Cross (more information here).

However, we have created a script that generates dummy data that allows you to test the running of `model workflow`. Follow the steps below to generate the dummy data and run the code.

### Clone and set up the repo

1. To run the models you will first need to setup the project. Follow the below two steps to do this:

```shell
$ git clone https://github.com/nestauk/cci_nepal
$ cd cci_nepal
```

2. Run the command `make install` to create the virtual environment and install dependencies

3. Inside the project directory run `make inputs-pull` to access the data from S3 (for those with access to the Nesta S3 account)

To note the project is setup using the Nesta Cookiecutter (guidelines on the Nesta Cookiecutter can be [found here](https://nestauk.github.io/ds-cookiecutter/structure/)).

### Create a dummy dataset

Run the below python file to create and save a dummy dataset that can be used for modelling. This is based on the questions used in our survey.

```shell
$ cd cci_nepal/pipeline
$ python3 dummy_data.py
```

#### Outputs

Running the `dummy_data.py` file saves a dummy version of the data you can use for modelling.

`survey_data.py` saved in `inputs/data`.

### Save and run the models

Perform the following steps to train and run the models:

Split the survey data into training / validation and test sets

```shell
$ cd cci_nepal/pipeline
$ python3 data_splitting.py
```

#### Outputs

There are six files created from running the `data_splitting.py` file. These are saved in `outputs/data/data_for_modelling` and are listed below. These form the training, validation and test sets used for modelling.

Training sets

- `train.csv`
- `train_hill.csv`
- `train_terai.csv`

Test sets

- `val.csv`
- `test_hill.csv`
- `test_terai.csv`

Move into the `model_workflow` folder and run the following file to train, save and run the models.

```shell
$ cd cci_nepal/pipeline/model_run
$ python3 model_save.py
$ python3 model_run.py
```

### Final Outputs

There are two files created from running the models and saved to outputs:

- `shelter_test_predictions.xlsx`
- `dignity_test_predictions.xlsx`

These contain the survey inputs and predictions for each basic and non-basic NFRI items respectively. The format of each file will be slighlty different as different numbers of features are used and the NFRI outputs are different. The first set of columns will contain the feature names and the next set will contain the NFRI items with a 0 to 1 probability as to whether they are the item is essential.

## Directory structure

The repository has the following main directories:

```
  ├── cci_nepal                       <- Packaged code (various modules, utilities and readme)
  │   ├── analysis
  │   │   ├── model_development       <- Model tuning on the training set to find optimum models and parameters
  │   │   ├── model_reporting         <- Scripts to collect / report the results on the test set
  │   │   ├── data_analysis           <- Exploratory data analysis of the survey data
  │   │   ├── free_text_analysis      <- Analysis of the text questions of suggestions for new items
  │   │   ...
  │   ├── config                      <- Holds variables, feature names and parameters used in the codebase
  │   ├── getters                     <- Functions for getting the data
  │   ├── pipeline                    <- Holds scripts for all pipeline components
  │   │   └── model_workflow          <- Fits and saves the model using training data and runs it on test data
  │   ├── utils                       <- Utility functions needed across different parts of the codebase
  │   ...
  ├── inputs
  │   └── data                        <- Holds original survey data (or dummy data)
  │   ...
  └── outputs
      ├── data
      │   └── data_for_modelling      <- Training, validation and test sets saved here
      ├── models                      <- Saved models after running model_workflow
      ...

```
