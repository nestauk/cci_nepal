<!-- #region -->

# Collective Crisis Intelligence Project for The Nepal Red Cross

**_Public repository for hosting the technical outputs of the CCI Nepal project._**

## Welcome!

This repository contains the code and documentation for a project uses Collective Crisis Intellegence (CCI) to help the Nepal Red Cross find the optimum Non-food-related-items (NFRI) for different households.

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

In line with these needs, the project uses new data collected from surveying two districts in Nepal - Sindhupalchok and Mahottari to:

1. Measure the extent to which different households perceive existing items as essential (also referred to as item “essentialness” henceforth), and
2. Collect and summarise suggestions on new NFRI items to include in packages in case of a flood crisis.

The analysis provided in this repository generates two outputs:

1. A model that predicts item essentialness based on provided household information
2. A stand alone piece of analysis showing what new items are suggested by households across different demographic features

Find out more about the project in our report (coming soon).

## Contents

`Published on July xx, 2022`

- [**Model workflow**](https://github.com/nestauk/cci_nepal/tree/15_model_pipeline/cci_nepal/pipeline/model_workflow): Python code for train the models and then running on the test set.
- [**NFRI suggestion analysis**](https://github.com/nestauk/cci_nepal/tree/15_model_pipeline/cci_nepal/analysis/free_text_analysis): Analysis of the free text survey responses asking participants for suggestions of new NFRI items.

## Data

This project uses new survey data collected from two districts in Nepal - Sindhupalchok and Mahottari. These districts were selected as they capture information from both the Hill and Plain regions in Nepal. Both districts are among the most flood affected districts of Nepal and have received NFRI from the NRC in the past. To design the survey we worked with members of the Red Cross and IFRC teams to develop the questions and ensure they were accessible.

3,265 responses were collected with a 50/50 split between male and female respondents, but otherwise using random sampling from households within the regions.

<b>The survey was split into three sections</b>

1. <b>Demographic information about the household</b>: The first section asks the respondent a series of questions to understand the demographic characteristics of their household. In addition this section also asks for the households lat/long location and if they have previously received NFRI items.
2. <b>NFRI preferences of the household in a flood crisis</b>: The second section asks the respondent to imagine a new flood crisis and asks them to state how important they see each NFRI item already distributed by the Red Cross.
3. <b>New NFRI items</b>: The final section asks the respondent to suggest new NFRI items their household might need in a flood crisis that are not listed in section 2.

The first two sections are used for modelling where the first provides the features and the second provides our target (output) variables. The last section is not used for modelling but for a one-off piece of analysis for NRCS to help them understand new items that different groups might need.

### Data dictionary of final features

The below table depicts the final features used by the model with their data type and a brief description.

| Column name        | Description                                                                                                                                                                            | Type  |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---- |
| household_size     | Count of total members in a household across all age groups.                                                                                                                           | int   |
| percent_non_male   | Percentage of total non male members in a household.                                                                                                                                   | float |
| children_under_5   | A binary variable that represents if a household has any member less than age 5 or not. 1 representing yes and 0 representing no.                                                      | int   |
| income_gen_ratio   | Ratio of total income generating members in a household.                                                                                                                               | float |
| health_difficulty  | A binary variable that represents if a household has any members with a heath difficulty.                                                                                                                           | int   |
| sindupalchowk      | A binary variable that represents if the district of the household is Sindupalchowk (represented as 1) or Mahottari (represented as 0), two districts that are present in the dataset. | int   |
| household_material | A categorical variable representing the house material (Wooden pillar, RCC pillar, Bricks and stone, etc. )                                                                            | str   |

## Installation

Unfortunately the dataset is not publicly available due to it being a survey collected directly by the Red Cross and it containing sensitive data (more information here).

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

The below two options depend on if you have access to real survey data or need to generate dummy data to save and run the model.

### OPTION A - With access to the real survey data

**Step 1: Save the file into `inputs/data/`**
<br>
When saving your file make sure to save it in `xlsx` format.

**Step 2: Update the file name in config**
<br>
Navigate to `cci_nepal/config` and open the `base.yaml` file. In that file you will see the below `file` variable:

```yaml
data:
  file: "dummy_data"
```

Change the value from `dummy_data` to the name of your file.

### OPTION B - Without access to the real survey data

#### Create a dummy dataset

Run the below python file to create and save a dummy dataset that can be used for modelling. This is based on the questions used in our survey.

```shell
python cci_nepal/pipeline/dummy_data.py
```

##### Outputs

Running the `dummy_data.py` file saves a dummy version of the data you can use for modelling. The values are assigned randomly from the list of values for each column.

`dummy_data.xlsx`\* saved in `inputs/data`.

\*this is the default file used when you clone the repo. If you change the config `file` variable in option A you just need to remember to change it back to `dummy_data` if you want to re-run the script using your generated dummy data.

### Save and run the models

Perform the following steps to train and run the models:

Split the survey data into training / validation and test sets

```shell
python  cci_nepal/pipeline/data_splitting_survey.py
```

#### Outputs

There are three files created from running the `data_splitting.py` file. These are saved in `outputs/data/data_for_modelling` and are listed below. These form the training, validation and test sets used for modelling.

- `train.csv`
- `val.csv`
- `test.csv`

Run the following modules to train, save and run the models.

```shell
python cci_nepal/pipeline/model_workflow/model_save.py
python cci_nepal/pipeline/model_workflow/model_test.py

### Final Outputs

There are four files created from running the models and saved to `outputs/data/test_evaluation_results`:

- `shelter_test_predictions.csv`
- `wash_test_predictions.csv`
- `shelter_test_evaluation.csv`
- `wash_test_evaluation.csv`

These contain the survey predictions and evaluation metrics for each shelter and wash/dignity NFRI items respectively. For the prediction files, the first set of columns will contain the feature names and the next set will contain the NFRI items with a 0 to 1 probability as to whether they are the item is predicted as essential.

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

<!-- #endregion -->
