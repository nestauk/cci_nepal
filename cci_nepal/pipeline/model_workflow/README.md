# Training and running the models

<br>
<br>

The python scripts in this folder can be used to train and test the NFRI predict models on new data. The first script `model_save.py` fits the x2 models (Shelter and Wash NFRI's) on the whole training set using the best model and parameters found in the model development stage and saves the fitted models to disk. The second script `model_test.py` loads the models and uses them to predict on a new data (the held out test set by default).

### Steps to take before runnning

The below two options depend on if you have access to real survey data or need to generate dummy data to save and run the model.

### OPTION A - With access to the real survey data

**Step 1: Save the file into `inputs/data`**
<br>
When saving your file make sure to save it in `xlsx` format.

**Step 2: Update the file name in config**
<br>
Navigate to `cci_nepal/config` and open the `base.yaml` file. In that file you will see the below `file` variable:

```shell
data:
  file: "dummy_data"
```

Change the value from `dummy_data` to the name of your file.

### OPTION B - Without access to the real survey data

#### Create a dummy dataset

Run the below python file to create and save a dummy dataset that can be used for modelling. This is based on the questions used in our survey.

```shell
$ cd cci_nepal/pipeline
$ python3 dummy_data.py
```

##### Outputs

Running the `dummy_data.py` file saves a dummy version of the data you can use for modelling. The values are assigned randomly from the list of values for each column.

`dummy_data.xlsx`\* saved in `inputs/data`.

\*this is the default file used when you clone the repo. If you change the config `file` variable in option A you just need to remember to change it back to `dummy_data` if you want to re-run the script using your generated dummy data.

### Save and run the models

Perform the following steps to train and run the models:

Split the survey data into training / validation and test sets

```shell
$ cd cci_nepal/pipeline
$ python3 data_splitting_survey.py
```

#### Outputs

There are three files created from running the `data_splitting.py` file. These are saved in `outputs/data/data_for_modelling` and are listed below. These form the training, validation and test sets used for modelling.

- `train.csv`
- `val.csv`
- `test.csv`

Move into the `model_workflow` folder and run the following file to train, save and run the models.

```shell
$ cd cci_nepal/pipeline/model_workflow
$ python3 model_save.py
$ python3 model_test.py
```

### Final Outputs

There are four files created from running the models and saved to `outputs/data/test_evaluation_results`:

- `shelter_test_predictions.xlsx`
- `wash_test_predictions.xlsx`
- `shelter_test_evaluation.xlsx`
- `wash_test_evaluation.xlsx`

These contain the survey predictions and evaluation metrics for each shelter and wash/dignity NFRI items respectively. For the prediction files, the first set of columns will contain the feature names and the next set will contain the NFRI items with a 0 to 1 probability as to whether they are the item is essential.
