# Training and running the models

The python scripts in this folder can be used to train and test the NFRI predict models on new data. The first script `model_save.py` fits the two models (Shelter and Wash NFRI's) on the whole training set using the best model and parameters found in the model development stage and saves the fitted models to disk. The second script `model_test.py` loads the models and uses them to predict on a new data (the held out test set by default). 

### Save and run the models

*Prior to running the scripts in this folder you will need to follow the steps in the [main readme](https://github.com/nestauk/cci_nepal/tree/15_model_pipeline#installation) to install the repository and access / save the survey data.*

Perform the following steps to train, save and run the models on a held out test set. 

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
