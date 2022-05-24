# Training and running the models

<br>
<br>

The python scripts in this folder can be used to train and run the NFRI predict models on new data. The first script `model_save.py` fits the x2 models (basic and non-basic NFRI's) on the whole training set using the best model and parameters found in the model development stage and saves the fitted models to disk. The second script `model_run.py` loads the models and uses them to predict on a new data (the held out test set by default).

Before feeding data into the models, a few different pre-processing and cleaning steps are taken on the data to make sure it is in the right format. These steps are all held in functions saved in `data_manipulation.py` and `model_tuning_report.py` scripts found in `pipeline/classfication_model/`.

### Steps to take before runnning

To run the models you will first need to setup the project. Follow the below two steps to do this:

1. Clone the project and cd into the `cci_nepal` directory
2. Run the command `make install` to create the virtual environment and install dependencies
3. Inside the project directory run `make inputs-pull` to access the data from S3 (for those with access to the Nesta S3 account)

To note the project is setup using the Nesta Cookiecutter (guidelines on the Nesta Cookiecutter can be [found here](https://nestauk.github.io/ds-cookiecutter/structure/)).

### Input needed

After you setup the project you will need your training and test datasets. To build our models we used new data collected by the Nepal Red Cross in two districts - Sindhupalchok and Mahottari. The raw data from these surveys are saved in `cci_nepal/inputs/data/real_data/Full_Data_District.csv`. Running the script `data_splitting_survey.py` in `pipeline/classfication_model/` produces the training and test dataset.

To run the scripts the data needs to be in the same format as the survey data collected and saved in `cci_nepal/outputs/data/data_for_modelling/`.

### Run the scripts

Perform the following steps to run the scripts:

- `cd` to inside this folder
- run `python3 model_save.py`
- run `python3 model_run.py`

### Outputs

There are two files created from running the models and saved to outputs:

- `basic_test_predictions.xlsx`
- `non_basic_test_predictions.xlsx`

These contain the survey inputs and predictions for each basic and non-basic NFRI items respectively. The format of each file will be slighlty different as different numbers of features are used and the NFRI outputs are different. The first set of columns will contain the feature names and the next set will contain the NFRI items with a 0 to 1 probability as to whether they are the item is essential.
