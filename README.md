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

## Exploratory Data Analysis

- The following report contains the exploratory data analysis of NFRI Survey data.

#### Introduction

- The following analysis is of the NFRI Survey dataset collected across two districts (Sindhupalchowk and Mahottari) of Nepal. The dataset consists of 2338 rows (observations) and 73 columns (variables) in total.

- The variables consist of demographic and geographic information at household level and their preference of NFRI items.

- Keeping in mind our project goal and further steps (like Modeling), the variables can be further divided into input and output variables. The demographic, geographic and other related variables of households can be treated as input variables and their NFRI preference as output variables.

#### Code Flow: The Story of four parts

- The analysis will be divided into four parts:

  - First Part: Data Pre-Processing
  - Second Part: Analysis of input variables / features.
  - Third Part: Analysis of output variables (i.e NFRI Preference)
  - Fourth Part: Analysis of output variables (NFRI Preferences) across different input variables.

#### To Run the Code

- Please run the exploratory_data_analysis.py file stored in the exploratory_data_analysis folder inside the analysis folder.
- File path: cci_nepal -> analysis -> exploratory_data_analysis -> exploratory_data_analysis.py
- The necessary functions to run the python file above are stored in separate python files in getters and pipeline folders respectively.

#### Final Output

- The final output of the repository is a python script heavy in summarisation and visualisation of data analysis insights.
- Since the script is Markdown heavy, the recommended mode is to read the document using Jupyter Notebook. (The python script is generated from a Jupyter Notebook file using Jupytext. So, the python file (.py) will smoothly render back to the notebook file (.ipynb))

For detailed guide on Jupyetex, one can look into the following link: [Jupytext Guide](https://jupytext.readthedocs.io/en/latest/install.html)
