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

## Free Text Analysis

- The following analysis contains the free texts analysis of NFRI Survey data.

#### Introduction

- In the community survey conducted in Mahottari and Sindhupalchok districts of Nepal, there were four questions asking respondents to give suggestions for new items that should be included in NFRI packages across the General, Women, Children and Health Difficulty categories.
- In our Kobo setting of survey, the questions for specific categories would only come up for households with the condition satisfied.
  (For example, the question for Children would only come up in households with children count greater than zero.)
- The analysis is about the new NFRI items that were suggested by the respondents.

#### Free Text Activity: Combining Machine Intelligence and Human Intelligence

- As the data was collected in different formats, some of the words are in Romanized Nepali (meaning Nepali words written using English script), Nepali script or in English. Sometimes the English has the wrong spelling. For us to be able to analyse the data, it is important to translatate different script/spelling variations of the same word into one.
- While we did try options for translation using python libraries, the library led translation was limited due to significant presence of Romanized Nepali terms.
- Along with translation, it is also important to filter non NFRI terms from the list. For example: food items, cash, other filler words or already distributed NFRI items that could not be considered for inclusion in NFRI packages. (And this would require a bit of domain knowledge about NFRI.)
- Thus, we conducted Free Text Activity both filtering and translation of terms, where the Red Cross members helped us translate, filter and review the terms.

#### Code Flow:

- In the first part, we identify the 'non translated' and ' non filtered' most frequent terms requested across all four categories, and create a single CSV file consisting of combined terms across all categories.
- The created CSV file is then translated and filtered using Collective Human Intelligence. (This part is performed outside of this code.)
- In the second part, we read the translated and filtered CSV file, and merge that with the terms across different categories to identify the new NFRI items suggested by the respondents.
- The most frequent terms across all four categories (translated and filtered) are then written in a different CSV file for each.

#### To Run the Code

- Please run the free_text_analysis.py file stored in the free_text_analysis folder inside the analysis folder.
- File path: cci_nepal -> analysis -> free_text_analysis -> free_text_analysis.py
- The necessary functions to run the python file above are stored in separate python files in getters and pipeline folders respectively.

#### Final Output

- The final output of the code is a CSV file, containing the most frequent new NFRI item terms suggested by the respondents.
- The CSV file will be created for all four categories mentioned above.
