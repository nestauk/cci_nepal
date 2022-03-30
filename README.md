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

- In the community survey conducted in Mahottari and Sindhupalchok districts of Nepal, there were four questions asking respondents to give suggestions for new items that should be included in NFRI packages across the General, Women, Children and Health Difficulty categories. The analysis is about the new NFRI items that were suggested by the respondents.

#### Free Text Activity: Combining Machine Intelligence and Human Intelligence

- As the data was collected in different formats, some of the words are in Romanized Nepali (meaning Nepali words written using English script), Nepali script or in English. Sometimes the English has the wrong spelling. For us to be able to analyse the data, it is important to translatate different script/spelling variations of the same word into one.
- It is also important to filter non NFRI terms from the list. For example: food items, cash, other filler words or already distributed NFRI items that could not be considered for inclusion in NFRI packages.
- Thus, we have used human intelligence in both filtering and translation of terms.

#### Code Flow: The Story of two halves

- It is easier to understand the flow of the code in terms of two halves:
- In the first half of the analysis, we identify the 'non translated' and ' non filtered' most frequent terms requested across all four categories, and create a single CSV file consisting of combined terms across all categories.
- The created CSV file is then translated and filtered using Collective Human Intelligence. (This part is performed outside of this code.)
- In the second half of the analysis, we read the translated and filtered CSV file, and merge that with the terms across different categories to identify the new NFRI items suggested by the respondents.

#### To Run the Code

- Please run the free_text_analysis.py file stored in the full_data folder inside the analysis folder.
- File path: cci_nepal -> analysis -> full_data -> free_text_analysis.
- The necessary functions to run the python file above are stored in separate python files in getters and pipeline folders respectively.

#### Final Output

- The final output of the Code is a dataframe table of most frequent new NFRI item terms suggested by the respondents.
- The table will be displayed for all four categories mentioned above.
