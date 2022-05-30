## Free Text Analysis

- The following analysis contains the free texts analysis of NFRI Survey data.

#### Introduction

- In the community survey conducted in Mahottari and Sindhupalchok districts of Nepal, there were four questions asking respondents to give suggestions for new items that should be included in NFRI packages across the General, Women, Children and Health Difficulty categories.
- In our Kobo setting of survey, the questions for specific categories would only come up for households with the condition satisfied.
  (For example, the question for Children would only come up in households with children count greater than zero.)
- The analysis is about the new NFRI that were suggested by the respondents.

#### Free Text Activity: Combining Machine Intelligence and Human Intelligence

- As the data was collected in different formats, some of the words are in Romanized Nepali (meaning Nepali words written using English script), Nepali script or in English. Sometimes the English has the wrong spelling. For us to be able to analyse the data, it is important to translatate different script/spelling variations of the same word into one.
- While we did try options for translation using python libraries, the library led translation was limited due to significant presence of Romanized Nepali terms.
- Along with translation, it is also important to filter non NFRI terms from the list. For example: food items, cash, other filler words or already distributed NFRI that could not be considered for inclusion in NFRI packages. (And this would require a bit of domain knowledge about NFRI.)
- Thus, we conducted Free Text Activity both filtering and translation of terms, where the Red Cross members helped us translate, filter and review the terms.

#### Code Flow:

- In the first part, we identify the 'non translated' and ' non filtered' most frequent terms requested across all four categories, and create a single CSV file consisting of combined terms across all categories.
- The created CSV file is then translated and filtered using Collective Human Intelligence. (This part is performed outside of this code.)
- In the second part, we read the translated and filtered CSV file, and merge that with the terms across different categories to identify the new NFRI items suggested by the respondents.
- The most frequent terms across all four categories (translated and filtered) are then written in a different CSV file for each.

#### To Run the Code

Follow the steps below to run the free_text_analysis python script.

```shell
$ cd cci_nepal/analysis/free_text_analysis
$ python3 free_text_analysis.py
```

#### Final Output

The final output of the code is a series of CSV file, each containing the most frequent new NFRI terms suggested for that category.

The following five files created from running the models and saved to `outputs/data/free_text_analysis`:

`combined_terms_to_translate.csv`
`general_terms_translated.csv`
`female_terms_translated.csv`
`children_terms_translated.csv`
`health_difficulty_terms_translated.csv`
