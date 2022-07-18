import pandas as pd
import re

from collections import Counter
import itertools
from nltk.stem import PorterStemmer

from wordcloud import STOPWORDS


def create_terms_per_category(df, category):

    """
    Takes a dataframe and a category to perform text analysis.
    Returns the top 150 most frequent terms for the chosen category.
    The returned terms are pre-translation, i.e includes non-english terms too.

    """

    if category == "General":
        category_index = 0
    elif category == "Women":
        category_index = 1
    elif category == "Children":
        category_index = 2
    elif category == "Health Difficulty":
        category_index = 3
    # For stemming and stopword removal
    ps = PorterStemmer()
    total_stopwords = list(STOPWORDS)
    total_stopwords_stemmed = [ps.stem(w) for w in total_stopwords]

    df = df.applymap(str)
    additional_total = " ".join(df.iloc[:, category_index].tolist())
    additional_splitted_total = re.split("\s|(?<!\d)[,.](?!\d)", additional_total)

    additional_splitted_total = [ps.stem(w) for w in additional_splitted_total]
    additional_splitted_total = list(filter(None, additional_splitted_total))

    additional_splitted_total = [
        word
        for word in additional_splitted_total
        if word not in total_stopwords_stemmed
    ]

    additional_count_total = Counter(additional_splitted_total)
    count_top_150_total = additional_count_total.most_common(150)
    df_top_150_total = pd.DataFrame(count_top_150_total, columns=["Term", "Frequency"])
    return df_top_150_total


def create_one_combined_file(df1, df2, df3, df4):

    """
    Takes in multiple dataframes.
    Returns a combined dataframe without repetition of words.
    In the current flow of code, takes four dataframes of four different categories.
    The returned combined dataframe will then be used for human translation activity.

    """

    # For stemming and stopword removal
    ps = PorterStemmer()
    total_stopwords = list(STOPWORDS)
    total_stopwords_stemmed = [ps.stem(w) for w in total_stopwords]

    total_words = list(
        itertools.chain(
            df1["Term"].tolist(),
            df2["Term"].tolist(),
            df3["Term"].tolist(),
            df4["Term"].tolist(),
        )
    )

    total_words_unique = list(set(total_words))

    total_words_unique_non_stopwords = [
        word for word in total_words_unique if word not in total_stopwords_stemmed
    ]
    words_to_translate = pd.DataFrame(
        total_words_unique_non_stopwords, columns=["Words"]
    )
    return words_to_translate


def create_translated_terms_per_category(df_combined, df_category):

    """
    Takes the human-translated combined terms dataframe and another dataframe to translate.
    Returns the translated most frequent terms for chosen category.

    """

    df_combined = df_combined.applymap(lambda s: s.lower() if type(s) == str else s)
    df_combined.columns = ["Term", "Is_NFRI", "English_Translation"]
    df_nfri = df_combined[df_combined["Is_NFRI"] == "yes"]
    df_translated = pd.merge(df_category, df_nfri, on="Term")
    df_translated = df_translated.loc[:, ["English_Translation", "Frequency"]]
    df_translated = (
        df_translated.groupby(["English_Translation"])
        .sum()
        .sort_values(by=["Frequency"], ascending=False)
    )
    df_translated = df_translated.reset_index()

    return df_translated
