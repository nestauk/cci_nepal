def nfri_preferences_to_numbers(df):
    """
    Takes in a dataframe a returns a dataframe with nfri categorical preferences transformed into numbers.
    """
    mapping = {"essential": 3, "desirable": 2, "unnecessary": 1}
    return df.applymap(lambda s: mapping.get(s) if s in mapping else s)
