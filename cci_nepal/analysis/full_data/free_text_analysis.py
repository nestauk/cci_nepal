# Import libraries
import logging

# Project libraries
import cci_nepal
from cci_nepal.getters.real_data import get_real_data as grd
from cci_nepal.pipeline.real_data import term_manipulation as tm

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# Read the original survey data

df = grd.read_complete_data()

# Select only the free-text columns from the original dataframe
df_text = df.iloc[:, -5:-1]

# Create dataframes with most frequent terms for each categories
df_general = tm.create_terms_per_category(df_text, "General")
df_women = tm.create_terms_per_category(df_text, "Women")
df_children = tm.create_terms_per_category(df_text, "Children")
df_health_difficulty = tm.create_terms_per_category(df_text, "Health Difficulty")

# Combine untranslated terms without repetition for human translation activity
tm.create_one_combined_file(
    df_general, df_women, df_children, df_health_difficulty
).to_csv(f"{project_dir}/outputs/data/free text/Combined Terms To Translate.csv")

# Read the human translated free text data
df_combined = grd.read_free_text_activity_data()

# Output translated terms per category using human translated data
logging.info(tm.create_translated_terms_per_category(df_combined, df_general))
logging.info(tm.create_translated_terms_per_category(df_combined, df_women))
logging.info(tm.create_translated_terms_per_category(df_combined, df_children))
logging.info(tm.create_translated_terms_per_category(df_combined, df_health_difficulty))
