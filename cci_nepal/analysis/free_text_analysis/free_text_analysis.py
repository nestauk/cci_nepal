# Import libraries
import logging

# Project libraries
import cci_nepal
from cci_nepal.getters import get_data as gd
from cci_nepal.pipeline import term_manipulation as tm

# %%
# Set the project directory
project_dir = cci_nepal.PROJECT_DIR

# Read the original survey data

df = gd.read_complete_data()

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
).to_csv(
    f"{project_dir}/outputs/data/free_text_analysis/combined_terms_to_translate.csv"
)

# Read the human translated free text data
df_combined = gd.read_free_text_activity_data()

# Write the translated terms per category in the Output folder

tm.create_translated_terms_per_category(df_combined, df_general).to_csv(
    f"{project_dir}/outputs/data/free_text_analysis/general_terms_translated.csv"
)
tm.create_translated_terms_per_category(df_combined, df_women).to_csv(
    f"{project_dir}/outputs/data/free_text_analysis/women_terms_translated.csv"
)
tm.create_translated_terms_per_category(df_combined, df_children).to_csv(
    f"{project_dir}/outputs/data/free_text_analysis/children_terms_translated.csv"
)
tm.create_translated_terms_per_category(df_combined, df_health_difficulty).to_csv(
    f"{project_dir}/outputs/data/free_text_analysis/health_difficulty_terms_translated.csv"
)

# Just to ensure the code runs smoothly til the end
logging.info("Please check the newly written files in the outputs folder.")
