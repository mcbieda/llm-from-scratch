# get_abstract_counts_by_year.py
# Mark Bieda

import csv
import pandas as pd

# %%
# PARAMS
filepath = "/home/markb/bio-llm/tokenizer/"
# pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_val.csv
filenm = "pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_val.csv"
fullnm_read = filepath + filenm
output_filenm = "pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_val_counts_by_year.csv"
output_fullnm = filepath + output_filenm

# %%
# read in a csv file. Set the filenm and path

df = pd.read_csv(fullnm_read)

# %%
# check the dataframe
print(df.head())
print(df.columns)
print(df.shape)

# %%
# count by year
df_counts = df.groupby("Year").size().reset_index(name="Count")
print(df_counts)

# %%
# save to csv
df_counts.to_csv(output_fullnm, index=False)
print(f"Counts saved to {output_fullnm}")
