# get_abstract_counts_by_year.py
# Mark Bieda

import csv
import pandas as pd

# %%
# PARAMS
filepath = "/home/markb/bio-llm/tokenizer/"
filenm = "pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_FULL.csv"
fullnm_read = filepath + filenm
output_filenm = "pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_counts_by_year.csv"
output_fullnm = filepath + output_filenm

# fraction for validate and test
val_fraction = 0.02
test_fraction = 0.05


# %%
# read in a csv file. Set the filenm and path

df = pd.read_csv(fullnm_read)

# %%
# check the dataframe
print(df.head())
print(df.columns)
print(df.shape)

# %%
df = df[df['Abstract'].apply(lambda x: isinstance(x, str))]
print(f"DataFrame after removing rows with float abstracts: {df.shape}")

# filter df so that if abstract is float, then eliminate that whole row 

# %%
# separate into train, val, test
# first shuffle the dataframe
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

num_rows = len(df_shuffled)
num_val = int(num_rows * val_fraction)
num_test = int(num_rows * test_fraction)
num_train = num_rows - num_val - num_test

df_val = df_shuffled.iloc[:num_val]
df_test = df_shuffled.iloc[num_val:num_val + num_test]
df_train = df_shuffled.iloc[num_val + num_test:]

print(f"Total rows: {num_rows}")
print(f"Train rows: {len(df_train)}")
print(f"Validation rows: {len(df_val)}")
print(f"Test rows: {len(df_test)}")

# %%
# save to csv files
base_name = filenm.replace("_FULL.csv", "")

df_train.to_csv(f"{filepath}{base_name}_train.csv", index=False)
df_val.to_csv(f"{filepath}{base_name}_val.csv", index=False)
df_test.to_csv(f"{filepath}{base_name}_test.csv", index=False)

print(f"Train data saved to {filepath}{base_name}_train.csv")
print(f"Validation data saved to {filepath}{base_name}_val.csv")
print(f"Test data saved to {filepath}{base_name}_test.csv")

# %%
# save only the last field from each df into a file for each set. This is a text file

def save_abstracts_to_txt(dataframe, output_path):
    with open(output_path, "w", encoding="utf-8") as fh:
        for abstract in dataframe["Abstract"]:
            #fh.write(abstract.replace("\n", " ").strip() + "\n")
            fh.write(abstract + "\n")

save_abstracts_to_txt(df_train, f"{filepath}{base_name}_train_abstracts.txt")
save_abstracts_to_txt(df_val, f"{filepath}{base_name}_val_abstracts.txt")
save_abstracts_to_txt(df_test, f"{filepath}{base_name}_test_abstracts.txt")

print(f"Train abstracts saved to {filepath}{base_name}_train_abstracts.txt")
print(f"Validation abstracts saved to {filepath}{base_name}_val_abstracts.txt")
print(f"Test abstracts saved to {filepath}{base_name}_test_abstracts.txt")

# %%
