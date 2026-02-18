from pathlib import Path
import pandas as pd

# %%
# PATH SETUP
def find_project_root(start: Path) -> Path:
    """
    Resolve repo root by walking upward until both `src/` and `notebooks/` exist.
    """
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "notebooks").exists():
            return p
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PARAMS
filenm = "pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_FULL.csv"
fullnm_read = DATA_DIR / filenm

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

train_csv = DATA_DIR / f"{base_name}_train.csv"
val_csv = DATA_DIR / f"{base_name}_val.csv"
test_csv = DATA_DIR / f"{base_name}_test.csv"

df_train.to_csv(train_csv, index=False)
df_val.to_csv(val_csv, index=False)
df_test.to_csv(test_csv, index=False)

print(f"Train data saved to {train_csv}")
print(f"Validation data saved to {val_csv}")
print(f"Test data saved to {test_csv}")

# %%
# save only the last field from each df into a file for each set. This is a text file

def save_abstracts_to_txt(dataframe, output_path):
    with open(output_path, "w", encoding="utf-8") as fh:
        for abstract in dataframe["Abstract"]:
            #fh.write(abstract.replace("\n", " ").strip() + "\n")
            fh.write(abstract + "\n")

train_txt = DATA_DIR / f"{base_name}_train_abstracts.txt"
val_txt = DATA_DIR / f"{base_name}_val_abstracts.txt"
test_txt = DATA_DIR / f"{base_name}_test_abstracts.txt"

save_abstracts_to_txt(df_train, train_txt)
save_abstracts_to_txt(df_val, val_txt)
save_abstracts_to_txt(df_test, test_txt)

print(f"Train abstracts saved to {train_txt}")
print(f"Validation abstracts saved to {val_txt}")
print(f"Test abstracts saved to {test_txt}")

# %%
