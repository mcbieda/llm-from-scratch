# get_raschka_downloader.py

import json
import os
import urllib.request
import tensorflow as tf

# === CONFIGURATION BLOCK ===
# Set these values before running the script:
# - model_size: GPT-2 variant to download (e.g., "124M", "355M", "774M", "1558M")
# - models_dir: local folder name where model files will be stored
# Options for reference (kept commented out):
# model_configs = {
#     "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
#     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
# }
model_size = "124M"
models_dir = "gpt2_all_2026"
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)


# %%
# %% SET MODEL SIZE TO DOWNLOAD
#
# (model_size is configured at the top of the file)
# %%
from gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt
models_dir_full = os.path.join(models_dir, model_size)
os.makedirs(models_dir_full, exist_ok=True)
expected_files = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]
if all(os.path.exists(os.path.join(models_dir_full, name)) for name in expected_files):
    abs_models_dir_full = os.path.abspath(models_dir_full)
    abs_expected_paths = [
        os.path.join(abs_models_dir_full, name) for name in expected_files
    ]
    print(
        "Files not downloaded again because copies already detected at download location:",
        abs_models_dir_full,
    )
    print("Detected files:")
    for path in abs_expected_paths:
        print(f"  {path}")
    tf_ckpt_path = tf.train.latest_checkpoint(models_dir_full)
    settings = json.load(
        open(os.path.join(models_dir_full, "hparams.json"), "r", encoding="utf-8")
    )
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
else:
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=models_dir
    )
    print(f"OpenAI model files are available in: {os.path.abspath(models_dir_full)}")
# %%
# PRINT settings and parameter keys
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

# %%
# CREATE file names for pickle

# create names
settings_nm = os.path.join(models_dir_full, "gpt2_openai_settings_" + model_size + ".pkl")
parameters_nm = os.path.join(models_dir_full, "gpt2_openai_params_" + model_size + ".pkl")

# %%
# save settings dictionaries using pickle
import pickle
with open(settings_nm, "wb") as f:
    pickle.dump(settings, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
# save params dictionary using pickle
with open(parameters_nm, "wb") as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Pickle files written to:")
print(f"  {os.path.abspath(settings_nm)}")
print(f"  {os.path.abspath(parameters_nm)}")

# %%
# load of pickle files - general form
#with open("data.pkl", "rb") as f:
#    loaded = pickle.load(f)   #  Never load pickles from untrusted sources
