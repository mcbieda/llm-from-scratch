# get_raschka_downloader.py

import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)


# %%
# %% SET MODEL SIZE TO DOWNLOAD

# OPTIONS HERE
# model_configs = {
#     "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
#     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
# }
model_size = "355M"


# %%
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2_all"
)
# %%
# PRINT settings and parameter keys
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

# %%
# CREATE file names for pickle

# create names
settings_nm = "gpt2_openai_settings_" + model_size + ".pkl"
parameters_nm = "gpt2_openai_params_" + model_size + ".pkl"

# %%
# save settings dictionaries using pickle
import pickle
with open(settings_nm, "wb") as f:
    pickle.dump(settings, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
# save params dictionary using pickle
with open(parameters_nm, "wb") as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# load of pickle files - general form
#with open("data.pkl", "rb") as f:
#    loaded = pickle.load(f)   #  Never load pickles from untrusted sources