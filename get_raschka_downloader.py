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
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
# %%
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
# %%
# save settings dictionaries using pickle
import pickle
with open("gpt2_openai_settings.pkl", "wb") as f:
    pickle.dump(settings, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
# save params dictionary using pickle
with open("gpt2_openai_params.pkl", "wb") as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# load of pickle files - general form
#with open("data.pkl", "rb") as f:
#    loaded = pickle.load(f)   #  Never load pickles from untrusted sources