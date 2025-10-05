# chapter7_initial.py




# %%
# INITIAL DEFINITIONS
device = "cpu" # hard code for now

# %%
# CHECK on end of text
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


# %%
# Load instruction data listing 7.1
import json
import os
import urllib

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))


# %%
# listing 7.2 for formatting
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

# %%
# data: train, val, test division
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# %%
# listing 7.4 instruction dataset class
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data=data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input=format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self,index):
        return self.encoded_texts[index]
    def __len__(self):
        return len(self.data)

# %%
# listing 7.5 custom batch collate
def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index = -100,
        allowed_max_length = None,
        device = "cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [],[]

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # because of above, last token is always pad_token_id
        # we always add at least one more than original, so ok
        #   to delete last one here in inputs
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # note this mask formulation would break if pad_token_id were allowed in the middle of text
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze() # indices of pad_token_id
        if indices.numel() >1:
            targets[indices[1:]] = ignore_index # keep first pad_token, replace rest with -100
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# %%
# fn: customized collate
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# %%
# DATALOADERS
from torch.utils.data import DataLoader

num_workers= 0
batch_size = 8

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn = customized_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle=False,
    drop_last = False,
    num_workers = num_workers
)

# %%
# EXAMINE DATA
print(f"val data 1: {val_data[1]}")
format_val = format_input(val_data[1])
print(f"formatted: {format_val}")

# %% 
# SIDE EXAMINATION OF logits, probs and crossentropy

import torch

logits_1 = torch.tensor(
    [[-1.0,  1.0],
     [-0.5,  1.5]]
)

probs = torch.softmax(logits_1, dim=1)   # or dim=-1
print(probs)
print(probs.sum(dim=1))  # sanity check: each row sums to 1









# %%
# MODEL CONFIG DICTIONARIES
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# %%
# LOAD PICKLE DATA and load into model
#from chapter5_gpt_loadonly_openai_gpt2 import download_and_load_gpt2
from chapter5_gpt_loadonly_openai_gpt2 import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# settings, params = download_and_load_gpt2(
#     model_size=model_size, models_dir="gpt2"
# )

# LOAD FROM PICKLE FILES - MB alteration  
# here is for GPT2-small
import pickle
filepath = "/home/markb/llm-from-scratch/data/"

# Load settings
# note this is for the 124M one - need to rename
filenm ="gpt2_openai_settings.pkl"
fullnm = filepath+filenm
with open(fullnm, "rb") as f:
    settings = pickle.load(f)

# Load params
# note this is for the 124M one - need to rename
filenm ="gpt2_openai_params.pkl"
fullnm = filepath+filenm
with open(fullnm, "rb") as f:
    params = pickle.load(f)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# %%
# TEST LOADING AND RUNNING
from chapter5_gpt_loadonly_openai_gpt2 import generate_text_simple
from chapter5_gpt_loadonly_openai_gpt2 import text_to_token_ids, token_ids_to_text

#text_1 = "Every effort moves you"
# text_1 = "The first step"
val_example = val_data[2]
format_val_example = format_input(val_example)
print(f"formatted val: {format_val_example}")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(format_val_example, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"]
)
print("\n\n")
print(f"RESULT:\n\n {token_ids_to_text(token_ids, tokenizer)}")


# %%
# fn: evaluate_model
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

# %%
# calc_loss_batch function
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# %%
# fn load: calc_loss_loader, train_model_simple
from chapter5_gpt_loadonly_openai_gpt2 import calc_loss_loader
from chapter5_gpt_loadonly_openai_gpt2 import train_model_simple


# %%
# calc initial loss for each dataset

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Train loss: {train_loss}")
print(f"Val loss: {val_loss}")
print(f"Test loss: {test_loss}")



# %%
# ACTUAL TRAINING HERE

import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5,weight_decay=0.1)
num_epochs = 2

# this is just for testing to speed up testing of model
# comment out this section for actual training
# testing_temp_num_epochs = 2
# num_epochs = testing_temp_num_epochs

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time=time.time()
execution_time_minutes = (end_time - start_time)/60
print(f"Training time total (min): {execution_time_minutes}")


# %%
# EXAMINE EXAMPLES OF OUTPUT
val_example = test_data[34]
format_val_example = format_input(val_example)
print(f"original data: {val_example}")
print(f"formatted val: {format_val_example}")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(format_val_example, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"]
)
print("\n\n")
print(f"RESULT:\n\n {token_ids_to_text(token_ids, tokenizer)}")


# %%
# fn: plot_values :: PLOT TRAINING RESULTS
import matplotlib.pyplot as plt

def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))


    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()


    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# %%
# PLOT TRAINING LOSS RESULTS
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


# %%
# SIDE POINT: examine the tokenizer  

textLST = ["$","500","$500","£","£800", "100000078", "350", "4882"]
for text in textLST:
    tokenids = tokenizer.encode(text)
    print(text," ",tokenids)

# INTERPRETATION: the $ and pound sign have different encodings, this is important in this case

# %%
# tensor histogram and by position from chatGPT
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_tensor_hist(t, bins=50, density=False, value_range=None, title=None, show=True, ax=None, drop_nonfinite=True):
    """
    Plot a histogram of values from a 1-D tensor (or array-like).

    Args:
        t: 1-D torch.Tensor (CPU or CUDA), list, or numpy array.
        bins (int or sequence): Number of bins or explicit bin edges.
        density (bool): Normalize to a probability density if True.
        value_range (tuple): (min, max) range of the histogram.
        title (str): Optional title. Defaults to a descriptive one.
        show (bool): If True, calls plt.show().
        ax (matplotlib.axes.Axes): Optional existing axes to draw on.
        drop_nonfinite (bool): If True, drop NaN/Inf values before plotting.

    Returns:
        (fig, ax): The matplotlib Figure and Axes used.
    """
    # Convert to a flat NumPy array, safely handling torch tensors (incl. CUDA)
    if isinstance(t, torch.Tensor):
        data = t.detach().flatten().to("cpu").numpy()
    else:
        data = np.asarray(t).ravel()

    if drop_nonfinite:
        data = data[np.isfinite(data)]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    counts, bins_edges, patches = ax.hist(data, bins=bins, range=value_range, density=density)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    if title is None:
        title = f"Histogram • n={len(data)} • bins={bins}"
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    if show:
        plt.show()

    return fig, ax



def plot_tensor_by_index(
    t,
    kind="line",                 # "line" | "scatter" | "bar"
    sample=None,                 # int or None
    figsize=(12, 4),             # make it wide by default
    title=None,
    show=True,
    ax=None,
    drop_nonfinite=True,
    bar_width=1.0                # width for bar mode (in index units)
):
    """
    Plot value vs. position (index) for a 1-D tensor/array.

    Args:
        t: 1-D torch.Tensor (CPU/CUDA), list, or numpy array.
        kind: "line", "scatter", or "bar".
        sample: If set and len(data) > sample, evenly subsample to this many points.
        figsize: (width, height) in inches for a new figure.
        title: Optional title. Auto if None.
        show: If True, calls plt.show().
        ax: Existing matplotlib Axes to draw on; if None, creates a new figure/axes.
        drop_nonfinite: Drop NaN/Inf before plotting.
        bar_width: Width of bars when kind="bar" (in index units).

    Returns:
        (fig, ax): Matplotlib Figure and Axes.
    """
    # Flatten & move to CPU if torch tensor
    if isinstance(t, torch.Tensor):
        data = t.detach().flatten().to("cpu").numpy()
    else:
        data = np.asarray(t).ravel()

    if drop_nonfinite:
        data = data[np.isfinite(data)]

    n = data.size
    if n == 0:
        raise ValueError("No data to plot (empty tensor after filtering).")

    # Optional even subsampling for long series
    if isinstance(sample, int) and sample > 0 and n > sample:
        idxs = np.linspace(0, n - 1, num=sample, dtype=int)
        x = idxs
        y = data[idxs]
    else:
        x = np.arange(n)
        y = data

    # Create axes if needed (use figsize here)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
    else:
        fig = ax.figure

    kind = kind.lower()
    if kind == "scatter":
        ax.scatter(x, y, s=8)
    elif kind == "bar":
        # For bars, ensure width doesn't exceed spacing between indices
        w = min(bar_width, 0.9 if len(x) > 1 else bar_width)
        ax.bar(x, y, width=w, align="center")
    else:  # "line" (default)
        ax.plot(x, y, linewidth=1)

    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    if title is None:
        title = f"Value by Position • n={len(y)} • kind={kind}"
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


