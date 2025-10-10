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
CHOOSE_MODEL = "gpt2-medium (355M)"
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
#model_size = "355M" # can change to 355M also
# Load settings
filenm ="gpt2_openai_settings_" + model_size +".pkl"
fullnm = filepath+filenm
with open(fullnm, "rb") as f:
    settings = pickle.load(f)

# Load params
filenm ="gpt2_openai_params_"+ model_size +".pkl"
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
val_example = val_data[42]
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
# !!!!!!!!!!!!!!!!!!!
# DO NOT RUN BELOW HERE EXCEPT TO SAVE A MODEL
# !!!!!!!!!!!!!!!!!!!


# %% save model
# save model and optimizer
filepath = "/home/markb/llm-from-scratch/output/"
descripstr ="chapter7_2epoch_"+ model_size
filenm = "model_and_optimizer_" + descripstr +".pth"
fullnm = filepath + filenm
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    fullnm
)

# save model only
filepath = "/home/markb/llm-from-scratch/output/"
descripstr ="chapter7_2epoch_"+ model_size
filenm = "model_ONLY_" + descripstr +".pth"
fullnm = filepath + filenm
torch.save({
    "model_state_dict": model.state_dict()
    }, 
    fullnm
)


# %%
# EXAMINE EXAMPLES OF OUTPUT
val_example = test_data[45]
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
# PLOT LOSS
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)




# %% 
# FINAL LOSS VALUES

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"FINAL Train loss: {train_loss}")
print(f"FINAL Val loss: {val_loss}")
print(f"FINAL Test loss: {test_loss}")


# %%
