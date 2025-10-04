# chapter7_initial.py








# %%
# CHECK on end of text
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))




# %%
# SANITY CHECK FOR DATA LOADERS
# for input_batch, target_batch in train_loader:
#     pass
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)

# %%
# EXAMINING THE DATA LOADER - will use TEST here to avoid shuffle
# focus on test, because do NOT want shuffle
# There are 38 batches in test_loader, each batch has 8 text messages, except the last
# from itertools import islice

# # Note batch 38 (37 index) is the last one here, was just curious about it
# # in test, batch 38 is a truncated batch
# input_batch_1, target_batch_1 = next(islice(test_loader,37,None))

# # look at input batch
# print(input_batch_1.shape)
# print(input_batch_1[0]) #  single sentence
# enc_1 = input_batch_1[0] # single text message, encoded
# tokenizer.decode(enc_1.tolist())

# # LOOK at target_batch
# print(f"ALL target for batch:{target_batch_1}")
# print(f"target for first example in this batch:{target_batch_1[0]}")
# print(f"target for first example in this batch as scalar:{target_batch_1[0].item()}")


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
# LISTING 6.6 ADD HERE
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
text_1 = "The first step"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))


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
# calc_loss_loader function
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

# %%
# calc initial loss (instead of accuracy) for each dataset

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
print(f"Train loss: {train_loss}")
print(f"Val loss: {val_loss}")
print(f"Test loss: {test_loss}")

    
    
# %%
# fn: train classifier simple; listing 6.10 here
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [],[],[],[]
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"EPOCH:{epoch+1}")
                print(f"Step:{global_step}")
                print(f"Train loss: {train_loss}")
                print(f"Val loss: {val_loss}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches = eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches = eval_iter)
        print(f"Train accuracy: {train_accuracy*100}")
        print(f"Val accuracy: {val_accuracy*100}")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# %%
# ACTUAL TRAINING HERE

import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5,weight_decay=0.1)
num_epochs = 4

# this is just for testing to speed up testing of model
# comment out this section for actual training
# testing_temp_num_epochs = 2
# num_epochs = testing_temp_num_epochs

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq = 50,
    eval_iter=5
)

end_time=time.time()
execution_time_minutes = (end_time - start_time)/60
print(f"Training time total (min): {execution_time_minutes}")


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
# PLOT TRAINING ACCURACY RESULTS
# NOTE: does not work well with num_epochs = 1 because train_accs is 1 in this case
if (num_epochs>1):
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(
        epochs_tensor, examples_seen_tensor, train_accs, val_accs,
        label="accuracy"
    )
else:
    print("num epochs is 1, so plot is not good")
    print(f"TRAIN accuracy: {train_accs[0]}")
    print(f"VAL accuracy: {val_accs[0]}")

# %%
# COMPARE ACCURACY: train, val, test sets
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# %%
# fn: classify a bit of text
import torch.nn.functional as F
def classify_review(
        text, model, tokenizer, device, max_length=None, pad_token_id = 50256):
    # note 50257 total tokens in vocab with the pad token, which was added above. So it is position 50256
    model.eval()
    input_ids =tokenizer.encode(text) # just leads to like (55,1043,999) token ids
    supported_context_length = model.pos_emb.weight.shape[0] # 1024 for GPT-2-small

    input_ids = input_ids[:min(max_length, supported_context_length)] # collect from 0 position to maximum allowed
    len_input_ids_truncated = len(input_ids)
    input_ids += [pad_token_id] * (max_length - len(input_ids)) # if below maximum, pad with padding token
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add the batch dimension - so will be 2D

    with torch.no_grad():
        # note the "-1" below kills that dimension
        logits = model(input_tensor)[:,-1,:] # choose last token position
        predicted_label=torch.argmax(logits, dim=-1).item() # last dim is the class dim

    # debug section
    #print(f"INPUT IDS:{input_ids}")
    print(f"length of truncated initial ids: {len_input_ids_truncated}")
    print(logits)
    probs = F.softmax(logits, dim=-1)
    print(f"PROBS: {probs}")

    return "spam" if predicted_label==1 else "not spam"

# %%
# DO CLASSIFICATION
# text_1 = (
#     "You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award."
# )

# VARIOUS EXAMPLES - note that some give spam, some do not
# text_1 = "If your number matches call 09064019014 to receive your £350 award." # spam from training
text_1 = "Hi, the SEXYCHAT girls are waiting for you to text them. Text now for a great night chatting. send STOP to stop this service" # training set spam
# text_1 = "URGENT! We are trying to contact U. Todays draw shows that you have won a £800 prize GUARANTEED." # training spam
#text_1 = "Todays draw shows that you have won a £800"
# text_1 = "URGENT! your bank indicates that you have a £350 award"
# text_1 = "Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award"
# text_1 = "Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your £350 award."
#text_1 = "WIN URGENT! If your number matches call 09567890986 to receive your £10000 reward." # made up spam one
text_1 = "call 09064019014 to receive your £350 award." # partial training, spam
text_1 = "You win -  todays Vodafone numbers: receive your £350 award when you call 09064019014." # rearranged training, spam
text_1 = "Winner! receive your £900 award when you call 09064019014." # rearranged training, spam
text_1 = "There is a great new opportunity for you based on your entry in the Vodafone sweepstakes. receive your £900 award when you call 09064019014" # reworked, spam
text_1 = "just call 09064019014 for information" # reworked, NOT SPAM
text_1 = "just call 09064019014 for £350 " # NOT SPAM

print(text_1)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

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


# %%
# EXAMINE MODEL PARAMETERS for classifer layer
# 0 is non-spam, 1 is spam
# the module itself
head = model.out_head            # nn.Linear(768 -> 2)

# tensors (track grads)
W = model.out_head.weight        # shape: [2, 768]
b = model.out_head.bias          # shape: [2]
print(W.shape, b.shape)

# LOOK at bias
print(f"Bias:{b}")
# for 4 epochs, leads to tensor([ 0.0145, -0.0187]
# these values are pretty small compared to the logits that I am seeing - note they are on logit scale, not probs

# Look at Weights
W_0 = W[0]
W_1 = W[1]
W_0_list = W_0.tolist()
W_1_list = W_1.tolist()
W_diff = W_1 - W_0
W_diff_abs = abs(W_diff)
plot_tensor_hist(W_diff_abs)
plot_tensor_by_index(W_diff_abs, kind = "line")
# CONCLUSION: interesting look to the histogram, not much pattern across the final 768 with weight difference

# COMMENTS ON THIS
# 1. maybe the predicted next token fits into a certain family or similarity vs the total set
#  - like spam would predict a certain set of next tokens and nonspam a different set, but those sets having regularities


