# chapter6_initial.py


# %%
# GET DATA
import urllib.request
import zipfile
import os
from pathlib import Path

# ------------------
# UNCOMMENT below to get the data - I have already gotten the data
# ------------------

# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


# def download_and_unzip_spam_data(
#         url, zip_path, extracted_path, data_file_path):
#     if data_file_path.exists():
#         print(f"{data_file_path} already exists. Skipping download "
#               "and extraction."
#         )
#         return

#     with urllib.request.urlopen(url) as response:
#         with open(zip_path, "wb") as out_file:
#             out_file.write(response.read())

#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extracted_path)

#     original_file_path = Path(extracted_path) / "SMSSpamCollection"
#     os.rename(original_file_path, data_file_path)
#     print(f"File downloaded and saved as {data_file_path}")

# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# %%
# LOAD DATA FROM FILE (if above already done)
import pandas as pd
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
df

# %%
# BALANCING DATASET
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

# %%
# change label naming
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# %%
# SPLIT DATASET
def random_split(df, train_frac, validation_frac):

    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)


    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(
    balanced_df, 0.7, 0.1)

# %%
# SAVE dataset splits to CSV
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

# %%
# CHECK on end of text
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# %%
# SPAM DATASET SETUP
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # note this line
        self.encoded_texts = [
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]


    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# %%
# SETUP DATASETS
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# %%
# SETUP DATALOADERS
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# %%
# SANITY CHECK FOR DATA LOADERS
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

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

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
# %%
# freeze model
for param in model.parameters():
    param.requires_grad=False

# %%
# ADD LAST CLASSIFIER
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features = BASE_CONFIG["emb_dim"],
    out_features = num_classes
)

# %%
# set last transformer block and final output block to training = True
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# %%
# Listing 6.8 here - calc accuracy
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0,0

    if num_batches is None:
        num_batches=len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch,target_batch) in enumerate(data_loader):
        if i< num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1,:] # (B, T, C) goes to (B,C) and it is -1 because we want last sequence
            predicted_labels = torch.argmax(logits, dim = -1) # for (B,C), want to look at the C dimension

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels ==target_batch).sum().item()
            )
        else:
            break
    return correct_predictions/num_examples

# %%
# (slow) TEST before training (initial) accuracy

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy =  calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Train: {train_accuracy*100}")
print(f"Val: {val_accuracy*100}")
print(f"Test: {test_accuracy*100}")

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
num_epochs = 5

# this is just for testing to speed up testing of model
# comment out this section for actual training
testing_temp_num_epochs = 2
num_epochs = testing_temp_num_epochs

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
