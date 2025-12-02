# dataloader.py
# Mark Bieda
# from Raschka book, with explorations

# %%
# libraries
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader

# dataloader.py (top of file)

import sys
from pathlib import Path

# this file: .../src/llm_from_scratch/dataloader/dataloader.py
# we want:   .../src on sys.path so we can import llm_from_scratch
SRC_DIR = Path(__file__).resolve().parents[2]  # -> .../src

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_from_scratch.configs import gpt2small_config


# %%
# function make_tokenized_batch

def make_tokenized_batch(batch):
    # starting with list of sentences, returns tokenized list
    # batch is batch of sentences in text form dim: num of sentences
    tokenizer = tiktoken.get_encoding("gpt2")
    batchout = []
    for i in range(len(batch)):
        batchout.append(torch.tensor(tokenizer.encode(batch[i])))
    batchoutstack = torch.stack(batchout, dim=0)
    return batchoutstack


# %%
# token untilities

def text_to_token_ids(text, tokenizer):
    # text must be a string
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# %%
# GPT dataset stuff  
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader




# %%
# LOOK AT ENTRIES IN LOADERS  
def loader_text_examine(loader,examplenum, tokenizer):
    # loader is like train_loader
    # examplenum is example num within batch
    # examplenum=0 will always exist

    # thisexample would be a tuple with (train, target)
    thisexample = loader.dataset[examplenum]
    thisexample_decode = token_ids_to_text(thisexample[0], tokenizer)
    print(thisexample_decode.replace("\n", " "))

# LOAD DATA FILE
def load_file(filenm):
    #filenm = cfg["training_file"] #  includes path
    with open(filenm, 'r', encoding='utf-8') as f:
        text_data = f.read()
    return(text_data)

# %%
# examine data
def examine_data(text_data, tokenizer_name):
    total_characters = len(text_data)
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

# %%
# train, val and test splits
# use only if needed

def train_val_test_split(text_data, train_ratio, val_ratio, test_ratio):
    total_characters = len(text_data)
    
    if test_ratio == 0.0:
        train_index = int(train_ratio * total_characters)
        train_data = text_data[:train_index]
        val_data = text_data[train_index:]
        test_data = None
    else:
        # train - val - test split
        train_index = int(train_ratio * total_characters)
        val_index = train_index + int(val_ratio * total_characters)
        train_data = text_data[:train_index]
        val_data = text_data[train_index:val_index]
        test_data = text_data[val_index:]
    return(train_data,val_data,test_data)

def create_dataloaders(train_data, val_data, test_data, batch_size, max_length, stride, num_workers=0):
    """
    Creates and returns train, validation, and test dataloaders.

    Args:
        train_data (str): The training text data.
        val_data (str): The validation text data.
        test_data (str): The testing text data.
        batch_size (int): The number of samples per batch.
        max_length (int): The maximum length of a sequence.
        stride (int): The step size to move the window for creating sequences.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing the train, validation, and test dataloaders.
    """
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        # Shuffle during training so that batches are different in different epochs
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        # No shuffle during validation so that batches are the same
        shuffle=False,
        num_workers=num_workers
    )
    if test_data:
        test_loader = create_dataloader_v1(
            test_data,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=False,
            # No shuffle during testing so that batches are the same
            shuffle=False,
            num_workers=num_workers
        )
    else:
        test_loader=None

    return train_loader, val_loader, test_loader

# Example of how you might call this function:
# Assuming train_data, val_data, and test_data have been created from the split

# train_loader, val_loader, test_loader = create_dataloaders(
#     train_data=train_data,
#     val_data=val_data,
#     test_data=test_data,
#     batch_size=2,
#     max_length=GPT_CONFIG_124M["context_length"],
#     stride=GPT_CONFIG_124M["context_length"]
# )

def generate_data_loaders(cfg):
    check_flag=True
    train_file = cfg['training_file']
    val_file = cfg['val_file']
    test_file = cfg['test_file']

    # quick sanity check
    if check_flag:
        # just examine train_file
        text_data=load_file(train_file)
        print("check_flag is True; output of train_file")
        print(train_file)
        examine_data(text_data, cfg['tokenizer'])
        print("\n")

    if (val_file=="" and test_file==""):
        text_data=load_file(cfg['training_file'])
        train_ratio = 1 - (cfg['val_ratio'] + cfg['test_ratio'])
        train_data,val_data,test_data=train_val_test_split(text_data, train_ratio, cfg['val_ratio'], cfg['test_ratio'])
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data,
            val_data,
            test_data,
            batch_size=cfg['batch_size'],
            max_length=cfg['model_config']['context_length'],
            stride=cfg['stride']
        )
    else:
        train_data=load_file(cfg['training_file'])
        val_data=load_file(cfg['val_file'])
        test_data=load_file(cfg['test_file'])
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data,
            val_data,
            test_data,
            batch_size=cfg['batch_size'],
            max_length=cfg['model_config']['context_length'],
            stride=cfg['stride']
        )  
    return(train_loader, val_loader, test_loader)


def main():
    torch.manual_seed(123)
    cfg = gpt2small_config.RUN_CONFIG
    tokenizer=tiktoken.get_encoding(cfg['tokenizer'])
    train_loader, val_loader, test_loader = generate_data_loaders(cfg)
    # add length of each loader here
    # print(f"length of train_loader: {len(train_loader)}")
    # print(f"length of val_loader: {len(val_loader)}")
    # if test_loader is not None:
    #     print(f"length of test_loader: {len(test_loader)}")
    print("Show trainer_loader first entry:")
    loader_text_examine(train_loader,0,tokenizer)
    print("Show val_loader first entry:")
    loader_text_examine(val_loader,0,tokenizer)
    if test_loader is not None:
        print("Show test_loader first entry:")
        loader_text_examine(test_loader,0,tokenizer)
    print("CONFIG OUTPUT")
    print(cfg)
    print("Success")


if __name__ =="__main__":
    main()
