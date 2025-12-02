# test_dataloader.py
# Mark Bieda

# tests/test_dataloader.py


import tiktoken
from torch.utils.data import Dataset, DataLoader


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> .../src
SRC_DIR = PROJECT_ROOT / "src"  # -> .../src

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_from_scratch.dataloader.dataloader import *
from llm_from_scratch.configs import gpt2small_config

def test_dataloader_basic():
    torch.manual_seed(123)
    cfg = gpt2small_config.RUN_CONFIG
    tokenizer=tiktoken.get_encoding(cfg['tokenizer'])
    train_loader, val_loader, test_loader = generate_data_loaders(cfg)
    
    print("Show trainer_loader first entry (converted to text):")
    loader_text_examine(train_loader,0,tokenizer)
    print("Show val_loader first entry (converted to text):")
    loader_text_examine(val_loader,0,tokenizer)
    if test_loader is not None:
        print("Show test_loader first entry (converted to text):")
        loader_text_examine(test_loader,0,tokenizer)
    print("CONFIG OUTPUT")
    print(cfg)
    print("Success")
    # Sanity checks
    # assert dataloader is not None
    # batch = next(iter(dataloader))
    # inputs, targets = batch

    # # Example shape checks
    # assert inputs.ndim == 2          # [batch, seq_len]
    # assert targets.ndim == 2
    # assert inputs.shape == targets.shape
    # assert inputs.shape[1] == cfg.model.context_length

if __name__ == "__main__":
    test_dataloader_basic() 
