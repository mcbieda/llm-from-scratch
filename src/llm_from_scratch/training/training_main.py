# training_main.py
# Mark Bieda


# setup paths correctly for packages
import sys
from pathlib import Path
import torch
import torch.nn as nn
import tiktoken
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


SRC_DIR = Path(__file__).resolve().parents[2]  # -> .../src

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_from_scratch.configs import gpt2small_config
from llm_from_scratch.training import training_utils
from llm_from_scratch.models import gpt2
from llm_from_scratch.dataloader import dataloader


def main():
    # get config
    cfg = gpt2small_config.RUN_CONFIG
    model_cfg = cfg['model_config']
    torch.manual_seed(cfg['seed'])
    print(model_cfg)
    # modify cfg
    cfg["num_epochs"]=3
    # setup model
    model = gpt2.setup_model(model_cfg)
    print(model)
    totparams = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", totparams)
    # tokenizer
    tokenizer=tiktoken.get_encoding(cfg['tokenizer'])
    # dataloaders
    train_loader, val_loader, test_loader = generate_data_loaders(cfg)
    print("Show trainer_loader first entry (converted to text):")
    loader_text_examine(train_loader,0,tokenizer)
    print("Show val_loader first entry (converted to text):")
    loader_text_examine(val_loader,0,tokenizer)
    if test_loader is not None:
        print("Show test_loader first entry (converted to text):")
        loader_text_examine(test_loader,0,tokenizer)
    # training loop
    train_losses, val_losses, tokens_seen, global_step = train_model_simple(
    model=model, train_loader=train_loader, val_loader=val_loader, 
    optimizer=setup_optimizer(model,cfg),
    device=cfg['device_name'],
    num_epochs = cfg['num_epochs'], eval_freq=5, eval_iter=5,
    start_context = "Every effort moves you", tokenizer=tokenizer
    )
    # plot output
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    # save checkpoint
    save_checkpoint(model=model, optimizer=optimizer, cfg=cfg,
                    epoch=num_epochs-1, global_step=global_step)
    

# %%
# if main
if __name__ == "__main__":
    main()