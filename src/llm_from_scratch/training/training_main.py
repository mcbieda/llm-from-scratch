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
import json
from copy import deepcopy


SRC_DIR = Path(__file__).resolve().parents[2]  # -> .../src

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_from_scratch.configs import gpt2small_config
from llm_from_scratch.training import training_utils
from llm_from_scratch.models import gpt2
from llm_from_scratch.dataloader import dataloader


# def main():
#     # get config
#     cfg = gpt2small_config.RUN_CONFIG
#     model_cfg = cfg['model_config']
#     torch.manual_seed(cfg['seed'])
#     print(model_cfg)
#     # modify cfg
#     cfg["num_epochs"]=2
#     cfg["run_name"]="gpt2small_test_2epochs_2025DEC07"
#     # setup model
#     model = gpt2.setup_model(model_cfg)
#     print(model)
#     totparams = sum(p.numel() for p in model.parameters())
#     print("Total number of parameters:", totparams)
#     # tokenizer
#     tokenizer=tiktoken.get_encoding(cfg['tokenizer'])
#     # dataloaders
#     train_loader, val_loader, test_loader = dataloader.generate_data_loaders(cfg)
#     print("Show trainer_loader first entry (converted to text):")
#     dataloader.loader_text_examine(train_loader,0,tokenizer)
#     print("Show val_loader first entry (converted to text):")
#     dataloader.loader_text_examine(val_loader,0,tokenizer)
#     if test_loader is not None:
#         print("Show test_loader first entry (converted to text):")
#         dataloader.loader_text_examine(test_loader,0,tokenizer)
#     # training loop
#     num_epochs = cfg['num_epochs']
#     optimizer=training_utils.setup_optimizer(model,cfg)
#     train_losses, val_losses, tokens_seen, global_step = training_utils.train_model_simple(
#     model=model, train_loader=train_loader, val_loader=val_loader, 
#     optimizer=optimizer,
#     device=cfg['device_name'],
#     num_epochs = num_epochs, eval_freq=5, eval_iter=5,
#     start_context = "Every effort moves you", tokenizer=tokenizer
#     )
#     # plot output
#     epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#     training_utils.plot_losses(cfg,epochs_tensor, tokens_seen, train_losses, val_losses)
#     # save checkpoint
#     training_utils.save_cfg_json(cfg=cfg, epoch=num_epochs, global_step=global_step)
#     training_utils.save_checkpoint(model=model, optimizer=optimizer, cfg=cfg,
#                     epoch=num_epochs, global_step=global_step)
    
def run_training(cfg):
    """Run a single training experiment given a config dict."""
    cfg = deepcopy(cfg)  # avoid in-place mutation

    model_cfg = cfg['model_config']
    torch.manual_seed(cfg['seed'])
    print(cfg)
    # modify cfg)
    # setup model
    model = gpt2.setup_model(model_cfg)
    totparams = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", totparams)

    # tokenizer
    tokenizer = tiktoken.get_encoding(cfg['tokenizer'])

    # dataloaders
    train_loader, val_loader, test_loader = dataloader.generate_data_loaders(cfg)
    print("Show train_loader first entry (converted to text):")
    dataloader.loader_text_examine(train_loader, 0, tokenizer)
    print("Show val_loader first entry (converted to text):")
    dataloader.loader_text_examine(val_loader, 0, tokenizer)
    if test_loader is not None:
        print("Show test_loader first entry (converted to text):")
        dataloader.loader_text_examine(test_loader, 0, tokenizer)

    # training loop
    num_epochs = cfg['num_epochs']
    optimizer = training_utils.setup_optimizer(model, cfg)
    train_losses, val_losses, tokens_seen, global_step = training_utils.train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=cfg['device_name'],
        num_epochs=cfg['num_epochs'],
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    # plot and save plot
    epochs_tensor = torch.linspace(0, cfg['num_epochs'], len(train_losses))
    training_utils.plot_losses(cfg, epochs_tensor, tokens_seen, train_losses, val_losses)

    # save cfg and checkpoint
    training_utils.save_cfg_json(
        cfg=cfg, 
        epoch=num_epochs, 
        global_step=global_step)  
    training_utils.save_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=cfg['num_epochs'],
        global_step=global_step,
    )
    training_utils.save_results(cfg,
        train_losses, val_losses, tokens_seen, global_step)
    

    # return some results
    return {
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "tokens_seen": int(tokens_seen[-1]),
        "global_step": int(global_step),
    }


def main():
    cfg = gpt2small_config.RUN_CONFIG
    run_training(cfg)

# %%
# if main
if __name__ == "__main__":
    main()