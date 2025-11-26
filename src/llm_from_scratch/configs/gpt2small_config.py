# gpt2small_config.py

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

RUN_CONFIG = {
    "run_name": "gpt2-small_theverdict_lr4e4_wd0p15_e20",
    "description": "gpt2-small on PubMed abstracts, 20 epochs",

    "device_name": "cpu",

    "model_name": "gpt2-small",
    "model_config": GPT2_CONFIG_124M,
    "tokenizer": "gpt2",


    "pretrained": False,
    "training_file": "/home/markb/llm-from-scratch/data/the-verdict.txt",
    "batch_size": 16,
    "lr": 4e-4,
    "weight_decay": 0.15,
    "num_epochs": 20,
    "seed": 42,
    "output_dir": "/home/markb/llm-from-scratch/output",
}