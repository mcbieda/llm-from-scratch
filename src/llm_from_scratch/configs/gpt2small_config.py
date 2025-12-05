# gpt2small_config.py

# NOTES ON PARAMETER COUNTS
# RASCHKA says 163,009,536 - this is with context_length = 1024 AND qkv_bias=FALSE
# This is what we should use for loading OPENAI weights

GPT_CONFIG_124M_OPENAI = {
    # OPENAI version if loadign openai weights
    # yields RASCHKA value of 163,009,536 if weight_tying is False
    # listed at 124M if weight_tying is True
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length was 256; this is OpenAI value
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias; False for GPT-2-small original
    "weight_tying": True    # OPENAI default; I added this; Raschka has False
}

GPT_CONFIG_124M_BETTERTRAIN = {
    # smaller context, a little more bias
    # this will be
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length changed from OPENAI of 1024
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias; FALSE in original GPT-2-small
    "weight_tying": False    # OPENAI default is True

}

GPT_CONFIG_124M_RASCHKA = {
    # OPENAI version if loadign openai weights
    # yields RASCHKA value of 163,009,536 if weight_tying is False
    # listed at 124M if weight_tying is True
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length was 256; this is OpenAI value
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias; False for GPT-2-small original
    "weight_tying": False    # OPENAI default; I added this; Raschka has False
}

RUN_CONFIG = {
    "run_name": "gpt2-small_theverdict_lr4e4_wd0p15_e4",
    "description": "gpt2-small on the verdict, 4 epochs",

    "device_name": "cpu",

    "model_name": "gpt2-small",
    "model_config": GPT_CONFIG_124M_OPENAI,
    "tokenizer": "gpt2",


    "pretrained": False,
    "training_file": "/home/markb/llm-from-scratch/data/the-verdict.txt",
    "val_file": "",
    "test_file": "",
    "val_ratio": 0.10,
    "test_ratio": 0.0,
    "stride": 128, # note context_length is in above definition

    "batch_size": 2,
    "lr": 4e-4,
    "weight_decay": 0.15,
    "num_epochs": 10,
    "seed": 42,
    "output_dir": "/home/markb/llm-from-scratch/output",
}