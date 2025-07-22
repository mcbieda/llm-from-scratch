# chapter4_initial.py
# GOAL: chapter 4 code exploration
# from Raschka book

# %%
# libraries
import torch
import torch.nn as nn

# %%
# config file
# start with GPT-2
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
GPT_CONFIG_test01 = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 6,  # Context length
    "emb_dim": 10,          # Embedding dimension
    "n_heads": 2,           # Number of attention heads
    "n_layers": 2,          # Number of layers
    "drop_rate": 0.5,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}


# %%
# Basic Dummy Transformer Block
# code listing 4.1
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):

        # note initialize inherited class
        super().__init__()

        # EMBEDDING LAYER
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        #   what is the type of positional encoding used here?
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        #  dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # TRANSFORMER BLOCKS
        #   each layer is a single transformer block
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        # Layer Normalization - note this is where could try tanh instead
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # setup to get the non-norm distribution of the "next word"
        # note that can just take the max value from this to get the token value for the predicted next word
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):

        # in_idx only has the token numbers, I believe
        batch_size, seq_len = in_idx.shape

        # get token embeddings and positional embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # CLASSIC step by step through the network
        # add token and positional embeddings
        x = tok_embeds + pos_embeds
        # dropout
        x = self.drop_emb(x)
        # transformer blocks
        x = self.trf_blocks(x)
        # layer norm
        x = self.final_norm(x)
        # linear layer to get the next token
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    

# %%
# setup dummy data
# I think that I would only need a set of sentences converted into token values
# "The cat ate the rat" and "A dog is a big" - each length 5
# then something with token values like:
dummy_data = torch.tensor([[45,32,198,7000,5003,99],[76,432,598,45000,22,371]])
print(dummy_data)
print(dummy_data.shape)


# %%
# run dummy data
testmodel = DummyGPTModel(GPT_CONFIG_test01)
output = testmodel(dummy_data)
# output here
print(output)
print(output.shape) # (B,T,vocab_size)
# why (B,T,vocab_size)?
# there are two sentences, so B=2 in example
# Each sentence had 6 tokens. We are used masked attention, so each row is the set of 
#   logits for the next word. For (1,1,:), the next word would be the highest value and would
#   correspond to the most probable next word after "The"; (1,2,:) is next word after "The cat"




# %%
