# gpt2.py
# Mark Bieda
# from Raschka book, with explorations

# %%
# libraries
import torch
import torch.nn as nn
import tiktoken


# !!!!!!!!!!!!!!!!!!!!
# from chapter 4
# !!!!!!!!!!!!!!!!!!!!

# %%
# config file
# start with GPT-2
# 124M is GPT-2-small
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
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



# %% multihead
# listing 3.5 multihead attention combined
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__() # initialize the inherited class
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # head dimension is number of features in the embedding space that each head gets
        self.head_dim = d_out // num_heads

        # Linear because want to make this easy to do the weight * input calculations
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # setup out_proj as a Linear layer too
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout, note default enabled but disabled if model.eval() is used
        self.dropout = nn.Dropout(dropout)

        # Mask prep - but see forward method
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):

        # for full dimensionality explanation of this method, see:
        #  https://chatgpt.com/share/687db459-fe78-8003-90f3-3999e18b73ff
        b, num_tokens, d_in = x.shape # note 3D tensor required

        # Because Linear is setup above, can just pass input
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the heads - view allows the split of d_out into:
        #  self.num_heads and self.head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)                                                                   

        # Transpose num_tokens and self.num_heads
        #   this makes sense because we want last two dimensions to be like (T,d_out)
        #   but note that it is NOT d_out, it is the portion of d_out that each head gets
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Classic attention head calculation QK^T
        attn_scores = queries @ keys.transpose(2, 3)

        # Mask creation step 2 (part1 was in the constructor)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Fill in the masked values here with -Inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale the attention weights by sqrt(d) and apply softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # Note dropout is present unless model.eval() is run
        attn_weights = self.dropout(attn_weights)

        # Context vector calculation and transpose so get back to original dimension order
        #   (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # contiguous - because we need to make a copy here for memory reasons
        #  overall puts back to (b, num_tokens, d_out)
        #  so "collapses" the last two dimensions into one
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # projection layer, can change dimension here - note is Linear layer and
        #  defined in constructor above
        context_vec = self.out_proj(context_vec)
        return context_vec


# %%
# layernorm
# listing 4.2
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# %%
# GELU
# listing 4.3
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# %%
# feedforward
# listing 4.4
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# %%
# transformer block

# comment this next line because I have MultiHeadAttention declared above
# from chapter03 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # each block has layernorm done with the input, the attention head, dropout for training, then
        #  shortcut connection is added back
        # note d_in=d_out=emb_dim for all of this
    
        shortcut = x #  dim: (B,T,d_in)
        x = self.norm1(x) #  dim: (B,T,d_in)
        x = self.att(x) #  dim: (B,T,d_out)
        x = self.drop_shortcut(x) # dim: (B,T,d_out)
        x = x + shortcut # add in the original input (shortcut) - still (B,T,d_out)

        # this is post-attention
        # do layer norm with the input, which has shortcut added; do feedforward;
        #   dropout for training; add the shortcut back and return output
        # IMPORTANT NOTE ON SHORTCUT: this is always the input to this section - so the shortcut
        #   here is different from the shortcut in the above part
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
    
# %%
# GPT class
# listing 4.7
class GPTModel(nn.Module):
    def __init__(self, cfg):
        # note initialize inherited class
        super().__init__()
        # cfg is dictionary holding key:value for parameters (parameter name:value)
        # embedding, token and position and setup dropout for learning at embedding layer
        # token and positional embedding are LEARNED parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # note positional embedding is vectors of d_in (= emb_dim) during training
        # NOT prespecified approach, all just learning
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # embedding level dropout is important because we are learning the token and pos embedding
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # one TransformerBlock per layer
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # after Transformer, do a final layer norm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # final linear layer that has embed dimension to vocab_size transformation
        # so input here is emb_dim nodes
        # output is vocab_size nodes
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # in_idx is input sentences in token number format
        # each input sentence converted in token numbers by tokenizer
        batch_size, seq_len = in_idx.shape
        # convert batch of sentences in token numbers -> embedding vectors
        tok_embeds = self.tok_emb(in_idx) #  dim: (b,T,d_in)

        # add the positional embedding
        # see above: these are learned vectors
        # note that positional embedding will be the same for the nth token of each batch
        #  so this only produces a positional embedding vector for each token position
        #  each T position -> NOT each (b,T). batch doesn't matter
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        ) #  dim (seq_len, d_in) in the end
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# %%
# functions: forward model
# run make_tokenized_batch
# run setup_model



def setup_model(cfg):
    # cfg must be the dictionary with parameters for configuration
    # eg GPT_CONFIG_124M
    torch.manual_seed(123)
    model = GPTModel(cfg)
    return model


if __name__ == "__main__":
    setup_model(GPT_CONFIG_124M)





