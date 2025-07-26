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
# just separating things
print("THIS IS THE START OF THE ACTUAL FULL FORWARD")


# %%
# tiktoken
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)


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

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

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

        super().__init__()
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
# testing GPT
torch.manual_seed(123)

# KEY LINE: setup the model
model = GPTModel(GPT_CONFIG_124M)

model.eval()
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# %%
# a second test
tokenizer = tiktoken.get_encoding("gpt2")
batch2 = []
txt1 = "The cat ate the big white"
txt2 = "The dog ran with the small"

batch2.append(torch.tensor(tokenizer.encode(txt1)))
batch2.append(torch.tensor(tokenizer.encode(txt2)))
batch2 = torch.stack(batch2, dim=0)
print(batch2)

# now GPT on batch2
out2 = model(batch2)
print("Input batch2:\n", batch2)
print("\nOutput shape:", out2.shape)
print(out2)

# %%
# generate text
# listing 4.8
def generate_text_simple(model, idx,
                         max_new_tokens, context_size): 
    # iteratively generate new tokens
    for _ in range(max_new_tokens):
        # only take a context_size window from the end
        idx_cond = idx[:, -context_size:]
        # don't calculate gradients, that is a waste here
        with torch.no_grad():
            logits = model(idx_cond)
        # look at last row only, because this gives logits for next token
        logits = logits[:, -1, :] 
        # do softmax, but don't really need to here as we are taking largest
        probas = torch.softmax(logits, dim=-1)
        # could do this next step directly on logits
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # add the token to the end of the token set
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# %%
# some batch2 adjustment
# extract the tokens in a simple way
first_entry = batch2[0,:]
print(first_entry)
print(first_entry.shape)
first_entry_copy = batch2[0,:].unsqueeze(0)
print(first_entry_copy)
print(first_entry_copy.shape)
    




# %%
# generate some
res = generate_text_simple(
    model=model,
    idx = batch2,
    max_new_tokens=2,
    context_size=10  
)
print(res)
print(res.shape)

# %%
