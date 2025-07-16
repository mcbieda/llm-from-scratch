# simple_forward_transformer.py
# Mark Bieda, from Raschka book


# %%
# LIBRARY INPUT
import torch

# %%
# make random set of embedding data
# this makes a batch of data: batchsize # of examples of (sentence_tokens,d_in)
#  each "example" would be like the embedding of a single sentence
# See first_one below for a single example taken from this batch
# note use normal here
torch.manual_seed(123)
batchsize=3
sentence_tokens=4
d_in = 6
inputs = torch.randn(batchsize,sentence_tokens,d_in)

# %%
# print the random tensor
print(inputs)

# %%
# slice tensor
first_one = inputs[0,:,:]
first_one_fulldim = inputs[:1]

# %%

# %%
# simple attention class
# listing 3.1 from Raschka
# note this does not take a batch, but rather just the (T,d_in) tensor of a single example
# this is really simple - bare-bones -  scaled dot-product attention
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
# %%
# run simple attention class
# use first_one from above

myattention = SelfAttention_v1(6,2)
context_res = myattention(first_one)
print(context_res)
print(context_res.shape)



# %%
# listing 3.3
# note takes a batch so 3D tensor needed
# uses nn.Linear, which is a linear layer
# a lot of the complexity here is from using the full 3D tensor instead of single examples
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Create the mask - but is altered to -Inf in forward method below
        self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        ) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # Note x is (batch, T, d_in)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   
        # Fill in the masked values here with -Inf
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        # Note dropout is present unless model.eval() is run
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
    
# %%
# run causal attention class
# use first_one_fulldim from above; could use inputs
myattention2 = CausalAttention(6,6,4,0.5)
context_res2 = myattention2(first_one_fulldim)
print(context_res2)
print(context_res2.shape)
print(f"Training status for myattention2 is {myattention2.training=}")

# %% multihead
# listing 3.5 multihead attention combined
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(                                             
            b, num_tokens, self.num_heads, self.head_dim                    
        )                                                                   

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        # Note dropout is present unless model.eval() is run
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec
    

# %% run data
# run data
myattention3 = MultiHeadAttention(6,6,4,0.5,2)
# myattention3.eval()  # this will eliminate dropout
context_res3 = myattention3(first_one_fulldim)
print(context_res3)
print(context_res3.shape)

