# chapter5_initial.py
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
# generate text as token ids and probas
# listing 4.8
def generate_text_simple(model, idx,
                         max_new_tokens, context_size): 
    # idx is the tokenized batch dim (B,T)
    # iteratively generate new tokens
    for _ in range(max_new_tokens):
        # only take a context_size window from the end
        idx_cond = idx[:, -context_size:]
        # don't calculate gradients, that is a waste here
        with torch.no_grad():
            logits = model(idx_cond)
        # look at last row only in each batch, because this gives logits for next token
        logits = logits[:, -1, :] 
        # do softmax, but don't really need to here as we are taking largest
        probas = torch.softmax(logits, dim=-1)
        # could do this next step directly on logits
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # add the token to the end of the token set
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def get_next_token_probas(model, idx, context_size, temperature=0.0): 
    # idx is the tokenized batch dim (B,T)
    # text can be converted to idx simply via text_to_token_ids()
    # only take a context_size window from the end
    idx_cond = idx[:, -context_size:]
    # don't calculate gradients, that is a waste here
    with torch.no_grad():
        logits = model(idx_cond)
    # look at last row only in each batch, because this gives logits for next token
    logits = logits[:, -1, :] 
    if temperature > 0.0:
        logits = logits/temperature
    # softmax probabilities (probas)
    probas = torch.softmax(logits, dim=-1)
    return probas





# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# chapter 5 actual
# !!!!!!!!!!!!!!!!!!!!!!!!!!!

# %%
# token untilities
# listing 5.1
import tiktoken
# from chapter04 import generate_text_simple

def text_to_token_ids(text, tokenizer):
    # text must be a string
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def token_ids_to_list(token_ids, tokenizer):
    return [tokenizer.decode_single_token_bytes(tid).decode("utf-8", errors="replace") for tid in token_ids]




# %%
# FUNCTION - generate and print sample
# important: had to hardcode the context_size here because wasn't working to get from model
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    #context_size=model.pos_embed.weight.shape[0]
    # MB temp override, due to error
    context_size = 256
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# %%
# FUNCTION: generate tokenids with topK and temperature
def generate_topk_temp(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    # idx is a list? or tensor? of tokens
    # returns token_ids, not text
    for _ in range(max_new_tokens):
        idx_cond =  idx[:, -context_size:]
        with torch.no_grad():
            logits =model(idx_cond)
        logits = logits[:, -1, :] # can handle a batch
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

    



# %%
# !!!!!!!!!!!!!!! LOAD GPT-2 OPENAI SETTINGS AND PARAMS !!!!!!!!!!!!!!!!!!!!!
# note this is where different models can be loaded
# currently, am only using model state_dict so can do inference

# these values can be adjusted!!
# I am using the pickled ones here for my convenience

# LOAD SETTINGS and PARAMS
import pickle
filepath = "/home/markb/llm-from-scratch/data/"

# Load settings
# note this is for the 124M one - need to rename
filenm ="gpt2_openai_settings.pkl"
fullnm = filepath+filenm
with open(fullnm, "rb") as f:
    settings = pickle.load(f)

# Load params
# note this is for the 124M one - need to rename
filenm ="gpt2_openai_params.pkl"
fullnm = filepath+filenm
with open(fullnm, "rb") as f:
    params = pickle.load(f)

# traditional model load, not used here
#filepath = "/home/markb/llm-from-scratch/output/"
#descripstr ="lrp0004wdp15"
#filenm = "model_and_optimizer_" + descripstr +".pth"
#fullnm = filepath + filenm
#lrval = 0.0004
# wdval = 0.15

# %%
# CONFIGURE FOR GPT-2 OPENAI PARAMS 124M
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

# %%
# SETUP OPENAI MODEL  
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

# %%
# LOAD WEIGHTS INTO GPT-2 MODEL - I
import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# %%
# LOAD WEIGHTS INTO MODEL - II
device = "cpu"
load_weights_into_gpt(gpt, params)
gpt.to(device)

# %%
# RUN OPENAI GPT-2 with openAI weights
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
# thistext = "Every effort moves you" # this text is quite good
thistext = "Leo is the name of my dog. He doesn't bite"
token_ids = generate_topk_temp(
    model=gpt,
    idx=text_to_token_ids(thistext, tokenizer).to(device),
    max_new_tokens=100,
    context_size=NEW_CONFIG["context_length"],
    top_k=10,
    temperature=1.0
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



# %%
# !!!!!!!!!!!!!!!!! ONLY RUN ABOVE THIS FOR OPENAI weights!!!!!!!!!!!!!!!


# %%
# LOAD PART 2 - get model going!!
# basic load of model/optimizer
# only setup for inference, don't put params in optimizer
device = "cpu"
# checkpoint = torch.load(fullnm, map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
#optimizer = torch.optim.AdamW(model.parameters(), lr=lrval, weight_decay=wdval)
#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train();


# %%
# !!!!!!!!!!!!!!RUN MODEL ON TEXT (best token returned) !!!!!!!!!!!!!!!!
torch.manual_seed(123)
start_context = "even through the prism of"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print(f"Input text: {start_context}")
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %%
# !!!!!!!!!!!!!!!!!! JUST EXPERIMENTS BELOW THIS POINT !!!!!!!!!!!!!

# %%
# EXPERIMENT: probas
# uses new function get_next_token_probas
torch.manual_seed(123)
start_context = "even through the prism of"
tokenizer = tiktoken.get_encoding("gpt2")


# this will get the probabilities across the vocab for next token
# playing with temperature here
#  temp = 1.0 is base model; temp = 2.0 is dramatic! temp = 0.5 really pushes toward one
temperature = 0.5
next_token_probas = get_next_token_probas(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=temperature
)




# get the top one
best_token_id = torch.argmax(next_token_probas)
best_token = tokenizer.decode([best_token_id])
print(f"best_token_id: {best_token_id}, token: {best_token}")

# get the top 3 and show probabilities
# note really big effect of temperature above: 0.5 to 1 to 2 is enormous
torch.manual_seed(123)
k = 10
topk_values, topk_indices = torch.topk(next_token_probas, k=k)
# note: improvement: could make a loop here to make nicer output
# I don't like how the tokenizer.decode adjusts the output, would rather have a list
best_tokens = token_ids_to_list((topk_indices.squeeze()).tolist(), tokenizer)
print("OUTPUT of next token possibilities")
print(f"input string:\"{start_context}\", temp:{temperature}, num_possibilities: {k}")
print(f"tokenids: {topk_indices}, tokens: {best_tokens}")
print(f", probs: {topk_values}")





# %%
# !!!!! RUN MODEL (text with topk and temperature)!!!!!!!!!!!!!
# enables playing with temp and topk settings

# generate_topk_temp(model, idx, max_new_tokens, context_size,
#             temperature=0.0, top_k=None, eos_id=None)

torch.manual_seed(123)
start_context = "even through the prism of"
idx = text_to_token_ids(start_context, tokenizer)
max_new_tokens = 25
context_size = 256
temperature = 0.5
top_k=10
eos_id = None
res = generate_topk_temp(model, idx, max_new_tokens, context_size,
                         temperature, top_k,eos_id)
res_text = token_ids_to_text(res,tokenizer)
print(res_text)

#

# %%
