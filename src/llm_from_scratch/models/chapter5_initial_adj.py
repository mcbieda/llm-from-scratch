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
# functions: forward model
# run make_tokenized_batch
# run setup_model
# run forward_model

def make_tokenized_batch(batch):
    # starting with list of sentences, returns tokenized list
    # batch is batch of sentences in text form dim: num of sentences
    tokenizer = tiktoken.get_encoding("gpt2")
    batchout = []
    for i in range(len(batch)):
        batchout.append(torch.tensor(tokenizer.encode(batch[i])))
    batchoutstack = torch.stack(batchout, dim=0)
    return batchoutstack

def setup_model(cfg):
    # cfg must be the dictionary with parameters for configuration
    # eg GPT_CONFIG_124M
    torch.manual_seed(123)
    model = GPTModel(cfg)
    return model

def forward_model(model, batch):
    # model is usually from setup_model
    # batch is usually from make_tokenized_batch
    model.eval()
    with torch.no_grad():
        res = model(batch)
    return res





                        


# %%
# testing GPT
#torch.manual_seed(123)

# KEY LINE: setup the model
#model = GPTModel(GPT_CONFIG_124M)

#model.eval()
#out = model(batch)
#print("Input batch:\n", batch)
#print("\nOutput shape:", out.shape)
#print(out)

# %%
# test my functions
# model = setup_model(GPT_CONFIG_124M)
# batch = ["The cat ate the little", "The dog ran with a"]
# batchtokenized = make_tokenized_batch(batch)
# print(batch)
# print(batchtokenized)
# out = forward_model(model, batchtokenized)
# print("Input batch:\n", batchtokenized)
# print("\nOutput shape:", out.shape)
# print(out)


# %%
# a  test
tokenizer = tiktoken.get_encoding("gpt2")
# batch2 = []
# txt1 = "The cat ate the big white"
# txt2 = "The dog ran with the small"

# batch2.append(torch.tensor(tokenizer.encode(txt1)))
# batch2.append(torch.tensor(tokenizer.encode(txt2)))
# batch2 = torch.stack(batch2, dim=0)
# print(batch2)

# # now GPT on batch2
# out2 = model(batch2)
# print("Input batch2:\n", batch2)
# print("\nOutput shape:", out2.shape)
# print(out2)

# %%
# generate text
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

# %%
# some batch2 adjustment
# extract the tokens in a simple way
# first_entry = batch2[0,:]
# print(first_entry)
# print(first_entry.shape)
# first_entry_copy = batch2[0,:].unsqueeze(0)
# print(first_entry_copy)
# print(first_entry_copy.shape)
    




# %%
# generate some
# res = generate_text_simple(
#     model=model,
#     idx = batch2,
#     max_new_tokens=2,
#     context_size=10  
# )
# print(res)
# print(res.shape)


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

# %%
# run the model on some text
# start_context = "The cat ate the little"
# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(start_context, tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONFIG_124M["context_length"]
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %%
# LOAD THE VERDICT
filenm = "/home/markb/llm-from-scratch/data/the-verdict.txt"
with open(filenm, 'r', encoding='utf-8') as f:
    text_data = f.read()

# %%
# check length of dataset and num characters and total tokens
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# %%
# setup train-test split
train_ratio=0.90
train_index = int(train_ratio *total_characters)
train_data = text_data[:train_index]
val_data = text_data[train_index:]

# %%
# dataloader from chapter 2
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# %%
# train and validate loader
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last=True,
    # shuffle during training so that batches are different in different epochs
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last=False,
    # no shuffle during training so that batches are same
    shuffle=False,
    num_workers=0
)

# %%
# loss function
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits=model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    # print(f"single batch loss is {loss}")  # added to look at it
    return loss  # scalar

# %%
# listing 5.2
# function to calculate loss
def calc_loss_loader(data_loader,model, device, num_batches=None):
    total_loss=0
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i< num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

# %%
# beginning loss check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# with torch.no_grad():
#     train_loss=calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)
# print("train loss:", train_loss)
# print("val loss:", val_loss)

# %%
# test dimensions
# note on output: there are only 5145 total tokens in the set; each batch is 512
# MATH around this

# total_tokens was calculated above; train_ratio set above
num_val_tokens = int(total_tokens * (1 - train_ratio))
num_train_tokens = int(total_tokens * train_ratio)
print(f"TOKEN MATH: int num train tokens: {num_train_tokens}, int num val tokens: {num_val_tokens}")
# each batch is 2 x 256 because 2 examples of 256. So total is 512 per batch
total_batch_size = 2 * 256 #  FIX: kill hardcoding
num_val_batch = int(num_val_tokens/total_batch_size)
num_train_batch = int(num_train_tokens/total_batch_size)
print(f"TOKEN MATH: int num train batches: {num_train_batch}, int num val batch: {num_val_batch}")

print("\nTrain loader:")
for i, (x, y) in enumerate(train_loader):
    print(f"batch: {i}, input shape: {x.shape}, output shape: {y.shape}")

print("\nValidation loader:")
for i, (x, y) in enumerate(val_loader):
    print(f"batch: {i}, input shape: {x.shape}, output shape: {y.shape}")


# %%
# LOOK AT ENTRIES IN LOADERS  
def loader_text_examine(loader,examplenum=0, tokenizer=tokenizer):
    # loader is like train_loader
    # examplenum is example num within batch
    # examplenum=0 will always exist

    # thisexample would be a tuple with (train, target)
    thisexample = loader.dataset[examplenum]
    thisexample_decode = token_ids_to_text(thisexample[0], tokenizer)
    print(thisexample_decode.replace("\n", " "))


# !!!!!!!!!!!!
# TRAINING PART
# !!!!!!!!!!!!

# %%
# functions: train model and evaluate model
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs,eval_freq,eval_iter,start_context, tokenizer):
    train_losses,val_losses, track_tokens_seen = [],[],[]
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
                
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

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
# SETUP FOR TRAIN MODEL - from scratch


IPEXflag = False
device = "cpu"
torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)
model.to(device)


# MB add for intel CPU only!
if IPEXflag:
    import intel_extension_for_pytorch as ipex
    model = model.to(memory_format=torch.channels_last)

# %%
# TRAIN MAIN LOOP - can be used after reloading model

# establishs IPEXflag
#IPEXflag = True
# MB add some timing
from datetime import datetime
start_time = datetime.now()
print(start_time)

lrval = 0.0004
weight_decayval = 0.1
print(f"learning rate:{lrval}, weight_decay:{weight_decayval}")
optimizer=torch.optim.AdamW(
    model.parameters(),
    # lr=0.0004 #  this is the suggested, but I am going to increase
    # weight_decay was 0.1, will make 0.02
    lr=lrval, weight_decay=weight_decayval
)

# added below by MB to use the IPEX package
if IPEXflag:
    model, optimizer = ipex.optimize(model,
                                    optimizer=optimizer,
                                    dtype=torch.float32,   # or torch.float32
                                    inplace=True)           # keep references

# back to usual code
num_epochs=20 # MB - this is usually 10! for this purpose
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs = num_epochs, eval_freq=5, eval_iter=5,
    start_context = "Every effort moves you", tokenizer=tokenizer
)
end_time =datetime.now()
print(f"end time is {end_time}")
total_time = end_time - start_time
print(f"total time = {total_time}")



# %%
# PLOT LOSS
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)



# %%
# !!!!!!!!!!!!!!!!!!!
# DO NOT RUN BELOW HERE EXCEPT TO SAVE OR LOAD A MODEL
# !!!!!!!!!!!!!!!!!!!


# %% save model
# save model and optimizer
# filepath = "/home/markb/llm-from-scratch/output/"
# descripstr ="lrp0004wdp15_20epoch"
# filenm = "model_and_optimizer_" + descripstr +".pth"
# fullnm = filepath + filenm
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     }, 
#     fullnm
# )


# %%
# LOAD MODEL AND OPTIMIZER

# these values must be adjusted!!
# filepath = "/home/markb/llm-from-scratch/output/"
# descripstr ="lrp0004wdp15"
# filenm = "model_and_optimizer_" + descripstr +".pth"
# fullnm = filepath + filenm
# lrval = 0.0004
# wdval = 0.15

# # basic load of model and optimizer params
# checkpoint = torch.load(fullnm, map_location=device)
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=lrval, weight_decay=wdval)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train();
# # %%
