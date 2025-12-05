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
# a  test
tokenizer = tiktoken.get_encoding("gpt2")


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
start_context = "The cat ate the little"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

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
model.to(device)
with torch.no_grad():
    train_loss=calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("train loss:", train_loss)
print("val loss:", val_loss)

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
IPEXflag = True
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
filepath = "/home/markb/llm-from-scratch/output/"
descripstr ="lrp0004wdp15_20epoch"
filenm = "model_and_optimizer_" + descripstr +".pth"
fullnm = filepath + filenm
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    fullnm
)


# %%
# LOAD MODEL AND OPTIMIZER

# these values must be adjusted!!
filepath = "/home/markb/llm-from-scratch/output/"
descripstr ="lrp0004wdp15"
filenm = "model_and_optimizer_" + descripstr +".pth"
fullnm = filepath + filenm
lrval = 0.0004
wdval = 0.15

# basic load of model and optimizer params
checkpoint = torch.load(fullnm, map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=lrval, weight_decay=wdval)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();
# %%
