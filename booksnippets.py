# booksnippets.py
# Mark Bieda and from Build an LLM book

# %%
# LIBRARY INPUT
import torch

# %%
# make random set of embedding data
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
first_one.shape
first_one_fulldim.shape

# %%
