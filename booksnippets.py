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
# make a tensor
simple_tensor = torch.tensor([1,2,3,4])
print(simple_tensor)
print(f"Shape of simple_tensor is {simple_tensor.shape}")
simple_2D =simple_tensor.view(2,2)
print(simple_2D)
simple_2D_2 = simple_tensor.view(4,1)
print(simple_2D_2)



# %%
# slice tensor
first_one = inputs[0,:,:]
first_one_fulldim = inputs[:1]

# %%
first_one.shape
first_one_fulldim.shape

# %%
# get second tensor element from inputs
second_one = inputs[1,:,:]
second_one_fulldim = inputs[1:2]

#
