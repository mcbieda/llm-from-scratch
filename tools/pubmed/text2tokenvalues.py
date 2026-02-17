# text2tokenvalues.py
# Mark Bieda
# 2025JUL02

# USAGE:
# python text2tokenvalues.py vocabfile textfile
# INPUT: 
#   vocabfile: a JSON vocabulary file (word:tokenvalue format)
#   textfile: a single line text file
# OUTPUT: each word in the text file and the corresponding token value from vocab
# Note that OOV words are replaced with <UNK> in output
# vocab files are JSON files from text2vocab

# GOAL: to map text into token values, taking the vocab and text as files from command line

# %%
# load modules
import json
import sys
import string
import re

# %%
# Get command line arguments
# should be vocabfilenm textfilenm
script, vocabfilenm, textfilenm = sys.argv

# %%
# FUNCTIONS
def cleantext(text: string):
    allowed_chars = r'\w' + re.escape(string.punctuation) + ' '
    cleaned = re.sub(f'[^{allowed_chars}]+', '', text)
    return cleaned
def simplesplit(text: string):
    import re
    pattern = rf"([{re.escape(string.punctuation)}]|\s)"
    results = re.split(pattern, text)
    return results
def killspaces(reslist: list):
    results = [item for item in reslist if item.strip()]
    return results
def getunique(x: list):
    xset = set(x)
    xlist = sorted(list(xset))
    return(xlist)
def tolower(x: list):
    return [a.lower() for a in x]


# %%
# Load vocab file
#filenm1 = "pubmed_80Kv5_english_index2word.json"
#datapath = "/home/mark/bio-llm/tokenizer"
# Path to the file (can be absolute or relative)
# json_path = Path("data/example.json")

try:
    with open(vocabfilenm,"r", encoding="utf-8") as f:
        word_index: dict = json.load(f)   # `data` is now a Python dictionary
except FileNotFoundError:
    print(f"{json_path} not found.")
except json.JSONDecodeError as err:
    print(f"Malformed JSON – {err}")
index_word={v: k for k, v in word_index.items()}
# Work with the resulting dict
# print(data)

# %%
# Load text file - assume one line
with open(textfilenm,"r",encoding="utf-8") as f:
    thistext = f.readlines()
print(thistext[0])

# %%
# decode text and print out
# %%
# test some text
#text02 = 'HER2-negative breast cancer involves chromatin changes.'
#text02_mod = text02.lower()
text_clean_lower = cleantext(thistext[0]).lower()
text_mod_list = killspaces(simplesplit(text_clean_lower))
token_numbers = [word_index.get(x, word_index["<UNK>"]) for x in text_mod_list]
token_words = [index_word[x] for x in token_numbers]
#print(word_index)
#print(token_words)
for k in text_mod_list:
    if k not in word_index.keys():
        k="<UNK>"
    print(k,word_index[k])
