# text2vocab.py
# Mark Bieda
# 2025JUN11

# USAGE: python text2vocab.py textfile
# INPUT:
#   textfile is just a file with text in it. Will be read line by line.
#       characters besides letters, digits, and ordinary punctuation are removed
#   Note that text is split at spaces and ordinary punctuation
# OUTPUT:
#   outputfiles is hardcoded here in the code
#   one file is a JSON file of word:tokenvalue
#   one file is a JSON file of tokenvalue:word
# NOTES:
#   1. files are output in most frequent to less frequent word order
#       so the first entry is the most frequent "word"
#   2. "words" can be units of punctuation, although these are a minority


# %%
# READ LIBRARIES
import re
import string
import collections as collec
from collections import Counter
import json
import os
from pathlib import Path

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
# initialize variables
mydict = collec.defaultdict(int)

def find_repo_root(start: Path) -> Path | None:
    """
    Resolve repo root by walking upward until both `src/` and `notebooks/` exist.
    """
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "notebooks").exists():
            return p
    return None


def resolve_project_root() -> Path:
    """
    Match notebook-style path resolution:
    1) LLM_PROJECT_ROOT env override
    2) Colab drive default
    3) Search upward from script location, then cwd
    """
    env_root = os.environ.get("LLM_PROJECT_ROOT", "").strip()
    default_colab_root = Path("/content/drive/MyDrive/llm-from-scratch-drive")

    if env_root:
        return Path(env_root).expanduser().resolve()
    if default_colab_root.exists():
        return default_colab_root.resolve()

    search_starts = [Path(__file__).resolve(), Path.cwd().resolve()]
    for start in search_starts:
        root = find_repo_root(start)
        if root is not None:
            return root

    # Final fallback if repository markers are unavailable.
    return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# %%
filenm = "pubmed_abstracts_80K_getv5_english_ABSTRACTS.txt"
# filenm = "testfile.txt"
fullnm = DATA_DIR / filenm

# %%
# read file line by line and fill in frequency dictionary
maxlines= 50
with open(fullnm) as tempfile:
    for i, line in enumerate(tempfile):
        #if i>=maxlines:
        #    break
        tokens = tolower(killspaces(simplesplit(cleantext(line))))
        #print(line, end="")
        #splitLST = re.split(r"([\s\.,])", line)
        #splitLST_nospc = [k for k in splitLST if k not in [" ","","\n"]]
        #print(splitLST_nospc)
        for k in tokens:
            mydict[k] +=1


# %%
# see the dictionary, which is (word, occurrences) 
maxitems = 30 
for i, (key, value) in enumerate(mydict.items()):
    if i >= maxitems:          # after maxitems pairs, stop
        break
    print(f"{key}: {value}")
# print(mydict)

# %%
# sort it
sorted_items = sorted(mydict.items(), key=lambda item: item[1], reverse=True)

# %%
mydict_sorted = {k:v for k,v in sorted_items}
# print(mydict_sorted)

# %%
# create numeric token assignments
indexLST =range(0, len(mydict_sorted.keys()))
# index_word = {k:v for zip(indexLST,mydict_sorted.keys())}
index_word={}
word_index={}
for i,k in enumerate(mydict_sorted.keys()):
    index_word[i]=k
    word_index[k]=i
# print(f"{word_index=} , {index_word=}")

# %%
# add <UNK>
unkdict = {"<UNK>":len(word_index.keys())}
unkdict_rev = {len(word_index.keys()):"<UNK>" }
word_index = word_index | unkdict
index_word = index_word | unkdict_rev


# %%
# get top n by frequency
n=10
list(mydict_sorted.keys())[0:n]



# %%
# output to json
filenm = "pubmed_80Kv5_english_word2index.json"
fullnm = DATA_DIR / filenm
with open(fullnm,"w",encoding="utf-8") as f:
    json.dump(word_index,f, ensure_ascii=False)
    
filenm = "pubmed_80Kv5_english_index2word.json"
fullnm = DATA_DIR / filenm
with open(fullnm,"w",encoding="utf-8") as f:
    json.dump(index_word,f, ensure_ascii=False)


