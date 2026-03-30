# llm-from-scratch

A modular GPT-2-small experimentation repository based on Sebastian Raschka’s *Build a Large Language Model (From Scratch)*, extended with biomedical domain-adaptive pretraining (DAPT) and analysis notebooks focused on **what changes in model behavior after continued pretraining**.

## Overview

This repository has two main goals:

1. **Provide a cleaner, more reusable GPT-2-small learning and experimentation codebase**
   - based on the code and concepts from Raschka’s book
   - reorganized into a modular `src/` layout by functional role rather than by book chapter
   - structured to make training, inspection, modification, and comparison easier

2. **Demonstrate biomedical domain-adaptive pretraining (DAPT)**
   - continued pretraining of GPT-2-small (124M) on a corpus of biomedical abstracts
   - analysis of how DAPT changes model behavior, including embedding-level changes and next-token probability shifts under biomedical context

## Why this repo matters

This repository is intended as more than a chapter-by-chapter reimplementation.

It shows three things:

- **Code organization and engineering**
  - chapter-based instructional code reorganized into a reusable modular project layout

- **Model training and adaptation**
  - GPT-2-small workflows for training, inspection, and biomedical domain-adaptive pretraining

- **Model analysis**
  - concrete comparison of base vs DAPT behavior, including evidence that DAPT changed next-token behavior in targeted biomedical contexts

## Project highlights

- Modular code layout under `src/llm_from_scratch`
- Executed notebooks with outputs included
- GPT-2-small training and inspection workflows
- Biomedical DAPT experiments on PubMed-style abstracts
- Comparison notebook analyzing base vs DAPT behavior
- Bulk PubMed abstract download utilities under `tools/pubmed`

## Results snapshot: what changed with DAPT?

The main comparison notebook is:

- **`notebooks/compare_base_vs_DAPT.ipynb`**

This notebook compares the base GPT-2-small OpenAI weights with a checkpoint after domain-adaptive pretraining on biomedical abstracts.

### Main finding

The most important result was **not** large movement in token embeddings alone.  
The stronger effect appeared in **context-sensitive next-token probabilities**, which is the behavior expected from successful domain-adaptive pretraining.

### Table 1. Embedding similarity summary

Across all 50,257 token embeddings, cosine similarity between the base and DAPT embedding vectors was:

| Metric | Cosine similarity |
|---|---:|
| Min | 0.884 |
| Mean | 0.971 |
| Max | 0.999 |

This suggests that token embedding changes were real but generally modest at the global level.  

### Table 2. DAPT does not drive biomedical embeddings to be more similar  

One hypothesis is that embeddings of tokens involved in the biomedical corpus would become more similar, but this was not observed:  

| word | tokenid1 | token1 | tokenid2 | token2 | cos_sim (base) | cos_sim (DAPT) | ratio (DAPT/base) |
|---|---:|---|---:|---|---:|---:|---:|
| **biomedical related** |  |  | |  | | |  |
| ' ER' vs 'BB' | 13793 | `' ER'` | 15199 | `'BB'` | 0.255247 | 0.260255 | 1.019622 |
| ' HER' vs '2' | 24906 | `' HER'` | 17 | `'2'` | 0.259792 | 0.263318 | 1.013573 |
| ' kin' vs 'ase' | 18967 | `' kin'` | 589 | `'ase'` | 0.270267 | 0.271175 | 1.003361 |
| ' EG' vs 'FR' | 41513 | `' EG'` | 10913 | `'FR'` | 0.256762 | 0.250341 | 0.974991 |
| **comparators** |  |  | |  | | |  |
| 'HER' vs '2' | 16879 | `'HER'` | 362 | `' 2'` | 0.171015 | 0.178956 | 1.046435 |
| 'EG' vs 'FR' | 7156 | `'EG'` | 8782 | `' FR'` | 0.266662 | 0.237642 | 0.891174 |
| 'BB' vs '2' | 15199 | `'BB'` | 17 | `'2'` | 0.275225 | 0.272602 | 0.990468 |
| ' cat' vs ' dog' | 3797 | `' cat'` | 3290 | `' dog'` | 0.549790 | 0.535333 | 0.973704 |

Note tokenids are from GPT-2 tokenizer from tiktoken package

### Table 3. Selected behavioral shifts after DAPT (Prompts in Appendix at end of README)

| Prompt / context | Expected Token / continuation | Base model | DAPT model | Interpretation |
|---|---|---:|---:|---|
| **cancer context prompts** | |  | | |
| cancer prompt ending with `HER` | `'2'` | 55.1% | 96.4% | Base model correct, but stronger biomedical association after DAPT |
| cancer prompt that ends with ` ER` (and should eventually lead to `ERBB2`) | `'BB'` | 0.1% | 57.0% | DAPT shifted continuation behavior strongly toward biomedical terminology |
| **control context prompts** | |  | | |
| emergency room as ER prompt that ends with ` ER` | several possible | top: `.` (3.9%) | top: `,` (2.5%) | DAPT did not override context pointing toward ER as "emergency room" |
| ordinary pet-context prompt ending in `dogs and` | `' cats'` | 94.0% | 72.3% | General language behavior remained substantially intact |

### Interpretation

The comparison suggests:

- **global embedding drift was modest**
- **behavioral differences were larger at the output level**
- **DAPT made biomedical continuations much more likely in relevant contexts**
- **the model still retained broad general-language behavior on ordinary prompts**

That is the main technical takeaway from this project.

## What is original in this repo

This project builds on the published code from Raschka’s book, but differs from the book/repo in several important ways:

### 1. Functional repo organization

Raschka’s material is organized primarily by chapter for instructional purposes.  
This repository reorganizes the code by logical role, making it easier to inspect, extend, and reuse.

### 2. Biomedical DAPT workflow

This repository adds a workflow for domain-adaptive pretraining on biomedical abstracts, including training notebooks and supporting utilities.

### 3. Base-vs-DAPT analysis

This repository adds comparison notebooks focused on **what changed after DAPT**, including:

- token embedding comparisons
- selected token-pair checks for biomedical terms
- next-token probability comparisons under control and biomedical prompts

## Main notebooks

The fastest way to review the project is to open the notebooks in `notebooks/`. These notebooks are committed as executed notebooks with outputs included.

### Recommended starting points

- **`gpt2_basic_training_small.ipynb`**  
  Basic GPT-2-small training workflow

- **`gpt2_basic_training_abstracts.ipynb`**  
  Biomedical domain-adaptive pretraining workflow

- **`compare_base_vs_DAPT.ipynb`**  
  Analysis of model differences between the base GPT-2-small model and the DAPT checkpoint

## Repo structure

- **`notebooks/`**  
  Executed notebooks for training, evaluation, and analysis

- **`src/llm_from_scratch/`**  
  Refactored source code organized by functional role

- **`tools/pubmed/`**  
  Utilities for query-based bulk PubMed abstract download

## Configuration

Model and run settings are intended to be easy to inspect and modify.

### Option 1: inspect configuration in a notebook

Early in a notebook, examine the `cfg` dictionary.

- `cfg` contains overall run configuration
- `cfg["model_config"]` contains model-specific parameters

### Option 2: inspect the config file directly

See:

- `src/llm_from_scratch/configs/gpt2small_config.py`

`RUN_CONFIG` is loaded into the notebooks and defines both model configuration and run configuration.

## Installation

Basic setup:

```bash
git clone <your-repo-url>
cd llm-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Notes

- The Intel acceleration package used in some early experiments is not installed by `requirements.txt`
- Some larger external files are not included in the repo
- OpenAI GPT-2-small pretrained weights are also not included

## Running the notebooks

After installation, open a notebook from `notebooks/` and run it.

A basic workflow is:

1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Open and run one of the notebooks in `notebooks/`

For small experiments, local CPU execution is possible, but a GPU is much faster.

## Using Google Colab / Google Drive

This project also works well in Google Colab.

### Suggested workflow

1. Clone the repository locally
2. Copy the repo into a folder in Google Drive
3. Open a notebook from `notebooks/` in Colab
4. Select a GPU runtime
5. Run the notebook

This is substantially faster than local CPU execution for training experiments.

## PubMed abstract download

For query-based bulk PubMed abstract download, see:

- `tools/pubmed/`
- `tools/pubmed/README.md`

## Current scope

This repository currently focuses on the core model-building and training workflow up to roughly the end of Chapter 5 of Raschka’s book, plus the biomedical DAPT extension and analysis work described above.

Not currently integrated into this refactored repo:

- Chapter 6 classifier code
- Chapter 7 instruction-tuning code
- Additional GPT-2 model sizes such as 355M wired into the current workflow
- Probabilistic next-token sampling utilities

Some of this functionality has been implemented separately during learning, but has not yet been migrated into the current refactored structure.

## Attribution

This project was built by working through Sebastian Raschka’s *Build a Large Language Model (From Scratch)* in detail.

A substantial portion of the core model-building code and concepts comes from the book’s published code. My main contributions in this repository are:

- reorganizing the code into a modular repo structure
- designing the notebook-driven training workflow
- defining the current configuration setup
- adding biomedical DAPT experiments
- adding analysis of base vs DAPT model behavior
- adding utilities related to biomedical abstract collection

## How to review this repo quickly

If you want the fastest overview of what is here:

1. Open the notebooks in `notebooks/`
2. Start with:
   - `gpt2_basic_training_small.ipynb`
   - `gpt2_basic_training_abstracts.ipynb`
   - `compare_base_vs_DAPT.ipynb`
3. Then inspect the code under `src/llm_from_scratch/`
