# llm-from-scratch

A modular GPT-2-small experimentation repository based on Sebastian Raschka's *Build a Large Language Model (From Scratch)*, extended with biomedical domain-adaptive pretraining (DAPT), model comparison notebooks, and utilities for PubMed abstract collection.

## What this demonstrates

- Transformer architecture implementation and training workflow in PyTorch
- Domain-adaptive pretraining on biomedical text
- Model comparison through prompt behavior, parameter-difference analysis, and validation-loss evaluation
- Controlled hybrid-model and weight-transplantation experiments
- Ability to move beyond tutorial reproduction into experiment design and interpretation

## Technical skills shown

- PyTorch
- Transformer / GPT-2 architecture
- Tokenization and pretrained-weight loading
- Training loops, checkpointing, and validation
- Experimental design for LLM behavior analysis
- Model introspection via parameter and prompt-level comparisons

## What this repo is

This project has two main purposes:

1. Provide a cleaner GPT-2-small learning and experimentation codebase.
2. Study what changes after domain-adaptive pretraining on biomedical abstracts.

Compared with a chapter-oriented learning repo, this codebase is organized under `src/llm_from_scratch/` by functional role so it is easier to inspect, modify, and reuse.

## Why this is not just a tutorial reproduction

This repository started from the ideas and baseline code structure in Raschka's book, but the main value here is the additional work built around that base:

- a refactored modular project layout
- notebook-based training and inspection workflows
- biomedical DAPT experiments
- comparison of base vs DAPT behavior
- hybrid-model swaps to test where the DAPT advantage lives in the network
- a technical report that summarizes the results and limitations

## Main result

The main analytical result is that DAPT changed model behavior in a distributed way.

- Token embeddings changed, but simple embedding similarity checks alone do not explain the behavioral shift.
- The clearest effect appears in context-sensitive next-token probabilities for biomedical prompts.
- Weight-transplantation experiments suggest that the DAPT advantage does not come from swapping in the DAPT embedding table alone.
- The full DAPT model outperforms hybrid models that transplant only selected components.

For the detailed write-up, see `reports/DAPT_alteration_analysis.md`.

## Results snapshot

Primary analysis notebook:

- `notebooks/compare_base_vs_dapt.ipynb`

Technical report:

- `reports/DAPT_alteration_analysis.md`

Selected outputs from the comparison:

| Result | Base | DAPT |
| --- | ---: | ---: |
| Mean embedding cosine similarity | 0.9707 | n/a |
| `P('2' | cancer context ending in HER)` | 55.1% | 96.4% |
| `P('BB' | biomedical ER context)` | 0.1% | 57.0% |
| Validation loss on sampled biomedical subset | 2.9581 | 2.5963 |

Key hybrid-model result:

- `base_encode_dapt_model`: base model with DAPT token embeddings replacing base embeddings.
- `base_encode_plusothers_dapt_model`: base model with DAPT token embeddings, DAPT positional embeddings, and a few selected layer-norm shifts.
- `base_encode_pos_trf0_11_model`: base model with DAPT token embeddings, positional embeddings, and full first and last transformer blocks.
- `base_encode_pos_trf5_6_model`: base model with DAPT token embeddings, positional embeddings, and full tranformer blocks 5 and 6.
- `dapt_encode_base_model`: DAPT model with token embeddings replaced by base embeddings.

| Model | Subset validation loss |
| --- | ---: |
| `dapt_model` | 2.5963 |
| `base_encode_pos_trf0_11_model` | 2.8796 |
| `dapt_encode_base_model` | 2.9055 |
| `base_model` | 2.9581 |
| `base_encode_pos_trf5_6_model` | 3.1716 |
| `base_encode_plusothers_dapt_model` | 3.2686 |
| `base_encode_dapt_model` | 3.3036 |

Interpretation:

- DAPT embeddings alone are not sufficient.
- Replacing the DAPT model's embeddings with base embeddings reduces much of the DAPT advantage, but does not remove it entirely.
- A broader set of coordinated changes across embeddings, norms, attention-related weights, and transformer blocks is needed to recover DAPT behavior.

## Best places to review quickly

If you are scanning this repo for technical signal, start here:

1. `reports/DAPT_alteration_analysis.md`
2. `notebooks/compare_base_vs_dapt.ipynb`
3. `src/llm_from_scratch/models/`
4. `src/llm_from_scratch/training/`

## Main notebooks

The fastest way to review the repository is to start with the notebooks:

- `notebooks/gpt2_basic_training_small.ipynb`
- `notebooks/gpt2_basic_training_abstracts.ipynb`
- `notebooks/compare_base_vs_dapt.ipynb`

These notebooks are committed with outputs and show the main training and analysis workflows.

## Repo structure

- `src/llm_from_scratch/`
  Refactored GPT-2-small code organized by model, training, dataloading, configs, and analysis utilities.

- `notebooks/`
  Executed notebooks for training, evaluation, and comparison experiments.

- `reports/`
  Longer-form analysis write-ups, including the DAPT technical report.

- `tools/pubmed/`
  Utilities for query-based bulk PubMed abstract download.

## Installation

Basic setup:

```bash
git clone <your-repo-url>
cd llm-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- Some large external data files are not included in the repository.
- OpenAI GPT-2-small pretrained weights are not included.
- For training experiments, GPU execution is strongly preferred over CPU.

## Running the notebooks

A typical workflow is:

1. Create and activate a virtual environment.
2. Install dependencies.
3. Open a notebook from `notebooks/`.
4. Run cells in order.

This repo also works well in Google Colab if the project is copied into Google Drive and run with a GPU runtime.

## Scope

This repository currently focuses on:

- GPT-2-small model-building and training workflows
- biomedical domain-adaptive pretraining
- analysis of base-vs-DAPT behavior
- biomedical abstract collection utilities

Not currently integrated into this refactored repo:

- later-book classifier and instruction-tuning workflows
- larger GPT-2 sizes wired into the same workflow
- broader sampling and inference utilities

## Attribution

This project was built by working through Sebastian Raschka's *Build a Large Language Model (From Scratch)* in detail. A substantial portion of the core model-building ideas and baseline code comes from the book's published code.
