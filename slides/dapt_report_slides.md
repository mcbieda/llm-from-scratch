---
marp: true
theme: default
math: mathjax
paginate: true
size: 16:9
---

<style>
section {
  font-size: 27px;
  padding-top: 35px;
  line-height: 1.2;
}
h1 { font-size: 40px; }
h2 { font-size: 35px; }
h3 { font-size: 31px; }
li, p { line-height: 1.15; }
table { font-size: 20px; }
section.small { font-size: 24px; }
section.small table { font-size: 16px; }
section.tiny { font-size: 20px; }
section.tiny table { font-size: 13px; }
.logic-text { font-size: 0.75em; line-height: 1.1; }
code { font-size: 0.9em; }
</style>

# Where Does Biomedical DAPT Change GPT-2 Small?
## Evidence from Weight Differences, Prompt Probes, and Transplant Experiments

_Slide deck (Marp) generated from report_ 

Mark Bieda 
Date: 2026-04-18; update 2026-06-04

---

## What this is
- Domain-adaptive pretraining (DAPT) on a biomedical abstract corpus, starting from base GPT-2 small (124M)

- Why GPT-2 small (124M)?
	- small enough to provide efficient platform for limited scale project
	- expect minimal data leakage and should see clear training signal
		- Pubmed abstracts do not appear to be part of training set
		- base model trained on data from 2019 and before
	
	
	
	
- Goals: 1) show DAPT effectiveness with limited DAPT and 2) **where the gain lives**
- Key tools:
  - validation loss
  - prompt probes
  - parameter-difference summaries
  - **weight transplantation** (strongest evidence)

---

## Summary

- DAPT was successful
	- loss on biomedical validation set decreased
	- shift toward appropriate biomedical next token predictions
	
- Biomedical DAPT produces a **distributed, coordinated change**.
	- DAPT produced changes across a range of blocks/layers in the model
	- Token embeddings matter, but **are not sufficient**
	- Embedding swap into base model does not produce DAPT model
	- Partial recovery requires embeddings **plus** other DAPT components / blocks

---

## Experimental setup (models)
Two primary models:
- `base_model`: GPT-2 small (base OpenAI weights)
- `dapt_model`: continued training from base weights (DAPT)

DAPT data:
- PubMed abstracts (query on cancer + ERBB2/HER2/EGFR), 2005–2025
- DAPT “training corpus”: 1,679,880 GPT‑2 tokens (≈ 7.27M characters)

Evaluation:
- Separate validation corpus: 671,020 tokens (≈ 2.90M characters)
- Some hybrid comparisons use a 2.5% subset of validation corpus for speed

- Limited DAPT regimen details (high level)
	- 1 epoch
	- batch size: 2
	- stride: 1024

---
# DAPT SHIFTS MODEL BEHAVIOR
---

## Validation loss after DAPT 
Full validation set:
- Loss: **2.9233 → 2.5513**
- Drop: **0.372 nats** (≈ **12.7%** relative)
- Perplexity decrease: **≈ 31.1%**

---

## Prompt Probes Show Shift Toward Biomedical Completions
DAPT shifts next-token behavior strongly **in biomedical contexts**:
- `ER` in cancer context → strongly prefers biomedical continuation (`ERBB2`)
- Still respects context (ER = emergency room prompt does not collapse to ERBB2)
- Some spillover drift on general prompts (pet context)



| Prompt / context | Expected token / continuation | Base model | DAPT model | Interpretation |
| --- | --- | ---: | ---: | --- |
| Cancer context ending with `HER` | `'2'` | 55.1% | 98.8% | DAPT strongly enhances a biomedical completion that the base model already partially favors. |
| Cancer context ending with `ER` and designed to continue toward `ERBB2` | `'BB'` | 0.1% | 77.4% | DAPT strongly shifts the continuation toward biomedical terminology. |
| Emergency-room context ending with `ER` | Several possible continuations | top: `.` (31.7%) | top: `' and'` (14.5%) | DAPT changes the ranking, but context still prevents a collapse onto `ERBB2`. |
| Ordinary pet-context prompt ending in `dogs and` | `' cats'` | 94.0% | 67.7% | General-language behavior remains recognizable, but there is noticeable biomedical drift. |

---
# LOCALIZATION OF CHANGES
---

## Parameter Differences (by tensor) 

### Top altered tensors:

| Tensor | Relative L2 |
| --- | ---: |
| `trf_blocks.11.norm2.shift` | 0.2601 |
| `tok_emb.weight` | 0.2483 |
| `trf_blocks.0.norm1.shift` | 0.2469 |
| `trf_blocks.10.norm2.shift` | 0.1673 |
| `trf_blocks.4.norm2.shift` | 0.1456 |
| `trf_blocks.0.att.W_value.weight` | 0.1410 |
| `trf_blocks.9.norm2.shift` | 0.1399 |


**Relative L2 (per tensor):**

$\text{Relative L2} = \frac{\lVert W_{\text{DAPT}} - W_{\text{base}} \rVert_2}{\lVert W_{\text{base}} \rVert_2 + \varepsilon}$


---

## Parameter-Difference Analysis (Block Level, analysis method 1)

_top 5 displayed here_
| Block | Simple block mean relative L2 | Simple block max relative L2 |
| --- | ---: | ---: |
| `tok_emb` | 0.2483 | 0.2483 |
| `pos_emb` | 0.0838 | 0.0838 |
| `trf_blocks.0` | 0.0735 | 0.2469 |
| `trf_blocks.11` | 0.0728 | 0.2601 |
| `trf_blocks.5` | 0.0698 | 0.1277 |


Simple Block Mean Relative L2:

$\text{Simple Block Mean Relative } L_2 = \frac{1}{N}\sum_{i=1}^{N}\frac{\left\| W_{\mathrm{DAPT}}^{(i)} - W_{\mathrm{base}}^{(i)} \right\|_2}{\left\| W_{\mathrm{base}}^{(i)} \right\|_2 + \varepsilon}$

---
<!-- _class: tiny -->
## Parameter-Difference Analysis (Block Level, analysis method 2)

_top 5 displayed here_
| Block | Aggregated block relative L2 |
| --- | ---: |
| `tok_emb` | 0.2483 |
| `pos_emb` | 0.0837 |
| `trf_blocks.5` | 0.0812 |
| `trf_blocks.6` | 0.0800 |
| `trf_blocks.7` | 0.0797 |




Aggregated Block Relative L2:

$\text{Aggregated Block Relative L2} =
\frac{\sqrt{\sum_i \lVert W^{(i)}_{\text{DAPT}} - W^{(i)}_{\text{base}} \rVert_2^2}}{\sqrt{\sum_i \lVert W^{(i)}_{\text{base}} \rVert_2^2} + \varepsilon}$

Note: this weights _tensor elements_ equally rather than weighting _tensors_ equally.

---

# ANALYSIS OF EMBEDDING CHANGES

_Token embedding consistently emerged above as a highly altered component. Also, due to weight-tying, token embeddings affected the embedding layer and the output layer, allowing large effects._

---

## Examination of embedding changes

_Across the full set of tokens, what is the min, mean, and max changes per token embedding?_

| Metric | Cosine similarity (base vs DAPT) |
| --- | ---: |
| Min | 0.8702 |
| Mean | 0.9684 |
| Max | 0.9993 |

These results are consistent with potentially important changes at the token embedding level.

---

## Embedding Comparisons

_Do embeddings from closely associated tokens in the training set come together?_

| Pair group | Token pair | Base similarity | DAPT similarity | Ratio (DAPT/base) |
| --- | --- | ---: | ---: | ---: |
| Biomedical-related | `' ER'` vs `'BB'` | 0.2552 | 0.2638 | 1.0336 |
| Biomedical-related | `' HER'` vs `'2'` | 0.2598 | 0.2653 | 1.0213 |
| Biomedical-related | `' kin'` vs `'ase'` | 0.2703 | 0.2714 | 1.0044 |
| Biomedical-related | `' EG'` vs `'FR'` | 0.2568 | 0.2434 | 0.9481 |
| Comparator | `'HER'` vs `' 2'` | 0.1710 | 0.1721 | 1.0061 |
| Comparator | `'EG'` vs `' FR'` | 0.2667 | 0.2304 | 0.8641 |
| Comparator | `'BB'` vs `'2'` | 0.2752 | 0.2702 | 0.9816 |
| Comparator | `' cat'` vs `' dog'` | 0.5498 | 0.5287 | 0.9617 |


'Similarity' is cosine similarity

---
# WEIGHT TRANSPLANTATION SUPPORTS DISTRIBUTED CHANGES HYPOTHESIS
---

## Weight Transplantation Experiments (Validation Loss)

<div class="logic-text">

Logic:
`base_model`, `dapt_model`: reference values
`dapt_embed_base_model`: embedding is important, add base embedding to dapt_model - should cause much worse loss
`base_embed_dapt_model`: embedding is important, add dapt embedding to base_model - should cause better loss
`base_embed_pos_trf0_11_model`: add top 4 most changed blocks (by simple weighting) to base model, should improve loss
`base_embed_pos_trf5_6_model`: add top 4 most changed blocks (by aggregated weighting) to base model, should improve loss
`base_embed_plusothers_dapt_model`: add dapt embedding and some other tensors to the base model, should improve loss

</div>


| Model | Validation loss (2.5% subset) |
| --- | ---: |
| **`dapt_model`** | 2.5855 |
| `base_embed_pos_trf0_11_model` | 2.8648 |
| `dapt_embed_base_model` | 2.9057 |
| **`base_model`** | 2.9581 |
| `base_embed_pos_trf5_6_model` | 3.1940 |
| `base_embed_plusothers_dapt_model` | 3.2675 |
| `base_embed_dapt_model` | 3.3093 |

A number of altered base models become _worse than base model_ with transplantation of dapt model components; but transplantation of the embedding + positional weights + first and last transformer blocks into base model leads to some improvement.

---

## Weight Transplantation Experiments (Prompt Behavior)

A biomedical prompt ending in `ER` is supplied and the correct next token is `BB` (to form `ERBB2`)

| Model | P(`'BB'` \|  biomedical `' ER'` context) |
| --- | ---: |
| `base_model` | <1% (0.000555) |
| `base_embed_plusothers_dapt_model` | <1% (0.000844) |
| `dapt_embed_base_model` | 37.2% (0.371979) |
| `dapt_model` | 77.4% (0.774406) |

_Massive increase in probability of `BB` for DAPT model, and this increase is greatly reduced by transplanting base model embedding layer into the dapt model_

---

## CONCLUSIONS

- Limited scope project with initial experiments and results shown here
- Limited biomedical DAPT created expected changes in model behavior, without collapse of normal behavior
- Loci of changes
	- Token embeddings appear important but not sufficient*
	- Clear indications of distributed and coordinated changes involving several layers
- **Potential Future Directions**
	- better evidence for distributed changes: do more transplantation experiments
	- better examination of DAPT effects on "normal text" model behavior: development and use of a normal text validation set
	- much deeper examination of layer representations of the DAPT changes; interpretability of these alterations
	- training from scratch with mixture of domain-specific data and generic text data
	- direct testing of whether training can be restricted to certain components only ("locking" other layers/components) to see if successful DAPT can be established this way
	
	
	
_*Note that because of weight-tying, token embeddings affected the embedding layer and the output layer_
	

---

# APPENDIX

---

<!-- _class: small -->
## A1. Top 10 most changed tokens by base-vs-DAPT embedding ...

| Rank | Token | Cosine similarity |
| --- | --- | ---: |
| 1 | `' You'` | 0.870204 |
| 2 | `' someone'` | 0.879040 |
| 3 | `' putting'` | 0.879239 |
| 4 | `' Don'` | 0.882754 |
| 5 | `' basically'` | 0.884582 |
| 6 | `' Matt'` | 0.885449 |
| 7 | `'If'` | 0.886136 |
| 8 | `' pretty'` | 0.886501 |
| 9 | `'You'` | 0.887201 |
| 10 | `' your'` | 0.887860 |

---

<!-- _class: small -->
## A2. Top 10 least changed tokens by base-vs-DAPT embedding...

| Rank | Token | Cosine similarity |
| --- | --- | ---: |
| 1 | `'antam'` | 0.999302 |
| 2 | `'illance'` | 0.999287 |
| 3 | `'edded'` | 0.999270 |
| 4 | `'itaire'` | 0.999259 |
| 5 | `'Aw'` | 0.999122 |
| 6 | `' congress'` | 0.999079 |
| 7 | `'ptions'` | 0.999056 |
| 8 | `'emaker'` | 0.999035 |
| 9 | `'%]'` | 0.999028 |
| 10 | `'mark'` | 0.999002 |

---

<!-- _class: tiny -->
## A3. Changed blocks (simple summary)

| Block | Mean relative L2 | Max relative L2 | Mean cosine similarity |
| --- | ---: | ---: | ---: |
| `tok_emb` | 0.248346 | 0.248346 | 0.973790 |
| `pos_emb` | 0.083753 | 0.083753 | 0.996992 |
| `trf_blocks.0` | 0.073543 | 0.246904 | 0.996391 |
| `trf_blocks.11` | 0.072841 | 0.260089 | 0.996180 |
| `trf_blocks.5` | 0.069812 | 0.127671 | 0.997597 |
| `trf_blocks.2` | 0.067052 | 0.137047 | 0.997792 |
| `trf_blocks.9` | 0.066633 | 0.139850 | 0.997828 |
| `trf_blocks.6` | 0.065370 | 0.135014 | 0.997941 |
| `trf_blocks.7` | 0.064789 | 0.121903 | 0.998029 |
| `trf_blocks.10` | 0.064318 | 0.167272 | 0.997726 |
| `trf_blocks.3` | 0.064316 | 0.108617 | 0.998022 |
| `trf_blocks.8` | 0.064096 | 0.112774 | 0.998113 |
| `trf_blocks.4` | 0.064057 | 0.145634 | 0.997872 |
| `trf_blocks.1` | 0.062857 | 0.119669 | 0.998135 |
| `final_norm` | 0.043623 | 0.049168 | 0.999872 |

---

<!-- _class: small -->
## A4. Submodule summary

| Submodule | Mean relative L2 | Max relative L2 |
| --- | ---: | ---: |
| `tok_emb` | 0.248346 | 0.248346 |
| `att.W_value` | 0.091979 | 0.141011 |
| `pos_emb` | 0.083753 | 0.083753 |
| `norm2` | 0.083445 | 0.260089 |
| `norm1` | 0.071555 | 0.246904 |
| `ff.layers.0` | 0.066398 | 0.088764 |
| `att.out_proj` | 0.063377 | 0.108617 |
| `ff.layers.2` | 0.059595 | 0.094519 |
| `att.W_query` | 0.055320 | 0.090683 |
| `att.W_key` | 0.052851 | 0.092317 |
| `final_norm` | 0.043623 | 0.049168 |
