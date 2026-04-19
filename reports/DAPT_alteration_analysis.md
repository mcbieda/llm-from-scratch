# Where Does Biomedical DAPT Change GPT-2 Small?
## Evidence from Weight Differences, Prompt Probes, and Transplant Experiments

## Summary

This report examines how domain-adaptive pretraining (DAPT) on a biomedical abstract corpus changes a GPT-2 small (124M) model initialized from the base OpenAI weights. The central question is not whether DAPT improves in-domain performance, but where that improvement appears to live in the model.

**The strongest result is that the DAPT gain is _not_ explained by token-embedding changes alone, but rather appears to be distributed across several layers.** Replacing the DAPT model's token embeddings with the base embeddings (DAPT + base embeddings) nearly completely removes the DAPT advantage, while transplanting only the DAPT token embeddings into the base model (base + DAPT embeddings) makes performance worse than the base model itself. In contrast, a base model with DAPT embeddings plus selected additional DAPT components partially recovers the in-domain gain. Taken together, these results support the view that biomedical DAPT produces a **distributed, coordinated change** across multiple components rather than a simple embedding-only shift.

The report also shows that prompt-conditioned next-token behavior shifts strongly in biomedical contexts after DAPT, while parameter-difference analyses nominate several candidate sites of adaptation, including the token embeddings, layer-norm shift vectors, attention value weights, and some transformer blocks. Embedding-similarity analyses detect real but modest global movement, but those analyses by themselves understate the much larger behavioral shift seen in prompt probes and transplantation experiments.

## Experimental Setup

GPT-2 small (124M) was originally trained with a cutoff date of 2019 and, from the described training corpus, does not appear to include pubmed abstracts. Hence, our biomedical corpus of pubmed abstracts, many of which are post-2019, should have been a minimal part of the training set of the base model. Therefore, we would expect some significant changes with DAPT.

The two primary models are GPT-2 small (124M) models:

- `base_model`: GPT-2 small using the base OpenAI weights
- `dapt_model`: a GPT-2 small model trained by continuing from the base model weights; DAPT details are found in the notebook `notebooks/gpt2_basic_training_abstracts.ipynb`

The DAPT corpus consists of PubMed abstracts (https://pubmed.ncbi.nlm.nih.gov/) retrieved with the query:


(cancer[Title/Abstract]) AND english[lang] AND (ERBB2[Title/Abstract] OR HER2[Title/Abstract] OR EGFR[Title/Abstract])


The abstracts were further limited to the year range 2005-2025. In the project files, the corpus used for DAPT is labeled as a `test` split, but functionally it serves here as the "DAPT training corpus". That corpus contains 7,267,986 characters (1,679,880 GPT-2-tokenizer tokens).

A separate validation corpus was used for evaluation. The full validation set contains 2,904,096 characters (671,020 tokens). Initial loss comparisons between the base and DAPT models used the full validation set. Later comparisons among hybrid models used a 2.5% subset of the same validation file for efficiency. In the original notebook outputs, the base and DAPT losses on that subset remained close to the full-set losses, supporting its use as a directional comparison set for the larger panel of hybrids.

Training was for a single epoch, with a batch size of 2 and a stride of 1024.


## DAPT Produces Significant Changes 

We began by examining whether our limited training set and training regimen shifted model behavior significantly.

### Validation Loss After DAPT
For the full validation set, loss decreased from 2.9233 to 2.5513 with DAPT, representing a drop of 0.372 nats or a relative decrease of 12.7%. In terms of perplexity, this represents a decrease of 31.1%.




### Prompt-Based Behavioral Changes

Prompt probes show that DAPT produces large, context-dependent behavioral shifts in biomedical settings. Selected examples are shown below.

| Prompt / context | Expected token / continuation | Base model | DAPT model | Interpretation |
| --- | --- | ---: | ---: | --- |
| Cancer context ending with `HER` | `'2'` | 55.1% | 98.8% | DAPT strongly sharpens a biomedical completion that the base model already partially favors. |
| Cancer context ending with `ER` and designed to continue toward `ERBB2` | `'BB'` | 0.1% | 77.4% | DAPT strongly shifts the continuation toward biomedical terminology. |
| Emergency-room context ending with `ER` | Several possible continuations | top: `.` (31.7%) | top: `' and'` (14.5%) | DAPT changes the ranking, but context still prevents a collapse onto `ERBB2`. |
| Ordinary pet-context prompt ending in `dogs and` | `' cats'` | 94.0% | 67.7% | General-language behavior remains recognizable, but there is noticeable biomedical drift. |

The clearest qualitative example is the `ER -> BB` prompt example in a cancer context. In that setting, the base model assigns very low probability to `'BB'` (0.01%), while the DAPT model makes it the top next predicted token at 77.4%. 

 Importantly, the DAPT model does not simply force biomedical continuations everywhere. In the emergency-room context, the surrounding words still constrain the model away from `ERBB2`. At the same time, the pet-context control shows that DAPT introduces some spillover outside the domain: after ` dogs and`, the DAPT model brings biomedical-flavored tokens such as `' patients'`, `' cancer'`, `' metast'`, `' tumor'`, and `' breast'` into its top predictions (data not shown; see compare_base_vs_dapt.ipynb). Hence, we appear to observe a strong biomedical specialization with some detectable general-context drift.


## Parameter-Difference Analysis

Given the clear signs that our training did affect the model, we next examined sites of modification. Parameter-difference analyses help identify candidate sites of adaptation.

The largest individual tensor changes were:

| Tensor | Relative L2 |
| --- | ---: |
| `trf_blocks.11.norm2.shift` | 0.2601 |
| `tok_emb.weight` | 0.2483 |
| `trf_blocks.0.norm1.shift` | 0.2469 |
| `trf_blocks.10.norm2.shift` | 0.1673 |
| `trf_blocks.4.norm2.shift` | 0.1456 |
| `trf_blocks.0.att.W_value.weight` | 0.1410 |
| `trf_blocks.9.norm2.shift` | 0.1399 |

Here, relative L2 is defined as:

```math
\text{Relative L2} = \frac{\lVert W_{\text{DAPT}} - W_{\text{base}} \rVert_2}{\lVert W_{\text{base}} \rVert_2 + \varepsilon}
```

At the block level, a simple mean-relative-L2 summary was calculated using the following equation:
$\text{Simple Block Mean Relative L2} = \frac{1}{N}\sum_{i=1}^{N}\frac{\left\lVert W^{(i)}_{\text{DAPT}} - W^{(i)}_{\text{base}} \right\rVert_2}{\left\lVert W^{(i)}_{\text{base}} \right\rVert_2 + \varepsilon}$

 Using this approach, the most changed components as:

| Block | Simple block mean relative L2 | Simple block max relative L2 |
| --- | ---: | ---: |
| `tok_emb` | 0.2483 | 0.2483 |
| `pos_emb` | 0.0838 | 0.0838 |
| `trf_blocks.0` | 0.0735 | 0.2469 |
| `trf_blocks.11` | 0.0728 | 0.2601 |
| `trf_blocks.5` | 0.0698 | 0.1277 |
| `trf_blocks.2` | 0.0671 | 0.1370 |

At the submodule level, the largest changes were concentrated in:

- `tok_emb`
- `att.W_value`
- `pos_emb`
- `norm2`
- `norm1`

An alternative aggregated block-level metric used the following equation:
$\text{Aggregated Block Relative L2} = \frac{\sqrt{\sum_i \lVert W^{(i)}_{\text{DAPT}} - W^{(i)}_{\text{base}} \rVert_2^2}}{\sqrt{\sum_i \lVert W^{(i)}_{\text{base}} \rVert_2^2} + \varepsilon}$



Importantly,this weights tensor elements equally rather than tensors equally. It produced a somewhat different ranking:

| Block | Aggregated block relative L2 |
| --- | ---: |
| `tok_emb` | 0.2483 |
| `pos_emb` | 0.0837 |
| `trf_blocks.5` | 0.0812 |
| `trf_blocks.6` | 0.0800 |
| `trf_blocks.7` | 0.0797 |
| `trf_blocks.3` | 0.0779 |
| `trf_blocks.8` | 0.0775 |
| `trf_blocks.4` | 0.0773 |
| `trf_blocks.9` | 0.0755 |
| `trf_blocks.2` | 0.0734 |
| `final_norm` | 0.0386 |
| `trf_blocks.10` | 0.0353 |
| `trf_blocks.11` | 0.0353 |
| `trf_blocks.0` | 0.0315 |
| `trf_blocks.1` | 0.0296 |

Both metrics point toward a **broad set of moved components**. Token embeddings and positional embeddings are prominent under both views, while layer norms and attention-value pathways also stand out. These analyses are therefore useful for generating transplantation hypotheses, but they should not be treated as causal proof on their own.

## Embedding Comparisons

Given that embedding appeared to be a major site of change based on the analyses above, we performed further analysis of changes at this locus. 

We began by calculting cosine similarity for each of the tokens in the tokenizer. For the entire set of 50,257 token embeddings, overall cosine similarity statistics for the population were:

| Metric | Cosine similarity |
| --- | ---: |
| Min | 0.8702 |
| Mean | 0.9684 |
| Max | 0.9993 |

Hence, the token embeddings clearly changed, but globally they remained fairly similar. The most changed tokens were also not obviously dominated by biomedical tokens. The bottom of the cosine-similarity ranking includes tokens such as `' You'`, `' someone'`, `' putting'`, `' Don'`, `' basically'`, and `' Matt'`. That weakens any claim that DAPT acted primarily by selectively rewriting a small, obvious set of biomedical token vectors.

We next examined whether DAPT was driving the representation of tokens comprising biomedical terms together. For example, " EGFR" is frequently found in our abstracts and decomposes into " EG" and "FR" at the token level. Because these tokens are much more frequently found together in our abstract set vs general text, we might expect their embeddings to coverge.
A small hand-picked set of token-pair tests showed only modest changes:

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

These embedding results appear to show somewhat variable results, but certainly do not point toward convergence of embeddings as a clear mechanism.

## Weight Transplantation Experiments

The transplantation experiments provide the clearest evidence for the distributed changes hypothesis. We examined the following hybrid models:

- `base_embed_dapt_model`: base model with transplanted DAPT token embeddings
- `base_embed_plusothers_dapt_model`: base model with DAPT token embeddings, DAPT positional embeddings, and selected DAPT layer-norm shifts (shifts of norm1 from block 0; shifts of norm2 from blocks 4,9,10,11)
- `base_embed_pos_trf0_11_model`: base model with DAPT token embeddings, DAPT positional embeddings, and full DAPT transformer blocks 0 and 11
- `base_embed_pos_trf5_6_model`: base model with DAPT token embeddings, DAPT positional embeddings, and full DAPT transformer blocks 5 and 6
- `dapt_embed_base_model`: DAPT model with transplanted base token embeddings

Validation losses on the 2.5% biomedical validation subset were:

| Model | Validation loss (2.5% subset) |
| --- | ---: |
| `dapt_model` | 2.5855 |
| `base_embed_pos_trf0_11_model` | 2.8648 |
| `dapt_embed_base_model` | 2.9057 |
| `base_model` | 2.9581 |
| `base_embed_pos_trf5_6_model` | 3.1940 |
| `base_embed_plusothers_dapt_model` | 3.2675 |
| `base_embed_dapt_model` | 3.3093 |

The 2.5% validation subset appears reasonable in that loss with the 2.5% validation set is very similar to the full validation set for both the base (2.9581 for 2.5% vs 2.9233 for full) and dapt models (2.5855 for 2.5% vs 2.5513 for full) and of the same direction (greater in the 2.5% set) and magnitude (~0.03 in both; relative change of ~1%).

These results support the distributed changes model. DAPT token embeddings appear to be **important but not sufficient**. Replacing the DAPT embeddings with the base embeddings (`dapt_embed_base_model`) substantially removes the DAPT advantage. But the reverse operation is even more informative: transplanting only the DAPT embeddings into the base model (`base_embed_dapt_model`) makes performance worse than the original base model. This result argues strongly against an embedding-only explanation.

Furthermore, the best hybrid model is not the embedding-only hybrid but `base_embed_pos_trf0_11_model`, which combines DAPT embeddings and positional embeddings with selected full DAPT blocks. This suggests that the DAPT gain depends on coordinated changes across multiple parts of the network.

Finally, the contrast between `base_embed_pos_trf0_11_model` and `base_embed_pos_trf5_6_model` suggests that some block locations may matter more than others for partial recovery of the DAPT gain. However, that result should be interpreted cautiously: it nominates candidate important blocks rather than proving that blocks 0 and 11 are uniquely causal.

To connect the loss results back to behavior, the report also examined the biomedical prompt context ending in `' ER'`, where the next token should be `'BB'` in an `ERBB2`-style continuation:

| Model | P(`'BB'` in biomedical ` ' ER'` context) |
| --- | ---: |
| `base_model` | 0.000555 |
| `base_embed_plusothers_dapt_model` | 0.000844 |
| `dapt_embed_base_model` | 0.371979 |
| `dapt_model` | 0.774406 |

This table reinforces the main loss-based conclusion. The full DAPT model strongly favors the biomedical continuation. Replacing its embeddings with the base embeddings weakens that behavior, but only partially back to the base model level. That pattern is consistent with embeddings mattering substantially while still being only part of the story.


## Interpretation

The strongest supported conclusion from the notebook is that biomedical DAPT changed the model in a **distributed, coordinated way**.

Three takeaways can be ranked by confidence.

**Highest confidence:** the DAPT improvement is not explained by token-embedding swaps alone. The embedding-only transplant fails badly, while removing DAPT embeddings from the DAPT model also sharply weakens performance.

**Moderate confidence:** the DAPT gain depends on coordinated changes across multiple components, including embeddings and at least some additional transformer machinery. The best hybrid model partially recovers the DAPT advantage only when embeddings are combined with other transplanted DAPT components.

**Lower confidence but plausible:** some block locations may matter more than others for partial recovery. The block-0-and-11 hybrid performed much better than the block-5-and-6 hybrid, but the current experiments are too selective to treat that as a final mechanistic localization.

Overall, the report supports a view in which DAPT on this biomedical corpus did not simply rewrite a few domain token vectors. Instead, it altered how multiple model components work together to produce biomedical continuations.

## Limitations

This report has several important limitations.

- The weight-transplantation study is selective rather than exhaustive, so it provides useful clues rather than a full causal decomposition.
- The prompt-based evaluation is qualitative and relies on a small hand-written prompt set.
- Out-of-domain evaluation is limited. Aside from a few control prompts, this report does not quantify how much general-language capability was preserved or degraded after DAPT.
- The parameter-difference analyses identify where weights moved, but not which of those movements are strictly necessary for the behavioral change.

## Appendix A: Extended Tables

Much more detailed output remains available in `compare_base_vs_dapt.ipynb`. The tables below are retained mainly as supporting reference material.

### A1. Top 10 most changed tokens by base-vs-DAPT embedding cosine

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

### A2. Top 10 least changed tokens by base-vs-DAPT embedding cosine

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

### A3. Changed blocks (simple summary)

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

### A4. Submodule summary

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

