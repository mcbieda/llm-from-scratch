# DAPT Alteration Analysis

## Summary

This report summarizes the executed analyses in [compare_base_vs_dapt.ipynb](/home/markb/cloned-llm-2026MAR16/llm-from-scratch/notebooks/compare_base_vs_dapt.ipynb). The main finding is that domain-adaptive pretraining (DAPT) materially changed model behavior, but the change is based on widespread changes in the model. Although the token embedding layer (which was weight tyed to the output layer) showed the largest changes between base and dapt models, simple transplantation of this layer from dapt to base did not lead to dapt behavior.

Overall, the strongest evidence comes from three places. First, prompt-level next-token behavior shifts sharply in biomedical contexts, especially for ERBB2-related completions. Second, the largest parameter changes include the token embedding matrix, some layer-norm shift vectors, and attention value/output weights. Third, weight-transplantation experiments show that swapping the base embedding layer into the DAPT model (DAPT model + base embedding) nearly removes the DAPT advantage for the validation set, but swapping the DAPT embedding layer into the base model (base + DAPT embedding) does not produce the DAPT loss advantage, but instead produces a much worse model than the base model. However, a hybrid model with more swaps (base model + DAPT embedding + DAPT positional encoding + DAPT transformer first block + DAPT transformer last block) does lead to slight improvement of base model loss (on the validation subset) and a great increment over the simple embedding swap (base + DAPT embedding). Hence, there is support for distributed changes in the DAPT model that are critical for the DAPT model advantage for validation subset loss.

## Experimental Setup

The notebook compares two GPT-2 small models:

- `base_model`: GPT-2 loaded from `data/gpt2_openai_params_124M.pkl`.
- `dapt_model`: a checkpoint loaded from `data/TEST_abstracts_epoch_lastsave_step_lastsave.pth`.

Both models use the GPT-2 tokenizer and are evaluated in three ways:

1. Embedding-level comparisons between the base and DAPT token embeddings.
2. Prompt-based next-token probability comparisons on general and biomedical prompts.
3. Validation-loss evaluation on a slice of `pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_val_abstracts.txt`.

The validation evaluation used:

- Validation fraction: `0.025`
- Characters evaluated: `72,602` of `2,904,096`
- Batch size: `2`
- Context length: `1024`
- Stride: `1024`
- Number of validation batches: `8`

The notebook also constructs several hybrid models by transplanting selected DAPT weights into the base model, or base embeddings into the DAPT model.

## Main Results

### Embedding comparisons

Across all `50,257` token embeddings, cosine similarity between the base and DAPT embedding vectors was:

| Metric | Cosine similarity |
| --- | ---: |
| Min | 0.8841 |
| Mean | 0.9707 |
| Max | 0.9995 |

This suggests that token embedding changes were real but generally modest at the global level.

Small hand-picked token-pair tests also showed only modest changes:

| Pair group | Token pair | Base cosine | DAPT cosine | Ratio |
| --- | --- | ---: | ---: | ---: |
| Biomedical-related | `' ER'` vs `'BB'` | 0.2552 | 0.2603 | 1.0196 |
| Biomedical-related | `' HER'` vs `'2'` | 0.2598 | 0.2633 | 1.0136 |
| Biomedical-related | `' kin'` vs `'ase'` | 0.2703 | 0.2712 | 1.0034 |
| Biomedical-related | `' EG'` vs `'FR'` | 0.2568 | 0.2503 | 0.9750 |
| Comparator | `'HER'` vs `' 2'` | 0.1710 | 0.1790 | 1.0464 |
| Comparator | `'EG'` vs `' FR'` | 0.2667 | 0.2376 | 0.8912 |
| Comparator | `'BB'` vs `'2'` | 0.2752 | 0.2726 | 0.9905 |
| Comparator | `' cat'` vs `' dog'` | 0.5498 | 0.5353 | 0.9737 |

The most changed tokens were not clear biomedical tokens or clearly related to the DAPT training set. The bottom of the cosine-similarity ranking includes tokens such as `' You'`, `' Matt'`, `' Get'`, `' putting'`, `' Make'`, and `' police'`. That weakens any claim that DAPT primarily rewired only a small set of biomedical token embeddings.

### Prompt-based behavior changes

Prompt behavior changed much more dramatically than the token-pair embedding checks suggest.

Selected examples:

| Prompt / context | Expected token / continuation | Base model | DAPT model | Interpretation |
| --- | --- | ---: | ---: | --- |
| Cancer context ending with `HER` | `'2'` | 55.1% | 96.4% | Base model was already correct, but DAPT sharply strengthens the biomedical completion. |
| Cancer context ending with `ER` and intended to continue toward `ERBB2` | `'BB'` | 0.1% | 57.0% | DAPT strongly shifts the continuation toward biomedical terminology. |
| Emergency-room context ending with `ER` | Several possible continuations | top: `.` (31.7%) | top: `,` (23.1%) | DAPT changes ranking, but context still prevents the model from strongly collapsing onto `ERBB2`. |
| Ordinary pet-context prompt ending in `dogs and` | `' cats'` | 94.0% | 72.3% | General language behavior remains substantially intact despite biomedical drift. |

The ERBB2-style prompt is the clearest qualitative result. In the colorectal-cancer context ending in `ER`, the base model assigns negligible probability to `BB`, while the DAPT model makes `BB` the top next token at about 57.0%. This is a large functional shift, not a minor ranking perturbation.

The more general control prompts also show drift. For example, after ` dogs and`, the DAPT model places biomedical-flavored tokens such as `' breast'`, `' HER'`, `' EG'`, `' plasma'`, and `' patients'` in its top-10 predictions, whereas the base model stays on ordinary pet-related continuations. This indicates that DAPT broadened the model's tendency to surface domain-associated tokens even in weakly related contexts.

### Parameter-difference analysis

The largest individual tensor changes were:

| Tensor | Relative L2 |
| --- | ---: |
| `trf_blocks.0.norm1.shift` | 0.2429 |
| `trf_blocks.11.norm2.shift` | 0.2410 |
| `tok_emb.weight` | 0.2391 |
| `trf_blocks.10.norm2.shift` | 0.1610 |
| `trf_blocks.4.norm2.shift` | 0.1348 |
| `trf_blocks.9.norm2.shift` | 0.1343 |
| `trf_blocks.0.att.W_value.weight` | 0.1328 |

Using the notebook's `Changed blocks (simple summary)` output, the embedding tables are still the most changed components, followed by blocks 0 and 11 and then a fairly broad spread across the rest of the transformer stack:

| Block | Mean relative L2 | Max relative L2 |
| --- | ---: | ---: |
| `tok_emb` | 0.2391 | 0.2391 |
| `pos_emb` | 0.0794 | 0.0794 |
| `trf_blocks.0` | 0.0693 | 0.2429 |
| `trf_blocks.11` | 0.0686 | 0.2410 |
| `trf_blocks.5` | 0.0657 | 0.1245 |
| `trf_blocks.2` | 0.0632 | 0.1333 |

At the submodule level, the largest changes are concentrated in:

- `tok_emb`
- `att.W_value`
- `pos_emb`
- `norm2`
- `norm1`

This points to a broader redistribution of model behavior across embeddings, normalization terms, and attention pathways rather than a single isolated source of adaptation.

## Weight Transplantation Experiments

The notebook evaluates several hybrid models:

- `base_embed_dapt_model`: base model with only DAPT token embeddings.
- `base_embed_plusothers_dapt_model`: base model with DAPT token embeddings, DAPT positional embeddings, and a few selected layer-norm shifts. This is based on the tensor analysis above, and positional embeddings because they appear in the block and submodule listings.
- `base_embed_pos_trf0_11_model`: base model with DAPT token embeddings, positional embeddings, and full blocks 0 and 11. These are the top 4 entries in the altered block analysis.
- `base_embed_pos_trf5_6_model`: base model with DAPT token embeddings, positional embeddings, and full blocks 5 and 6. This acts as a partial control for use of blocks 0 and 11 in the other hybrid, as 5 and 6 are lower on the list of altered blocks.
- `dapt_embed_base_model`: DAPT model with token embeddings replaced by base embeddings.

Validation losses on the sampled biomedical validation set:

| Model | Subset validation loss |
| --- | ---: |
| `dapt_model` | 2.5963 |
| `base_embed_pos_trf0_11_model` | 2.8796 |
| `dapt_embed_base_model` | 2.9055 |
| `base_model` | 2.9581 |
| `base_embed_pos_trf5_6_model` | 3.1716 |
| `base_embed_plusothers_dapt_model` | 3.2686 |
| `base_embed_dapt_model` | 3.3036 |

These hybrids support three conclusions:

1. To begin, DAPT did lead to a decrease in the subset validation loss, as expected and seen in prompt completions above. 
2. DAPT embeddings appear necessary, but not sufficient, for the loss improvement. The (DAPT + base embeddings) model shows almost total loss of the DAPT advantage while (base + DAPT embeddings) makes validation loss worse than the original base model and much worse than the DAPT model.
3. All the base model hybrids had worse loss than the base model, except for the (base + DAPT embeddings + DAPT positional encoding + DAPT first block + DAPT last block), which showed a marginal improvement over the base model. This supports the idea that coordinated changes are necessary for the DAPT advantage.
3. Our block analysis may point toward more important blocks for DAPT loss advantage, and this makes sense with simple logic of LLM action. The (base + DAPT embeddings + DAPT positional encoding + DAPT first block + DAPT last block) is much better than (base + DAPT embeddings + DAPT positional encoding + DAPT block 5 + DAPT block 6). This is consistent with the first and last tranformer blocks playing a key role in dealing with the altered DAPT embedding.

The prompt-based `BB` probability results are consistent with this pattern:

| Model | `P('BB' | biomedical ER context)` |
| --- | ---: |
| `base_model` | 0.000555 |
| `base_embed_plusothers_dapt_model` | 0.008450 |
| `dapt_embed_base_model` | 0.154248 |
| `dapt_model` | 0.569766 |

So the strongest ERBB2 behavior depends mainly on the broader DAPT transformer state, not on token embeddings alone.

## Interpretation

The notebook's strongest supported conclusion is that DAPT altered the model in a distributed way. The embedding matrix changed substantially in parameter space, but embedding-space similarity checks by themselves understate the behavioral shift. Large functional effects appear in context-conditioned predictions, especially in biomedical completions, and these effects survive partial removal of the DAPT embedding changes.

A reasonable interpretation is:

- DAPT moved the model toward biomedical continuations through coordinated changes across embeddings, attention value pathways, layer norms, and multiple transformer blocks.
- The token embedding matrix is an important site of change, but it must work with other changes to be effective.
- The middle transformer blocks and attention/value pathways may show relatively small changes, but are critical for the DAPT loss advantage. 

## Limitations

- The validation analysis uses only `2.5%` of one domain-specific validation file, so the loss ranking should be treated as directional rather than definitive.
- Prompt-based evaluation is qualitative and uses a small hand-written prompt set.
- The notebook compares only one base model and one DAPT checkpoint; there is no run-to-run variance estimate.
- The weight-transplantation study is selective rather than exhaustive, so it identifies useful clues, not a complete causal decomposition.
- Several notebook interpretations are stronger than the evidence supports, especially claims that focus narrowly on biomedical token-pair embedding similarity.

## Appendix A: Extended Tables

### A1. Top 10 most changed tokens by base-vs-DAPT embedding cosine

Lowest cosine similarlity tokens from the notebook:

| Rank | Token | Cosine similarity |
| --- | --- | ---: |
| 1 | `' You'` | 0.884130 |
| 2 | `' Matt'` | 0.893998 |
| 3 | `' Get'` | 0.899295 |
| 4 | `' putting'` | 0.899304 |
| 5 | `' Make'` | 0.899364 |
| 6 | `' police'` | 0.899783 |
| 7 | `' Dan'` | 0.900174 |
| 8 | `' pretty'` | 0.900250 |
| 9 | `' Chris'` | 0.900631 |
| 10 | `'You'` | 0.900823 |

### A2. Top 10 least changed tokens by base-vs-DAPT embedding cosine

Highest cosine similarity tokens from the notebook:

| Rank | Token | Cosine similarity |
| --- | --- | ---: |
| 1 | `'iterranean'` | 0.999482 |
| 2 | `'algia'` | 0.999239 |
| 3 | `'urations'` | 0.999215 |
| 4 | `'IVES'` | 0.999181 |
| 5 | `'GROUND'` | 0.999162 |
| 6 | `'opsy'` | 0.999158 |
| 7 | `'itative'` | 0.999126 |
| 8 | `'asms'` | 0.999122 |
| 9 | `'CRIPTION'` | 0.999104 |
| 10 | `'ETHOD'` | 0.999102 |

### A3. Changed blocks (simple summary)

| Block | Mean relative L2 | Max relative L2 | Mean cosine similarity |
| --- | ---: | ---: | ---: |
| `tok_emb` | 0.239102 | 0.239102 | 0.976101 |
| `pos_emb` | 0.079381 | 0.079381 | 0.997290 |
| `trf_blocks.0` | 0.069316 | 0.242853 | 0.996674 |
| `trf_blocks.11` | 0.068649 | 0.241008 | 0.996605 |
| `trf_blocks.5` | 0.065701 | 0.124532 | 0.997838 |
| `trf_blocks.2` | 0.063246 | 0.133276 | 0.997986 |
| `trf_blocks.9` | 0.062557 | 0.134301 | 0.998021 |
| `trf_blocks.6` | 0.062055 | 0.131899 | 0.998078 |
| `trf_blocks.7` | 0.060903 | 0.111660 | 0.998237 |
| `trf_blocks.10` | 0.060601 | 0.160957 | 0.997926 |
| `trf_blocks.8` | 0.060534 | 0.109388 | 0.998276 |
| `trf_blocks.3` | 0.060469 | 0.103831 | 0.998220 |
| `trf_blocks.4` | 0.060012 | 0.134788 | 0.998114 |
| `trf_blocks.1` | 0.059228 | 0.110822 | 0.998295 |
| `final_norm` | 0.039702 | 0.044994 | 0.999887 |

### A4. Submodule summary

| Submodule | Mean relative L2 | Max relative L2 |
| --- | ---: | ---: |
| `tok_emb` | 0.239102 | 0.239102 |
| `att.W_value` | 0.086988 | 0.132808 |
| `pos_emb` | 0.079381 | 0.079381 |
| `norm2` | 0.079009 | 0.241008 |
| `norm1` | 0.067458 | 0.242853 |
| `ff.layers.0` | 0.062708 | 0.083550 |
| `att.out_proj` | 0.059264 | 0.103831 |
| `ff.layers.2` | 0.055963 | 0.089390 |
| `att.W_query` | 0.052161 | 0.086554 |
| `att.W_key` | 0.049370 | 0.088731 |
| `final_norm` | 0.039702 | 0.044994 |

## Appendix B: Additional Notes

- The notebook's embedding analyses and prompt analyses should not be read as equivalent evidence. The prompt analyses are much more informative about actual model behavior.
- Because GPT-2 ties the token embedding and output head weights, token-embedding changes also directly affect the output distribution, but the transplantation experiments show that this direct effect is still not enough to explain the full DAPT improvement.
- The best-performing hybrid in the notebook is `base_embed_pos_trf0_11_model`, which suggests that some targeted block swaps can recover part of the domain gain, but the result is still clearly inferior to the full DAPT checkpoint.
- The poor performance of `base_embed_dapt_model` is a useful negative result: a transplanted embedding table can be mismatched to the rest of the base network.
