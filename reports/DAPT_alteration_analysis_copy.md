# DAPT Alteration Analysis

## Summary

In this report, we examine the differences between a  gpt2-small model (124M parameters) using the base OpenAI weights (base_model) and after domain-adaptive pretraining with a set of PubMed biomedical abstracts related to cancer (dapt_model).
This report summarizes the executed analyses in compare_base_vs_dapt.ipynb found in /notebooks. The notebook gpt2_basic_training_abstracts.ipynb was used for DAPT.  
The main finding is that domain-adaptive pretraining (DAPT) significantly changed model behavior, but the change is not localized to a single layer/component, but rather is from alterations in multiple components. 

Overall, the strongest evidence for distributed representation comes from three observations. First, prompt-level next-token behavior shifts sharply in biomedical contexts with DAPT. Second, direct evaluation of the largest parameter changes points toward several sites, including the token embedding matrix, some layer-norm shift vectors, and attention value/output weights. Third - and comprising the strongest evidence -  weight-transplantation experiments support distributed representation. Swapping the base embedding layer into the DAPT model (DAPT model + base embedding) nearly removes the DAPT advantage for the validation set, but swapping only the DAPT embedding layer into the base model (base + DAPT embedding) does not produce the DAPT loss advantage, and instead creates a worse model than base alone. A base model with DAPT embedding and additional changes performs better than base and much better than base + DAPT embedding only, supporting the critical importance of alterations across various components. Hence, there is support for distributed changes in the DAPT model that are critical for the DAPT model advantage.

## Experimental Setup

The two fundamental models are GPT-2 small (124M) models:

- `base_model`: GPT-2-small using OpenAI weights; loaded from `data/gpt2_openai_params_124M.pkl`.
- `dapt_model`: a DAPT model  loaded from `data/TEST_abstracts_epoch_lastsave_step_lastsave.pth`; this model was trained starting with the base_model weights

The DAPT model was trained on a set of biomedical abstracts that were derived by Pubmed Entrez system using the prompt "(cancer[Title/Abstract]) AND english[lang] AND (ERBB2[Title/Abstract] OR HER2[Title/Abstract] OR EGFR[Title/Abstract])". These abstracts were further limited to the year range of 2005 - 2025. A random subset of these abstracts was used to create three different sets: a training, test, and validation set. However, for this small investigation, the "test" set was used for DAPT, comprising 7,267,986 characters and, after tokenization using the gpt-2 tokenizer, 1,679,880 tokens. For initial determination of validation loss, the full validation set was used (characters: 2,904,096, tokens: 671,020). In later investigations, only 2.5% of this validation set was used to compare a range of hybrid models. The loss values in the 2.5% set were close to the loss values for the full validation set for the base and DAPT models, supporting the usage of this smaller subset for efficient comparison across a set of models.

Both models use the GPT-2 tokenizer and are evaluated in three ways:

1. Embedding-level comparisons between the base and DAPT token embeddings.
2. Prompt-based next-token probability comparisons on general and biomedical prompts.
3. Validation-loss evaluation on wither the entire validation set or a slice of `pubmed_abstracts_2005to2025ONLY_ERBB2_ABSTRACTS_getv7_english_val_abstracts.txt`.

The notebook also constructs several hybrid models by transplanting selected DAPT weights into the base model, or base embeddings into the DAPT model.

## Main Results

# FIX
 - check numbers
 - add hypotheses - why these are important
 - add the measures

### Embedding comparisons

We began by examining changes with DAPT in the token embedding layer.

Across all `50,257` token embeddings, cosine similarity between the base and DAPT embedding vectors was:

| Metric | Cosine similarity |
| --- | ---: |
| Min | 0.8702 |
| Mean | 0.9684 |
| Max | 0.9993 |

This suggests that token embedding changes were real but generally modest at the global level.

The most changed tokens were not clear biomedical tokens or clearly related to the DAPT training set. The bottom of the cosine-similarity ranking includes tokens such as `' You'`, `' someone'`, `' putting'`, `' Don'`, `' basically'`, and `' Matt'`. That weakens any claim that DAPT primarily rewired only a small set of biomedical token embeddings.

Even if token embedding changes are small overall, changes in the similarity of token embeddings, especially those linked to biomedical terms, could be important.
Due to the abstract selection criteria, our set will be much enriched for "HER2" (the protein overexpressed in HER2-positive cancers); "ERBB2" (the gene name for the gene that produces HER2); "EGFR" (the gene name for epidermal growth factor); "kinase" (technical term for proteins that activate other proteins). Due to the nature of GPT-2 standard tokenization with the inclusion of spaces in tokens, we can examine these vs closely related terms, as shown in the token-pair test table. For example, "HER2" as used would usually be " HER" and "2" and not "HER" and " 2". Examination of a set of these showed changes, but overall, the results are somewhat unclear.

Small hand-picked token-pair tests also showed only modest changes:

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

_Note "similarity" in above table is cosine similarity_



### Prompt-based behavior changes

Prompt behavior changed much more dramatically than the token-pair embedding checks suggest.

Selected examples:

| Prompt / context | Expected token / continuation | Base model | DAPT model | Interpretation |
| --- | --- | ---: | ---: | --- |
| Cancer context ending with `HER` | `'2'` | 55.1% | 98.8% | Base model was already correct, but DAPT sharply strengthens the biomedical completion. |
| Cancer context ending with `ER` and designed to continue toward `ERBB2` | `'BB'` | 0.1% | 77.4% | DAPT strongly shifts the continuation toward biomedical terminology. |
| Emergency-room context ending with `ER` | Several possible continuations | top: `.` (31.7%) | top: `' and'` (14.5%) | DAPT changes ranking, but context still prevents the model from strongly collapsing onto `ERBB2`, which would be the `'BB'` token. |
| Ordinary pet-context prompt ending in `dogs and` | `' cats'` | 94.0% | 67.7% | General language behavior remains substantially intact despite biomedical drift. |

The ERBB2-style prompt is the clearest qualitative result. In the cancer context ending in `ER`, the base model assigns negligible probability to `BB`, while the DAPT model makes `BB` the top next token at about 77.4%. This is a large functional shift.

The more general control prompts also show drift. For example, after ` dogs and`, the DAPT model places biomedical-flavored tokens such as `' patients'`, `' cancer'`, `' metast'`, `' tumor'`, and `' breast'` in its top-10 predictions, whereas the base model stays on ordinary pet-related continuations. This indicates that DAPT broadened the model's tendency to surface domain-associated tokens even in weakly related contexts.

### Parameter-difference analysis



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

For the above table, "Relative L2" is computed as the magnitude of the change divided by the base magnitude from:
$\text{Relative L2} = \frac{\lVert W_{\text{DAPT}} - W_{\text{base}} \rVert_2}{\lVert W_{\text{base}} \rVert_2 + \varepsilon}$



To examine changes on a block level, we computed the mean change for all tensors in a block using the equation above for each tensor; this output is in the notebook's `Changed blocks (simple summary)` output. The equation used is: $\text{Simple Block Mean Relative L2} = \frac{1}{N}\sum_{i=1}^{N}\frac{\left\lVert W^{(i)}_{\text{DAPT}} - W^{(i)}_{\text{base}} \right\rVert_2}{\left\lVert W^{(i)}_{\text{base}} \right\rVert_2 + \varepsilon}$

 (The "max relative L2" is simply the value for the relative L2 of the most changed tensor in a block.) The embedding tables are still the most changed components, followed by blocks 0 and 11 and then a fairly broad spread across the rest of the transformer stack:

| Block | Simple Block Mean Relative L2 | Simple Block Max Relative L2 |
| --- | ---: | ---: |
| `tok_emb` | 0.2483 | 0.2483 |
| `pos_emb` | 0.0838 | 0.0838 |
| `trf_blocks.0` | 0.0735 | 0.2469 |
| `trf_blocks.11` | 0.0728 | 0.2601 |
| `trf_blocks.5` | 0.0698 | 0.1277 |
| `trf_blocks.2` | 0.0671 | 0.1370 |

At the submodule level, the largest changes are concentrated in:

- `tok_emb`
- `att.W_value`
- `pos_emb`
- `norm2`
- `norm1`

This points to a broader redistribution of model behavior across embeddings, normalization terms, and attention pathways rather than a single isolated source of adaptation.

Arguably, taking the mean across the relative L2 values is not the proper way to aggregate the values; instead we could use an alternative measure: $\text{Aggregated Block Relative L2} = \frac{\sqrt{\sum_i \lVert W^{(i)}_{\text{DAPT}} - W^{(i)}_{\text{base}} \rVert_2^2}}{\sqrt{\sum_i \lVert W^{(i)}_{\text{base}} \rVert_2^2} + \varepsilon}$

By this measure, the rankings were somewhat different:

| Block | Aggregated Block Relative L2 |
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

Note that the max is not listed here, because that would be a value from a single position in a tensor and would not be informative.

Note that token embedding, positional embedding are the top 2, as with the simple block L2, but the next two are quite different. Here, the 3rd and 4th most changed are transformer blocks 5 and 6, while in the simple relative L2, it is blocks 11 and 0.

It is worth considering the differences in these two measures. The "simple" measure weights _each tensor_ equally; the "aggregated" measure weights _each element of all the tensors in a block_ equally. 



## Weight Transplantation Experiments

The notebook evaluates several hybrid models:

- `base_embed_dapt_model`: base model with only DAPT token embeddings.
- `base_embed_plusothers_dapt_model`: base model with DAPT token embeddings, DAPT positional embeddings, and a few selected layer-norm shifts. This is based on the tensor analysis above, and positional embeddings because they appear in the block and submodule listings.
- `base_embed_pos_trf0_11_model`: base model with DAPT token embeddings, positional embeddings, and full blocks 0 and 11. These are the top 4 entries in the "simple" altered block analysis.
- `base_embed_pos_trf5_6_model`: base model with DAPT token embeddings, positional embeddings, and full blocks 5 and 6. This acts as a partial control for use of blocks 0 and 11 in the other hybrid, as 5 and 6 are lower on the list of "simple" altered blocks; note that these alterations are the top 4 in the "aggregated" block change analysis.
- `dapt_embed_base_model`: DAPT model with token embeddings replaced by base embeddings.

Validation losses on the sampled biomedical validation set (2.5% of the entire validation set):

| Model | Subset validation loss |
| --- | ---: |
| `dapt_model` | 2.5855 |
| `base_embed_pos_trf0_11_model` | 2.8648 |
| `dapt_embed_base_model` | 2.9057 |
| `base_model` | 2.9581 |
| `base_embed_pos_trf5_6_model` | 3.1940 |
| `base_embed_plusothers_dapt_model` | 3.2675 |
| `base_embed_dapt_model` | 3.3093 |

These hybrids support four conclusions:

1. To begin, DAPT did lead to a decrease in the subset validation loss, as expected and seen in prompt completions above. 
2. DAPT embeddings appear necessary, but not sufficient, for the loss improvement. The (DAPT + base embeddings) model shows almost total loss of the DAPT advantage while (base + DAPT embeddings) makes validation loss worse than the original base model and much worse than the DAPT model.
3. All the base model hybrids had worse loss than the base model, except for the (base + DAPT embeddings + DAPT positional encoding + DAPT first block + DAPT last block), which showed a modest improvement over the base model. This supports the idea that coordinated changes are necessary for the DAPT advantage.
4. The block analysis still points toward some blocks being more important than others for the DAPT loss advantage. The (base + DAPT embeddings + DAPT positional encoding + DAPT first block + DAPT last block) model is much better than (base + DAPT embeddings + DAPT positional encoding + DAPT block 5 + DAPT block 6), which is consistent with the first and last transformer blocks playing a key role in accommodating the altered DAPT embedding. 
5. This also points toward the "simple" block change analysis being potentially superior to the "aggregated" block change analysis.

To examine this further, we tested the biomedical prompt ending in ' ER' that should lead to 'BB' as the next token, just as in the above table of prompt completion percentages. This was tested in a subset of models:


| Model | P(`'BB'` in biomedical `' ER'` context) |
| --- | ---: |
| `base_model` | 0.000555 |
| `base_embed_plusothers_dapt_model` | 0.000844 |
| `dapt_embed_base_model` | 0.371979 |
| `dapt_model` | 0.774406 |

These results broadly support the conclusions from the loss values above; the dapt_model becomes significantly worse with usage of the base_model encoding and the base model becomes slightly better with usage of some dapt_model tensors.

## Interpretation

The notebook's strongest supported conclusion is that DAPT altered the model in a distributed way. The embedding matrix changed substantially in parameter space, but embedding-space similarity checks by themselves understate the behavioral shift. Large functional effects appear in context-conditioned predictions, especially in biomedical completions, and these effects survive partial removal of the DAPT embedding changes.

A reasonable interpretation is:

- DAPT moved the model toward biomedical continuations through coordinated changes across embeddings, attention value pathways, layer norms, and multiple transformer blocks.
- The token embedding matrix is an important site of change, but it must work with other changes to be effective.
- The first and last transformer blocks appear to be important sites of change, with middle blocks being potentially less important.

 
## Limitations

- The weight-transplantation study is selective rather than exhaustive, so it identifies useful clues, not a complete causal decomposition.
- Prompt-based evaluation is qualitative and uses a small hand-written prompt set.
- The validation analysis uses only `2.5%` of one domain-specific validation file, so the loss ranking should be treated as directional rather than definitive.
- Each model was fully deterministic, using the best next token prediction. Furthermore, the models did not start from random weights, but rather with the OpenAI supplied weights. Hence, run to run variance should be very small in magnitude.


## Appendix A: Extended Tables

Note that much more detailed and lengthy outputs are available in the compare_base_vs_dapt.ipynb notebook.

### A1. Top 10 most changed tokens by base-vs-DAPT embedding cosine

Lowest cosine similarlity tokens from the notebook:

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

Highest cosine similarity tokens from the notebook:

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

## Appendix B: Additional Notes

- The notebook's embedding analyses and prompt analyses should not be read as equivalent evidence. The prompt analyses are much more informative about actual model behavior.
- Because GPT-2 ties the token embedding and output head weights, token-embedding changes also directly affect the output distribution, but the transplantation experiments show that this direct effect is still not enough to explain the full DAPT improvement.
- The best-performing hybrid in the notebook is `base_embed_pos_trf0_11_model`, which suggests that some targeted block swaps can recover part of the domain gain, but the result is very from the full dapt_model loss.
- The poor performance of `base_embed_dapt_model` is a useful negative result: a transplanted embedding table can be mismatched to the rest of the base network.
