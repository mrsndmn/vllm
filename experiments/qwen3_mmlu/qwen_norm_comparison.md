# Cached Qwen decoder norm comparison

## Hypothesis

Neighboring decoder normalization scales and persistent coordinate profiles
may differ across Qwen generations, model sizes and dense/MoE architectures.

## Setup

Analysis function: `qwen_norm_compare.main` in `qwen_norm_compare.py`.
The script is the source of truth for checkpoint selection and analysis settings.

## Results

Eleven cached models and an additional Qwen3.5 MoE checkpoint were analyzed using
only decoder norm scales. Neighbor similarity is already high in Qwen2.5-7B-Instruct.
The Qwen3-30B-A3B variants have substantially stronger similarity across arbitrary
layer pairs and more persistent high coordinates than the smaller dense models.
Thinking and Instruct variants have nearly identical aggregate measurements;
VL/Omni language decoders broadly follow their corresponding size families.
Qwen3.5-9B input norms have high cosine similarity but much weaker coordinate-rank
agreement, especially across distant layers. Its effective scale is reconstructed
from the checkpoint parameterization before comparison.
Qwen3.5-35B-A3B has median all-pair rank correlations of 0.754 for input norms
and 0.893 for post-attention norms. Its input profile is substantially more
persistent than the dense Qwen3.5-9B profile. Only norm tensors and metadata
were fetched for this additional checkpoint; the full model was not downloaded.

## Conclusions

High neighbor similarity and a shared coordinate profile across the entire depth
are distinct effects. The available checkpoints do not identify when either
first emerged: no earlier Qwen generation is available locally, and model size,
architecture and training are confounded. Earlier checkpoints would be needed to
test the historical onset rather than infer it from version names.
The weak global input profile of the dense Qwen3.5-9B model is not shared by all
Qwen3.5 variants; the MoE comparison alone still does not isolate a causal factor.

## Changelog

- 2026-10-14: Compared cached language-decoder norm scales; generated aggregate
  similarity, persistence and per-model rank maps.
- 2026-10-14: Added a Qwen3.5 MoE norm-only subset and updated comparisons.
