# Cached Qwen decoder norm comparison

## Hypothesis

Neighboring decoder normalization scales and persistent coordinate profiles
may differ across Qwen generations, model sizes and dense/MoE architectures.

## Setup

Analysis function: `qwen_norm_compare.main` in `qwen_norm_compare.py`.
The script is the source of truth for checkpoint selection and analysis settings.

## Results

All eleven cached models with available weights were analyzed using only decoder
norm scales. Neighbor similarity is already high in Qwen2.5-7B-Instruct.
The Qwen3-30B-A3B variants have substantially stronger similarity across arbitrary
layer pairs and more persistent high coordinates than the smaller dense models.
Thinking and Instruct variants have nearly identical aggregate measurements;
VL/Omni language decoders broadly follow their corresponding size families.
Qwen3.5-9B input norms have high cosine similarity but much weaker coordinate-rank
agreement, especially across distant layers. Its effective scale is reconstructed
from the checkpoint parameterization before comparison.

## Conclusions

High neighbor similarity and a shared coordinate profile across the entire depth
are distinct effects. The available checkpoints do not identify when either
first emerged: no earlier Qwen generation is available locally, and model size,
architecture and training are confounded. Earlier checkpoints would be needed to
test the historical onset rather than infer it from version names.

## Changelog

- 2026-10-14: Compared cached language-decoder norm scales; generated aggregate
  similarity, persistence and per-model rank maps.
