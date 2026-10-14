# Qwen and OLMo decoder norm comparison

## Hypothesis

Neighboring decoder normalization scales and persistent coordinate profiles
may differ across model families, generations, sizes and dense/MoE architectures.

## Setup

Analysis function: `qwen_norm_compare.main` in `qwen_norm_compare.py`.
The script is the source of truth for checkpoint selection and analysis settings.

## Results

The comparison includes Qwen plus dense and MoE OLMo checkpoints, using only
decoder norm scales. Neighbor similarity is already high in Qwen2.5-7B-Instruct.
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
OLMoE has median all-pair rank correlations of 0.254 and 0.574 for its attention
and FFN input norms. Dense OLMo 2 has 0.394/0.526 at 7B and 0.481/0.695 at 32B
for its attention and FFN output norms. Olmo 3-7B-Instruct has 0.395/0.541 for
these output norms. OLMo scales were also obtained through
norm-only downloads. Actual family names remain in the saved data; plots identify
the branch role and whether normalization is applied before or after the branch.

## Conclusions

High neighbor similarity and a shared coordinate profile across the entire depth
are distinct effects. The available checkpoints do not identify when either
first emerged: no earlier Qwen generation is available locally, and model size,
architecture and training are confounded. Earlier checkpoints would be needed to
test the historical onset rather than infer it from version names.
The weak global input profile of the dense Qwen3.5-9B model is not shared by all
Qwen3.5 variants; the MoE comparison alone still does not isolate a causal factor.
OLMoE is a counterexample to treating MoE as sufficient for the strong global
coordinate profile seen in Qwen3 MoE. OLMo 2/3 branch-output norms and Qwen/OLMoE
branch-input norms occupy different functional locations; comparisons describe
parameter profiles and do not make these operators equivalent.

## Changelog

- 2026-10-14: Compared cached language-decoder norm scales; generated aggregate
  similarity, persistence and per-model rank maps.
- 2026-10-14: Added a Qwen3.5 MoE norm-only subset and updated comparisons.
- 2026-10-14: Added OLMo dense/MoE norm subsets and distinguished pre/post branch
  roles in comparison figures.
