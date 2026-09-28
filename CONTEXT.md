# Qwen3 MMLU activation-zeroing experiments

Local experiments measuring how Qwen3-30B-A3B's MMLU accuracy changes when small
intermediate activations are set to zero during inference.

## Language

**Zeroing site**:
The place in each layer where activations are tested and zeroed: `mlp` (selected
experts' SiLU-and-multiply outputs before down projection) or `block` (the FP32
residual-stream sum before the next add/RMSNorm).
_Avoid_: rounding site, hook, location

**Token row**:
All activations of one token at one layer and zeroing site: the concatenated
outputs of its 8 selected experts for `mlp`, or its hidden-size residual vector
for `block`.
_Avoid_: activation vector, sample

**Fixed-ε zeroing**:
Zeroing every nonzero finite activation whose magnitude is at most a single
constant ε shared by all layers and tokens.
_Avoid_: rounding, thresholding (unqualified)

**Quantile zeroing**:
Zeroing, within each token row independently, every activation whose magnitude
is at most the row's q-quantile magnitude (its ⌊q·n⌋-th smallest), so each token
row reaches at least sparsity q regardless of batching; ties at the cutoff are
zeroed too.
_Avoid_: percentile threshold, dynamic ε

**Zeroed share**:
The fraction of observed activations newly set to zero by zeroing, excluding
values that were already zero.
_Avoid_: sparsity (unqualified), wiped values

**Local baseline**:
The locally served full-MMLU run with the zeroing hook active and nothing zeroed
(`mlp`, ε = 0; 81.98%), against which zeroing results are compared.
_Avoid_: control, original run
