# Qwen3 MMLU activation-zeroing experiments

Local experiments measuring how Qwen3-30B-A3B's MMLU accuracy changes when small
intermediate activations are set to zero during inference.

## Language

**Zeroing site**:
The place in each layer where activations are tested and zeroed: `mlp` (selected
experts' SiLU-and-multiply outputs before down projection), `block` (the FP32
residual-stream sum before the next add/RMSNorm), or `moe_out` (the layer's MoE
output alone).
_Avoid_: rounding site, hook, location

**MoE output**:
A layer's router-weighted sum of its selected experts' outputs, before it is added
to the residual stream; the `moe_out` site tests and zeroes it alone, leaving the
residual stream untouched.
_Avoid_: block residual, residual branch, block output

**Token row**:
All activations of one token at one layer and zeroing site: the concatenated
outputs of its 8 selected experts for `mlp`, its hidden-size residual vector
for `block`, or its hidden-size MoE output for `moe_out`.
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

**Total zero share**:
The fraction of observed activations that are zero after zeroing, including
values that were already zero; quantile zeroing guarantees at least q of it in
every token row.
_Avoid_: sparsity (unqualified)

**Local baseline**:
The locally served full-MMLU run with the zeroing hook active and nothing zeroed
(`mlp`, ε = 0; 81.98%), against which zeroing results are compared.
_Avoid_: control, original run

**Generative check**:
The seeded 304-question MMLU STEM evaluation in which the chat model writes its
reasoning and then an answer letter; its arms are compared with the unzeroed chat
baseline (92.76%).
_Avoid_: chat eval (unqualified), generative MMLU

**Noise floor**:
The number of answers that change between two identical unzeroed generative-check
runs (9), the change expected from batching nondeterminism alone.
_Avoid_: variance, error bar
