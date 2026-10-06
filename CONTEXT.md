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

**Residual r**:
The stream a layer's MoE output is added to: the residual stream after that
layer's attention and before its MoE add (vLLM's `residual`). The `block` site
tests r plus the MoE output; the `moe_out` site leaves r untouched.
_Avoid_: residual only stream, residual (unqualified), skip branch

**Expert outputs**:
The down-projection outputs of a token's selected experts at one layer, each taken
before its routing weight is applied; the MoE output is their weighted sum.
_Avoid_: moe_expert, expert activations (the `mlp` site's SiLU-and-multiply outputs)

**Token row**:
All activations of one token at one layer and site: its 8 selected experts'
SiLU-and-multiply outputs (8 × 768) for `mlp`, its hidden-size residual vector
for `block`, or its hidden-size MoE output for `moe_out`. The hidden-state viewer
also has `residual` rows (residual r, hidden size) and `experts` rows (the expert
outputs, 8 × hidden size).
_Avoid_: activation vector, sample

**Hidden state**:
One token's residual-stream vector (hidden size 2048) at one depth: state 0 is
the embedding output and state ℓ (1–48) is the residual stream after decoder
layer ℓ−1, all before the final norm, so there are 49 per token. After a layer
this is the same sum the `block` site tests.
_Avoid_: HF output_hidden_states (its last entry is after the final norm), activation (unqualified)

**Token stats**:
The nine numbers summarising one token row (a hidden state or another site's
row) of n values: the signed minimum, maximum, mean, and median of its values,
and the q10, q25, q50, q75, and q90 magnitude quantiles, each the row's ⌊q·n⌋-th
smallest |x| as in quantile zeroing.
The signed median and the magnitude q50 are therefore different numbers.
_Avoid_: percentiles (unqualified), activation stats

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

**Generative evaluation**:
Evaluation in which the model writes its answer through the chat template and
the answer is parsed from the generated text; the only kind used from 2026-09-30
(`docs/adr/0004`).
_Avoid_: likelihood evaluation, 5-shot scoring (retired)

**Local baseline** (historical):
The locally served 5-shot letter-likelihood full-MMLU run with the zeroing hook
active and nothing zeroed (`mlp`, ε = 0; 81.98%), against which the retired
5-shot results were compared. New results compare with the chat baseline.
_Avoid_: control, original run

**Generative check**:
The seeded 304-question MMLU STEM evaluation in which the chat model writes its
reasoning and then an answer letter; its arms are compared with the unzeroed chat
baseline (92.76%).
_Avoid_: chat eval (unqualified), generative MMLU

**Reference response**:
A finished run's recorded generated text for one question (its reasoning, plus
the forced-answer suffix and letter if it was forced), fed back to the model
token by token instead of being generated again.
_Avoid_: rerun, regenerated sample

**Noise floor**:
The number of answers that change between two identical unzeroed generative-check
runs (9), the change expected from batching nondeterminism alone.
_Avoid_: variance, error bar

**Weight report**:
A view of checkpoint weights by decoder layer, tensor family, and MoE expert,
with distributions and matrix characteristics; it describes parameters rather
than token-dependent activations.
_Avoid_: activation viewer, hidden-state viewer

**Weight stable rank estimate**:
The squared Frobenius norm of a checkpoint matrix divided by an estimate of its
largest squared singular value; it measures spectral concentration rather than
the number of nonzero singular values.
_Avoid_: numerical rank, exact rank, effective rank (unqualified)
