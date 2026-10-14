# Quantile zeroing is per token row, with ties at the cutoff zeroed

Quantile zeroing computes the cutoff separately for every token row (one token at
one layer and zeroing site): the row's ⌊q·n⌋-th smallest magnitude, zeroing every
value at or below it. We chose this over a per-layer ε calibrated offline and over
a per-batch quantile because it gives every token the target sparsity and cannot
change with request batching or concurrency, so results are reproducible across
local and MLSpace runs. Ties at the cutoff are zeroed (bf16 expert outputs tie
often), so the zeroed share can slightly exceed q; the counters report the actual
share.

## Considered Options

- Calibrated per-layer ε: static and kernel-friendly, but sparsity only
  approximates q and needs a calibration set.
- Per-batch quantile: depends on batch composition, so not reproducible.
- Exactly ⌊q·n⌋ values per row: exact share, but which tied value survives is
  arbitrary.
