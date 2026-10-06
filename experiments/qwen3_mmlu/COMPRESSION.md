# Expert weight compression screening

## Hypothesis

Expert weights with small stable rank may permit useful compression through
pruning or low-rank factorization. Activation-aware compression should preserve
held-out outputs better than compression based on weights alone.

## Setup

`compression.py:capture`, `compression.py:screen`, and `compression.py:plot`.
All model, corpus, selection, storage, and analysis settings live in the script's
`experiment_configs` registry. Submit with `run_jobs.py --analysis`; the existing
launcher stages committed code and deduplicates queued and completed runs.

Capture uses disjoint corpus splits and uniform reservoir samples of tokens
actually routed to each expert. Screening compares magnitude pruning, per-output
Wanda pruning, ordinary SVD, and damped covariance-whitened SVD. The latter uses
the whitening principle of SVD-LLM, without its sequential parameter update.

Storage comparisons assume BF16 factors versus BF16/int32 CSR, including row
pointers. These are representation byte estimates, not measurements of serving
memory or speed. Replay uses FP32 arithmetic on the original BF16 parameters and
stored BF16 factors. Each trial compresses one matrix and also measures the
resulting complete expert output through its nonlinearity. It does not compress
all three projections simultaneously or evaluate full-model generation.

The sample deliberately spans rank strata and routing extremes; its correlations
are exploratory, not unbiased population estimates. Experts without sufficient
observations in either split are recorded as insufficiently covered.

## Results

Completed screening produced 720 trials for 45 matrices in 15 experts. Across
these matrices, median stable rank was 65.4, but median rank needed to retain 99%
of squared singular-value energy was 714 (range 456–728).

At half the dense representation bytes, median complete-expert relative output
errors for gate/up/down were 48.0%/56.8%/55.1% for magnitude pruning,
40.7%/44.7%/49.1% for Wanda, 29.0%/37.3%/44.5% for ordinary SVD, and
24.9%/23.6%/37.3% for covariance-whitened SVD. Whitened SVD reached error below
10% for six of the 45 matrices at this budget; all six belonged to two sampled
experts in the first layer. The expert error measures one compressed projection
at a time, not the jointly compressed branch.

Eligible coverage was 114/128, 83/128, and 84/128 experts in the sampled layers;
remaining experts did not satisfy calibration/held-out sample requirements.
Results are conditional on this corpus, coverage, and stratified expert selection.

Artifacts: `results.csv`, `coverage.json`, `capture.json`, `screen.json`,
`compression-error.png`, and `rank-vs-compression.png` in the registry's output
directory. Source/config fingerprints and result hashes were checked.

Successful MLS run: `lm-mpi-job-8c3d8e9c-c9bd-41fe-b6d9-cfcdeb118505`,
source commit `15f9580613991e0cd233f736e04eda7ed7db5207`.
The initial run `lm-mpi-job-47493559-8d63-4298-b7ed-a6980fd17fd3` failed during
FP32 covariance Cholesky; its partial observations are excluded. FP64 whitening
resolved the failure on the repeated run.

## Conclusions

Small stable rank does not imply a short spectral tail or uniformly cheap
compression. Activation-aware factorization is the strongest tested method here,
but aggressive storage reduction often changes expert outputs substantially.

Next, test selective compression of the locally tolerant projections against a
paired generative baseline, expand corpus/rare-expert coverage, and check jointly
compressed projections before extending to every layer. Serving measurements
require actual stored representations and kernels. Small local errors alone do
not establish that errors remain small across the complete model.

## Changelog

- 2026-10-06: Add routed-activation screening and storage-aware comparisons.
- 2026-10-06: Complete screening, stabilize whitening, and record held-out results.
