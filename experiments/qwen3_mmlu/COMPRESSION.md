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

Pending first screening run.

## Conclusions

Use held-out matrix and complete-expert error curves to select candidates for
subsequent generative evaluation and serving benchmarks. Small local errors alone
do not establish that errors remain small across the complete model.

## Changelog

- 2026-10-06: Add routed-activation screening and storage-aware comparisons.
