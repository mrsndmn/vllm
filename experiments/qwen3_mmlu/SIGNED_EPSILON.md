# Signed epsilon replacement in MLPs

## Hypothesis

Replacing small MLP activations or expert weights with a sign-preserving
truncated-normal centroid reduces tensor reconstruction error and may preserve
more generative benchmark accuracy than replacing the same values with zero.

## Setup

`run.py:job`, with designs and launch settings in `run.py:experiment_configs`.
`rounding_stats.py:threshold` transforms routed post-SiLU-and-multiply activations;
`mlp_weight_rounding.py:apply` transforms loaded gate/up/down expert projections.
The normal coefficient is computed by `rounding_stats.py:normal_coefficient`.
Existing zeros remain zero; cutoff ties are included in both replacement modes.
Compare against matching existing zeroing arms and benchmark baselines.

## Results

The full local test suite passed, as did generative activation and expert-weight
smokes on the real model. Independent adversarial review approved the implementation.
The benchmark sweep has been submitted to MLSpace; full quality results are pending.
Completed baseline and activation-zeroing controls are reused.

## Conclusions

Pending. Tensor MSE improvements do not guarantee improvements in model accuracy.

## Changelog

- 2026-10-09: Added signed epsilon activation and expert-weight evaluation arms.
- 2026-10-09: Completed preflight and independent review; submitted the sweep
  and registered durable observation separately for each benchmark.
