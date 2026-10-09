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
MMLU STEM completed successfully. Against matching zero controls, signed expert
weights gained 1.64 and 2.96 percentage points; signed activations changed accuracy
by -0.33 and +0.66 points. All runs used the same 304 questions, with no unparsed
answers. The baseline scored 92.76%. Exact paired tests do not reject equal accuracy
at the 0.05 level (the largest weight gain has p=0.0636, without multiplicity
correction). Detailed results and paired comparisons are saved with the sweep's
external artifacts. Other benchmark wait groups continue independently.
Completed baseline and activation-zeroing controls are reused.

## Conclusions

The first benchmark favors signed replacement for expert weights, while the
activation differences are small. This is preliminary evidence from one benchmark;
wait for the remaining benchmarks before drawing a general conclusion. Tensor MSE
improvements do not guarantee improvements in model accuracy.

## Changelog

- 2026-10-09: Added signed epsilon activation and expert-weight evaluation arms.
- 2026-10-09: Completed preflight and independent review; submitted the sweep
  and registered durable observation separately for each benchmark.
- 2026-10-09: All MMLU STEM jobs completed; validated matching questions and
  recorded accuracy differences, paired comparisons, and a heatmap.
