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
GSM8K also completed, with matching prompts/answers for all 1,319 questions and
recomputed result aggregates verified. The baseline scored 95.15%. Signed
activations gained 0.23 and 0.99 points against zero controls; signed expert
weights changed accuracy by -0.23 and +1.44 points. The larger weight gain
corresponds to 41 fixed versus 22 broken answers (exact paired p=0.0226 before
multiplicity correction). It does not survive Bonferroni correction over the four
GSM8K comparisons. Parsing failures remain included in accuracy. Summaries and
paired tests are saved externally, along with a combined MMLU/GSM8K heatmap.
IFEval completed with all 541 questions, matching source configs, dataset manifests,
question content and prompts; saved metric aggregates were recomputed. The primary
metric is strict prompt accuracy; the baseline scored 83.55%. Signed activations
changed accuracy by -0.74 and +2.77 points against zero controls, and signed expert
weights by -0.55 and +10.35 points. The larger weight gain was 73 fixed versus 17
broken answers (exact paired p=1.95e-9), surviving Bonferroni correction across the
28 planned primary zero-versus-epsilon comparisons. Truncated outputs dropped from
65 to 8; 46 of the 56 net additional correct answers came from pairs with at least
one truncated output, and 10 from pairs where neither output was truncated.
Truncations count as failures under the existing evaluation protocol. Strict/loose
prompt and instruction metrics, paired comparisons and truncation decomposition
are saved externally; the combined heatmap now includes three completed benchmarks.
Completed baseline and activation-zeroing controls are reused.

## Conclusions

The stronger tested weight transformation benefits from signed replacement on
all three completed benchmarks. IFEval shows a large gain, with most of the net
improvement coming from pairs involving a truncated output. The lighter weight
transformation is inconsistent; the stronger activation transformation improves
all three benchmarks, while the lighter activation transformation is inconsistent.
These results do not establish a universal advantage; wait for the remaining
benchmarks and account for multiple comparisons. Tensor MSE improvements do not
guarantee improvements in model accuracy.

## Changelog

- 2026-10-09: Added signed epsilon activation and expert-weight evaluation arms.
- 2026-10-09: Completed preflight and independent review; submitted the sweep
  and registered durable observation separately for each benchmark.
- 2026-10-09: All MMLU STEM jobs completed; validated matching questions and
  recorded accuracy differences, paired comparisons, and a heatmap.
- 2026-10-09: All GSM8K jobs completed; validated full records and aggregates,
  recorded paired comparisons, and refreshed the combined completed-benchmark heatmap.
- 2026-10-09: All IFEval jobs completed; validated provenance and full metrics,
  recorded paired/truncation comparisons, and refreshed the three-benchmark heatmap.
