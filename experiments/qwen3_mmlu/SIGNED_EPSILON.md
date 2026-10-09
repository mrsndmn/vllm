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
MATH-500 completed with all 500 questions, matching source configs, manifests,
question content and prompts; saved aggregates were recomputed. The baseline
scored 89.20%. Signed activations changed accuracy by +2.20 and -0.80 points
against zero controls; signed expert weights by -2.00 and +6.20 points. The larger
weight gain was 46 fixed versus 15 broken answers (exact paired p=8.84e-5), surviving
Bonferroni correction across the 28 planned primary comparisons. Its accuracy was
86.80%, still below baseline. Truncated outputs dropped from 91 to 54: pairs with
at least one truncated output contributed +34 net correct answers; the 400 pairs
with neither output truncated contributed -3. This descriptive decomposition
shows the importance of the generation budget, without isolating a causal effect.
Full metrics, paired tests and decomposition are saved externally; the heatmap now
includes four completed benchmarks.
HumanEval+ completed with all 164 tasks, matching source configs, manifests,
question content and prompts. Saved aggregates and the base/plus pass conditions
were verified without rerunning generated code. The primary metric is pass@1 on
both base and extended tests; baseline scored 86.59%. Signed activations changed
accuracy by -2.44 and +2.44 points against zero controls; signed expert weights by
+3.05 and +4.88 points. The larger weight gain was 13 fixed versus 5 broken tasks
(exact paired p=0.0963). Both weight variants had two truncated outputs; all eight
net additional passes came from pairs where neither output was truncated. No
primary pair reaches unadjusted 0.05 significance on this benchmark. Full base/plus
metrics and paired/truncation comparisons are saved externally; the combined
heatmap includes five completed benchmarks.
Completed baseline and activation-zeroing controls are reused.

## Conclusions

The stronger tested weight transformation benefits from signed replacement on
all five completed benchmarks under the existing end-to-end evaluation protocol.
The large IFEval and MATH-500 gains involve fewer truncated outputs. MATH-500 does
not show a net improvement among pairs where both outputs finish within budget,
so its result does not establish better answer correctness independent of budget.
HumanEval+ also favors signed replacement of the stronger weight transformation
among pairs without truncation, although its paired evidence is not significant.
The lighter weight transformation and both activation variants vary by benchmark.
Wait for the remaining benchmarks and account for multiple comparisons. Tensor MSE
improvements do not guarantee improvements in model accuracy.

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
- 2026-10-09: All MATH-500 jobs completed; validated provenance and full metrics,
  recorded paired/truncation comparisons, and refreshed the four-benchmark heatmap.
- 2026-10-09: All HumanEval+ jobs completed; validated provenance and saved
  base/plus outcomes, recorded paired comparisons, and refreshed the five-benchmark heatmap.
