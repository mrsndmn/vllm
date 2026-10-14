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
external artifacts.
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
GPQA Diamond completed with all 198 questions, matching source configs, manifests,
question content and prompts. Saved answer parsing/correctness was recomputed from
the generated responses, and all metric aggregates matched. Baseline scored 44.95%.
Signed activations changed accuracy by -0.51 and +7.07 points against zero controls;
signed expert weights by -0.51 and +11.62 points. The larger weight gain was 39 fixed
versus 16 broken answers (exact paired p=0.00267); it survives correction over four
GPQA pairs but not over all 28 planned primary comparisons. For that weight pair,
truncations dropped from 84 to 41 and unparsed answers from 100 to 48 (overlapping
categories, included as failures). Pairs with neither output truncated contributed
+10 net correct answers, and pairs with at least one truncation contributed +13.
The activation gain has unadjusted p=0.0488 and does not survive correction. Results
reflect both answer correctness and completion/parsing behavior under the existing
protocol. Full metrics and paired/decomposition results are saved externally; the
heatmap includes six completed benchmarks.
MBPP+ completed with all 378 tasks, matching source configs, manifests,
question content and prompts. Saved metric aggregates and the base/plus pass
conditions were verified without rerunning generated code. The baseline scored
75.66% pass@1 on both base and extended tests. Signed activations changed accuracy
by +1.59 and +0.53 points against zero controls; signed expert weights by -0.79 and
+1.85 points. The larger weight gain was 19 fixed versus 12 broken tasks (exact
paired p=0.281). Truncated outputs dropped from five to zero, but all seven net
additional passes came from pairs where neither output was truncated. No primary
pair reaches unadjusted 0.05 significance on this benchmark.
All 42 newly submitted jobs completed successfully; completed baseline and
activation-zeroing controls were reused. The final external summaries cover all
seven benchmarks and all 28 paired primary comparisons, including Bonferroni
adjustments. Only the larger weight gains on IFEval and MATH-500 survive that
correction. The final combined heatmap and per-benchmark CSVs are saved externally
and copied to the local run artifacts.

## Conclusions

Signed replacement of the stronger tested weight transformation beats matching
zero controls on all seven benchmarks, with gains from 1.44 to 11.62 percentage
points under the existing end-to-end evaluation protocol. The IFEval and MATH-500
paired gains survive correction over all 28 planned primary comparisons. Their
large gains involve fewer truncated outputs: MATH-500 has no net improvement
among pairs where both outputs finish within budget, so this result does not
establish better answer correctness independent of budget. GPQA likewise combines
fewer truncated/unparsed outputs with improved accuracy; its gain does not survive
the full-sweep correction. HumanEval+ and MBPP+ have positive stronger-weight gains
among pairs without truncation, but their paired evidence is not significant.
The lighter weight transformation and both activation variants vary by benchmark.
Tensor MSE improvements therefore do not guarantee improvements in model accuracy,
and signed replacement is not a universal improvement across the tested settings.
A useful follow-up would repeat the stronger-weight comparison with multiple seeds
and a larger generation budget to assess stability and completion effects.

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
- 2026-10-09: All GPQA Diamond jobs completed; validated provenance, parsing
  and metrics, recorded paired comparisons, and refreshed the six-benchmark heatmap.
- 2026-10-09: All MBPP+ jobs completed; validated provenance and saved base/plus
  outcomes, finalized seven-benchmark results and all 28 paired comparisons.
