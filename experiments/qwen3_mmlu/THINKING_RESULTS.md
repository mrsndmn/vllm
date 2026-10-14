# Thinking transfer evaluation results

## Hypothesis

The selected sparsity interventions may preserve mathematics and scientific
reasoning accuracy with a reasoning checkpoint and longer generation budget.

## Setup

The submitted source of truth is commit
`a13fc09b0db6514984e495ab489919f4529ae42d`:
`run.py:experiment_configs`, `run.py:prepare_snapshot`, and
`benchmark_eval.py:run_eval`. The current branch need not contain that launcher.
`benchmark_eval.py:grade` evaluates the final response after the reasoning block;
truncated generations count as incorrect. Comparisons use this protocol's own
baseline and repeat.

## Results

All 18 evaluations completed. Saved configurations, frozen benchmark manifests,
question coverage, paired prompts, and recomputed metrics passed validation.
Final-response extraction and correctness were independently rechecked for all
6,282 saved responses using the submitted grading code.

MATH500 baseline/repeat accuracy is 98.6%/97.8%; intervention accuracy ranges from
97.2% to 98.4%. GPQA baseline/repeat accuracy is 67.68%/70.71%; intervention
accuracy ranges from 65.15% to 70.71%. Embedding 90% reaches 98.2%/70.71% on the
two benchmarks; MLP 0.5 reaches 98.4%/68.69%.

Embedding + output 50% reaches 97.8%/65.66%, with 5/500 MATH500 and 19/198 GPQA
responses exhausting the generation limit. Of those, 5 and 18 respectively lack
a closing reasoning delimiter. All baseline GPQA responses finish within budget.

The ordered heatmap, CSV, validated summary and submission manifest are bundled
in `run/thinking32k/selected-nine-thinking32k-results.zip`; individual artifacts
and validation logs are in the same directory.

## Conclusions

The selected interventions retain high accuracy in this protocol. Small GPQA
improvements are within the observed baseline-repeat difference of 3.03
percentage points and do not establish a reliable gain. Joint embedding/output
pruning increases generation-limit exhaustion, especially on GPQA.

Comparisons with earlier instruct evaluations also change the checkpoint and
budget, so they do not isolate the effect of enabling reasoning.

## Changelog

- 2026-10-09: Validated all completed evaluations and recorded final results.
