# Generative transfer benchmark suite

## Hypothesis

Activation zeroing selected on MMLU transfers to harder mathematics, scientific
reasoning, instruction following, broad knowledge, and executable coding tasks.

## Setup

`run.py:evaluate` dispatches the transfer registry to `benchmark_eval.py:run_eval`.
All dataset revisions, arms, generation budgets, and grader versions live in
`run.py:experiment_configs` and its source configuration constants.

`benchmark_data.py:prepare` freezes inputs and official evaluator files on shared
storage, with checksums verified before evaluation. Code reference outputs are
computed once from the trusted benchmark solutions. Generated code is graded
with EvalPlus's restricted subprocess execution and time limits in MLS containers.
The code grader is pinned to upstream source with the polynomial-root progress
counter fix; its checksum is verified without changing the installed package.
Do not execute generated code in the interactive workspace.

The suite uses mathematical equivalence for MATH-500, Google's strict/loose
prompt and instruction checks for IFEval, generated answer letters for MMLU-Pro
and GPQA Diamond, and base plus expanded executable tests for HumanEval+/MBPP+.
IFEval receives its original instructions. No task uses likelihood scoring or
forced-answer continuations in these six benchmark families. The long-context
RULER extension uses the official answer prefixes; see [RULER.md](RULER.md). Truncated generations count as incorrect.
These are our zero-shot chat protocols, not reproductions of few-shot leaderboards.

Prepare with the configured evaluation environment:

```bash
python experiments/qwen3_mmlu/benchmark_data.py
```

Use the existing MLS client environment for submission:

```bash
python experiments/qwen3_mmlu/run_jobs.py --filter 'suite_*' --dry
python experiments/qwen3_mmlu/run_jobs.py --filter 'suite_*'
```

`--limit N` creates separate smoke outputs and never marks a full evaluation done.
`benchmark_eval.py summarize` recomputes saved metrics and compares each task's
arms against its baseline and repeat. Artifacts remain outside the checkout.

## Results

All prepared input counts and checksums passed validation. Every mathematical
gold answer parses, every instruction configuration grades, and all prompts fit
the server context window. All six benchmark families passed MLS model smoke
checks, with complete artifacts and no truncated responses.

All 142 regression tests and pre-commit checks passed. Independent review identified
an upstream EvalPlus polynomial-root counter bug; the corrected upstream grader
is now pinned and a regression checks acceptance of an alternative valid root.
Fresh independent review returned APPROVE and validated the affected canonical
solution and representative MBPP special-oracle cases.

All 42 full MLS evaluations were submitted from commit `4fb9f0c8a6`. The launch
manifest, full job identifiers, submission logs, smoke results, and watcher state
are under the configured artifacts root at `launches/suite-4fb9f0c8a6/`.
The initial sweep completed 40 jobs; two MMLU-Pro evaluations failed during
HTTP response reads while shared NFS storage was full. Both completed successfully
on retry. The retry launch manifest and full job identifiers are recorded under
`launches/mmlu-pro-retry-20261008-091158/`.

For MMLU-Pro, the original same-suite baseline scored 73.0219% and its repeat
72.6729%. The completed retries scored 71.9581% for `block_q0.25` and 68.1433%
for `adapt_mlp_b0.8`: respectively -1.0638/-0.7148 and -4.8787/-4.5296 percentage
points against baseline/repeat. All four runs contain the same 12,032 questions
and identical comparison prompts. Configs, data manifests, IDs, source question
fields, and recomputed metrics passed validation; regrading all 48,128 saved
answers reproduced their stored predictions and correctness. Detailed scores
and paired correctness changes are saved as `validated-results.json` and
`validated-results.csv` in the retry launch directory.

Truncated responses were 915/927 for baseline/repeat, 1,077 for the block retry,
and 1,024 for the adaptive retry; they count as incorrect under this protocol.

## Conclusions

Both MMLU-Pro retries score below both original controls. The block penalty is
moderate; the transferred adaptive schedule loses substantially more quality.
The baseline repeat differs by 0.3491 percentage points on the fixed questions,
smaller than either observed retry penalty. These results include generation-cap
and answer-parsing failures, so they do not isolate reasoning quality alone.
The transferred adaptive schedule is held fixed without tuning on these datasets.

## Changelog

- 2026-10-07: Added the six datasets and existing seven arms to the MLS registry.
- 2026-10-07: Passed all model smoke checks and independent review; submitted the
  full transfer sweep and started its completion watcher.

- 2026-10-09: Validated successful MMLU-Pro retries against original controls,
  regraded all four runs, and recorded the completed comparisons.
