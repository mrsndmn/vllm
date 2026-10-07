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
forced-answer continuations. Truncated generations count as incorrect.
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
the server context window. The first five benchmark families passed MLS model
smoke checks. Full evaluations and the MBPP+ model smoke check are pending.

The regression suite and pre-commit checks passed. Independent review identified
an upstream EvalPlus polynomial-root counter bug; the corrected upstream grader
is now pinned and a regression checks acceptance of an alternative valid root.

## Conclusions

Compare paired correctness changes with baseline repeat variation. The transferred
adaptive schedule is held fixed, without tuning on these evaluation datasets.

## Changelog

- 2026-10-07: Added the six datasets and existing seven arms to the MLS registry.
