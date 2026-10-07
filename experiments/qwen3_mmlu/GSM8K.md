# Generative GSM8K benchmark

## Hypothesis

The generative evaluation engine can measure mathematical answer accuracy on
GSM8K independently of MMLU and support subsequent activation-zeroing comparisons.

## Setup

`run.py:evaluate` and `run.py:job` call `chat_eval.py:run_eval`;
`gsm8k.py` loads questions and parses numeric answers. Benchmark and job settings,
including the pinned dataset revision, live in `run.py:experiment_configs`.

Local evaluation against a matching running server:

```bash
python experiments/qwen3_mmlu/run.py start --round-activations-eps 0
python experiments/qwen3_mmlu/run.py eval --name gsm8k_baseline --limit 3
python experiments/qwen3_mmlu/run.py eval --name gsm8k_baseline
```

MLS submission through the existing launcher:

```bash
/home/jovyan/.mlspace/envs/arkhip/bin/python experiments/qwen3_mmlu/run_jobs.py --filter 'gsm8k_*' --dry
/home/jovyan/.mlspace/envs/arkhip/bin/python experiments/qwen3_mmlu/run_jobs.py --filter 'gsm8k_*'
```

Aggregate completed full jobs separately from MMLU:

```bash
python experiments/qwen3_mmlu/chat_eval.py summarize --task gsm8k_chat
```

The model receives the question through its chat template, never the gold
solution. Numeric answers are compared exactly after normalizing thousands
separators and decimal formatting. Ambiguous answers, expressions, and truncated
forced continuations count as incorrect. This is our zero-shot generative
protocol, rather than a reproduction of another harness's few-shot score.

## Results

The local model smoke check answered all three sampled questions correctly;
all answers were parsed naturally, with no forced continuation. This validates
the integration, not full-test accuracy. The pinned dataset loaded successfully
and every gold answer was parsed. All 107 regression tests and pre-commit checks
passed; independent adversarial review returned APPROVE.

Full benchmark results are pending. MLS payloads and dispatch were tested with
the existing test fixtures; no jobs were submitted. A live launcher dry run
could not proceed because the existing MLS client environment lacks `regex`.

## Conclusions

The shared launcher and generation engine support an independent GSM8K baseline
and repeat. Further zeroing arms can be added through the source registry.

## Changelog

- 2026-10-07: Added GSM8K loading, numeric exact-match evaluation, registry rows,
  task-specific summaries, and regression checks.
- 2026-10-07: Completed the local model smoke check and independent review.
