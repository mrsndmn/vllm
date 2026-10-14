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

`watch_jobs.py` consumes the launch manifest, polls MLS, and writes `status.json`
in its state directory. It triggers `event.json` on the first failed/stopped/
terminated/deleted job or when every job completes successfully; API errors are
recorded and retried. An optional Arkhip resume config binds the event to the
original session and Telegram Topic. The bot token is passed only through the
environment or a private FIFO. A per-directory lock prevents duplicate watchers,
and a completed event prevents duplicate resume calls. Raised callback errors are
recorded and retried; restart recovers an event whose callback did not return.
The watcher runs detached
in tmux, independently of the interactive worker.

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
the existing test fixtures. The missing `regex` dependency in the MLS client
environment was installed before preparing submission of the selected sweep.
All seven selected jobs were submitted from commit `1fa8ce7561`; MLS reports
them running and each job has created its vLLM server log. The expanded suite
passed all 113 tests, pre-commit, and independent adversarial review.

Submission manifest and logs are under the configured artifacts directory at
`launches/gsm8k-1fa8ce7561/`. Complete MLS job identifiers:

- `gsm8k_baseline`: `lm-mpi-job-9cd1e00c-edbc-49a8-8026-af480a7dd273`
- `gsm8k_baseline_repeat`: `lm-mpi-job-b5e5f68a-b484-4c05-a061-ea2d058fdf88`
- `gsm8k_mlp_q0.5`: `lm-mpi-job-071dd433-4e30-4222-bc9f-961f58447d90`
- `gsm8k_mlp_q0.8`: `lm-mpi-job-74449975-8e4d-4cc0-9265-3762239dea98`
- `gsm8k_moe_out_q0.4`: `lm-mpi-job-993ab992-4879-4cdd-b716-748e5b18efdf`
- `gsm8k_block_q0.25`: `lm-mpi-job-1ebc442a-2335-4c49-a166-243775651c5c`
- `gsm8k_adapt_mlp_b0.8`: `lm-mpi-job-03884c77-29c1-4eb6-97a0-dd04707e5640`

## Conclusions

The shared launcher and generation engine support an independent GSM8K baseline
and repeat plus selected uniform/adaptive zeroing arms transferred from MMLU.
The adaptive schedule is reused without tuning on GSM8K.

## Changelog

- 2026-10-07: Added GSM8K loading, numeric exact-match evaluation, registry rows,
  task-specific summaries, and regression checks.
- 2026-10-07: Completed the local model smoke check and independent review.
- 2026-10-07: Registered the selected MMLU zeroing variants for transfer to GSM8K.
- 2026-10-07: Submitted all seven full GSM8K jobs and verified MLS running status.
- 2026-10-07: Added a terminal-condition watcher with automatic conversation resume.
