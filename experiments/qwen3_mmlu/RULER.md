# Long-context activation zeroing

## Hypothesis

Activation zeroing selected on short MMLU inputs preserves retrieval, variable
tracking, aggregation, and question answering when evidence is far from the query.

## Setup

`run.py:experiment_configs` defines the long-context rows. `ruler.py:prepare`
invokes unchanged, pinned NVIDIA RULER v1 generators; `benchmark_data.py:prepare`
freezes shared questions and source provenance with checksums. The adapter adds
Qwen's chat template followed by the upstream assistant answer prefix. It measures
actual token counts, reserves space for each task's upstream generation budget,
and rejects overflowing inputs without truncating evidence. Source settings,
lengths, samples, seeds, package versions, and server parameters live in code.

Use the configured evaluation environment for preparation and the existing MLS
client environment for `run_jobs.py --filter 'suite_ruler_*'`. `--dry` previews
payloads; `--limit N` produces separate smoke artifacts. Inputs alternate between
subtasks so a smoke batch can cover every task.

`benchmark_eval.py:run_eval` generates continuations and saves individual task
scores. Grading follows RULER v1: case-insensitive substring recall for retrieval
and aggregation, and any matching answer alias for QA. Matched answers retain
credit when generation reaches its cap, as in the official metric. Overall score
is the mean of subtask means; all-answer correctness and capped generations are
reported separately. `benchmark_eval.py summarize` compares each context length
with its own baseline and repeat. This is a controlled transfer experiment with
our chat protocol, not an exact reproduction of published leaderboard runs.

## Results

Preparation, local checks, model smoke runs, and full submission are in progress.

## Conclusions

Compare both the macro score and individual subtask scores across context lengths;
use baseline repeat variation when interpreting zeroing effects.

## Changelog

- 2026-10-07: Added RULER to the existing generative evaluation and MLS paths.
