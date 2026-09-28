# Adaptive per-group activation zeroing

## Hypothesis

Layers differ in how much activation zeroing they tolerate. Choosing one
quantile per group of 6 consecutive layers (8 groups) from measured sensitivity
keeps accuracy closer to the unzeroed model than uniform quantile zeroing with
the same mean zero share, at both the `mlp` and `block` sites. The key case is
`block` at a 0.5 mean, where uniform zeroing lost 11.7 pp (5-shot) and 8.9 pp
(chat).

## Setup

All settings live in `run.py:experiment_configs` (`GROUPS`, `PROFILE_LEVELS`,
`ADAPTIVE_BUDGETS`, `ADAPTIVE_SCHEDULES`); zeroing semantics are in
[ROUNDING.md](ROUNDING.md).

- Stage A (`prof_*` rows, `chat_eval.py:run_eval` with `chat_reference`): zero
  one group at a time in the generative chat setup on exactly the 304
  `chat_baseline` STEM questions. Each job reports accuracy and answer changes
  and re-scores `chat_baseline`'s responses under the zeroed model
  (`chat_eval.py:score_reference`). Two unzeroed `prof_baseline*` runs give the
  reference NLL and the noise floor.
- Stage B (`sensitivity.py`): cost = increase in per-token NLL of the reference
  responses; exhaustive search picks per-group levels meeting each budget at the
  lowest summed cost (group effects assumed additive).
- Stage C (`adapt_*`, `chat_adapt_*` rows): adaptive schedules versus uniform
  controls at the same mean (`mlp_q0.6`-`0.8`, `block_q0.3`-`0.5`) on full 5-shot
  MMLU and on the chat check.

Block levels are in 0.25 steps, so the 0.3 and 0.4 budgets are met at means of
0.3125 and 0.40625, slightly above the uniform controls (a conservative bias
against the adaptive arm). Re-scoring counts every model-written token,
including the end of a response that stopped by itself, but not the injected
forced-answer suffix; the cost subtracts the unzeroed repeat's difference as
noise, and ties go to the lower unclamped loss, then the more even schedule.

Stage A tunes on the chat check's own questions (user decision), so Stage C's
chat comparison is in-sample; the full 5-shot test (14,042 questions) is the
largely independent check.

## Results

Pending.

## Conclusions

Pending.

## Changelog

- 2026-09-28: Planned; per-group schedules, profiling rows, re-scoring, and the
  allocation search implemented.
