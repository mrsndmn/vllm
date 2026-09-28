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

Stage A (2026-09-28, 50 profiling jobs, commit `747cfb715b`): increase in
per-token NLL of chat_baseline's responses when one group is zeroed (noise floor
from the unzeroed repeat: -0.00007). Full table: `adaptive-profile.csv`.

| Group (layers) | mlp 0.5 | mlp 0.75 | mlp 0.9 | block 0.25 | block 0.5 | block 0.75 | block 0.75 acc % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (0-5) | 0.0007 | 0.0042 | 0.0159 | 0.0045 | 0.0412 | 0.364 | 89.1 |
| 1 (6-11) | 0.0013 | 0.0077 | 0.0263 | 0.0072 | 0.0851 | 0.405 | 90.1 |
| 2 (12-17) | 0.0006 | 0.0083 | 0.0210 | 0.0085 | 0.0863 | 0.421 | 89.8 |
| 3 (18-23) | 0.0022 | 0.0146 | 0.0560 | 0.0081 | 0.0998 | 0.487 | 80.3 |
| 4 (24-29) | 0.0012 | 0.0082 | 0.0292 | 0.0090 | 0.0968 | 0.424 | 82.9 |
| 5 (30-35) | 0.0012 | 0.0096 | 0.0372 | 0.0104 | 0.0911 | 0.345 | 87.8 |
| 6 (36-41) | 0.0015 | 0.0106 | 0.0416 | 0.0098 | 0.0963 | 0.437 | 83.2 |
| 7 (42-47) | 0.0023 | 0.0101 | 0.0215 | 0.0057 | 0.0529 | 0.370 | 88.5 |

Accuracy of every mlp cell and every block cell below 0.75 stayed within the
baseline noise (91.4-93.8% vs 92.4-93.1%). Stage B schedules are in
`run.py:ADAPTIVE_SCHEDULES`; block at 0.5 allocated uniform 0.5. Stage C pending.

## Conclusions

Pending.

## Changelog

- 2026-09-28: Planned; per-group schedules, profiling rows, re-scoring, and the
  allocation search implemented.
- 2026-09-28: Stage A done; schedules registered and Stage C launched.
