# Activation zeroing experiment

## Hypothesis

Setting small intermediate activations to zero may create sparsity without a
material change in MMLU accuracy. Expert activations and complete transformer
block outputs are separate experiments. Quantile zeroing tests whether a fixed
per-token sparsity (25% or 50% of each token row) is tolerated at either site.
The `moe_out` site (added later) zeroes a layer's MoE output alone, separating
sparsification of one layer's update from wiping the accumulated stream.

## Setup

`run.py:serve`, `run.py:job`, and `run.py:evaluate` implement the experiment.
Model selection, zeroing arms, and benchmark settings live in
`run.py:experiment_configs`. All evaluation is generative (the chat check below;
`docs/adr/0004`); the 5-shot likelihood harness, its local sweep, and its
artifacts were removed on 2026-09-30, so 5-shot numbers below are history. The
wrapper is pinned to the installed vLLM version; it does not modify the shared
installation or import this checkout's vLLM.

From the repository root, using the documented Python environment:

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py restart --round_activations_eps 1e-4 --round-activations-site mlp
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py status
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py stop
```

Use `block` or `moe_out` for the other sites. Both hyphenated and underscored epsilon flags
are accepted. Omit the epsilon flag on restart to restore the original server.
The existing [serving README](README.md) describes the port, persistence, and logs.
`start` remains idempotent; use `restart` to change a running server's settings.

`mlp` zeros the post-SiLU-and-multiply intermediate values of selected experts,
before their down projection. Each token's denominator includes all its selected
experts. `block` tests the combined residual stream in FP32, matching the sum
inside fused add/RMSNorm. Both residual operands are zeroed at selected positions;
all other operands retain their original bit patterns. This avoids introducing
an extra BF16 rounding operation throughout the residual stream.

`moe_out` tests each token's MoE output alone (the router-weighted sum of its
selected experts, 2048 values, before it is added to the residual stream) and
zeroes selected entries of it only; the residual stream keeps its original bits,
so the next fused add computes r + h' instead of r + h. It isolates sparsifying
one layer's update from wiping the accumulated stream, which `block` does (see
`docs/adr/0003-moe-out-site-tests-the-moe-output-alone.md`).

### Quantile zeroing (MLSpace jobs)

`--round-activations-quantile q` (q in [0, 1), exclusive with a nonzero
epsilon) replaces the constant with a per-token-row cutoff: the row's
floor(q * width)-th smallest magnitude (width 6144 for `mlp`, 2048 for `block` and `moe_out`).
Nonzero finite values at or below it are zeroed, ties included, so each row ends
with at least a q share of zeros; values already zero count toward q. The rule is
independent of batching. See `docs/adr/0001-per-token-quantile-zeroing.md`.

Registry rows with `launcher="mls"` run as MLSpace jobs, one `a100.1gpu` job per
row. `run_jobs.py` needs the `mls` client environment; each job runs
`run.py job --name <row>` from a per-commit snapshot of this directory, serves
that row inside the container, runs the generative check with the same settings
as a local `run.py eval --name <row>`, and writes `jobs/<row>/done.json` last. Job containers use the environment
path `/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126` because `~/.mlspace` is
not mounted there (see `docs/adr/0002-single-registry-and-subdir-staging-for-mls-jobs.md`).

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py restart --round-activations-quantile 0.5 --round-activations-site mlp
M=/home/jovyan/.mlspace/envs/arkhip/bin/python
$M experiments/qwen3_mmlu/run_jobs.py --dry                  # print payloads only
$M experiments/qwen3_mmlu/run_jobs.py --filter chat_mlp_q0.5 --limit 1   # smoke job
$M experiments/qwen3_mmlu/run_jobs.py                        # submit pending rows
```

The launcher refuses uncommitted tracked changes in this directory, skips rows
with a done marker (unless `--force`) or an identical queued/running job, and
stages `experiments/qwen3_mmlu` once per commit under `<artifacts>/staging/`.
Smoke runs (`--limit N`) use `jobs-smoke/limitN/` instead of `jobs/`.
Summaries report the total zero share (newly zeroed plus already zero) per
layer; quantile zeroing guarantees it is at least q.

### Generative chat check (MLSpace jobs)

Every registry row (`task="mmlu_stem_chat"`) tests whether zeroing is safe for
the deployed chat model. `chat_eval.py` samples 16 test questions per MMLU STEM subject (19
subjects, 304 questions, fixed seed) and asks each zero-shot through the model's
chat template. The prompt asks for short step-by-step reasoning ending with
`Answer: $LETTER`. Decoding is greedy with at most 1024 reasoning tokens. A
response that hits the cap is continued in the same assistant turn with
`[reasoning interrupted] Answer:` and at most 5 tokens; a finished response
without an answer line gets `Answer:` the same way. The letter is parsed from
the last `Answer:` line or from the forced continuation; unparsed answers count
as wrong. Qwen3-30B-A3B-Instruct-2507 has no thinking mode (its chat template
has none); `enable_thinking=False` is passed anyway. `chat_baseline` serves the
wrapper with nothing zeroed so the four quantile arms share its protocol;
`chat_baseline_repeat` repeats it to measure answer changes from batching
nondeterminism alone. Requests retry up to 3 times.

Fixed-ε arms `chat_{mlp,block,moe_out}_eps{0.01,0.1,0.5}` and quantile arms
`chat_moe_out_q{0.4,0.5,0.7,0.8,0.9,0.95,0.99}` use the same protocol. The moe_out
quantile arms pair with `chat_block_q0.4`/`chat_block_q0.5` at the same per-token
share, and 0.7-0.99 reach the levels of the high mlp arms; ε arms zero
different shares at different sites, so they are compared through their measured
per-layer zero shares, not as matched pairs.

```bash
$M experiments/qwen3_mmlu/run_jobs.py --filter 'chat_*' --dry
$M experiments/qwen3_mmlu/run_jobs.py --filter chat_baseline --limit 1   # smoke
$M experiments/qwen3_mmlu/run_jobs.py --filter 'chat_*'
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/chat_eval.py summarize
```

### Thresholds and counters

The interval is inclusive: nonzero finite values with `abs(x) <= epsilon` become
zero. Existing zeros are counted separately. Epsilon zero is an exact no-op.
NaNs and infinities are retained. This is experimental eager-mode instrumentation,
not a sparse acceleration kernel; zeroing does not skip dense matrix operations.

Successful completion/chat responses include `rounding_statistics`. Streaming
responses emit one extra JSON data event with empty `choices` immediately before
`[DONE]`. Statistics aggregate all prompts/choices in that HTTP request:

- Per-layer total floats, newly zeroed floats, already-zero floats, and percentage
  of total floats newly zeroed.
- Processed token positions, tokens with at least one newly zeroed value, and
  tokens whose observed activation vector is entirely zero after thresholding.
- Cached token positions omitted at initial scheduling and number of engine
  sequences represented in the response.

Counts cover executed prefill and decode positions. Cached prefixes are excluded;
recomputation after preemption counts again. The final sampled token does not pass
through the model unless another token is generated. These are computational
counts, not the response's `usage.total_tokens`. Warmup and dummy profiling are
excluded. Requests receive unique generated IDs, replacing supplied request IDs.

`GET /rounding/config` reports the active site, epsilon, and statistics database.
`GET /rounding/stats` aggregates all executed requests since server startup,
including canceled requests. Databases and benchmark artifacts remain outside
the checkout under the artifact root specified in `run.py`.

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python -m pytest experiments/qwen3_mmlu/test_rounding.py --confcutdir=experiments/qwen3_mmlu -q
# Generative check of one row against a local server serving the same zeroing:
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py eval --name chat_mlp_q0.5 --limit 1
```

Each evaluation records the active server configuration and statistics
before/after. Before/after benchmark counter snapshots cover all server traffic
in that interval. Run without unrelated inference when interpreting them as MMLU
sparsity; per-response counters remain isolated even under concurrent traffic.

`validate_live.py capture` preserves original-server reference completions.
`validate_live.py check` exercises the running instrumented server, including
chunked prompts, cached prefixes, concurrent requests, multiple prompts/choices,
chat, streaming, and errors. Fresh cache salts isolate epsilon-zero parity checks
from differences in cached versus uncached execution shapes.

## Results

At the time, thirteen CPU correctness/API/sweep-resume/session tests passed. Live GPU checks
pass for both locations with zero and nonzero thresholds, including exact
original completion/logprob parity at zero, chunked prefill, cached prefixes,
concurrent requests, multiple prompts/choices, chat, streaming, and error
handling. The implementation passed independent adversarial review.

Historical: full 5-shot letter-likelihood MMLU sweep, completed on 2026-09-28
(14,042 questions per run, all samples validated; harness and artifacts removed
2026-09-30). "Zeroed" is the share of all
observed activation floats newly set to zero; "flips" counts changed predicted
answers versus `mlp` epsilon 0 (fixed/broken).

| Experiment | MMLU % | Hum. | Soc. | STEM | Other | Zeroed % | Flips (fixed/broken) |
|---|---:|---:|---:|---:|---:|---:|---:|
| original server | 81.98 | 75.11 | 90.06 | 82.84 | 83.52 | — | — |
| mlp 0 | 81.98 | 75.11 | 90.12 | 82.87 | 83.42 | 0.00 | — |
| mlp 1e-4 | 82.01 | 75.20 | 90.02 | 82.94 | 83.46 | 0.46 | 172 (72/68) |
| mlp 1e-3 | 82.15 | 75.32 | 90.15 | 83.13 | 83.55 | 2.62 | 171 (78/55) |
| mlp 1e-2 | 82.10 | 75.18 | 89.99 | 83.35 | 83.49 | 14.90 | 212 (92/76) |
| mlp 1e-1 | 81.60 | 75.07 | 88.76 | 82.75 | 83.23 | 60.39 | 826 (314/368) |
| block 1e-2 | 82.22 | 75.77 | 89.63 | 83.35 | 83.52 | 6.80 | 347 (151/117) |
| block 1e-1 | 24.46 | 24.55 | 22.75 | 26.17 | 24.27 | 75.71 | 10736 (697/8775) |

Zeroing is concentrated in early layers: at block 1e-2, layer 0 loses 46.7% and
the last layer 0.5%; at block 1e-1, layers 0-2 lose about 99%; at mlp 1e-1, the
per-layer share ranges from 99.2% (layer 0) to 23.4% (layer 47). The original
server versus mlp 0 differs in 16 predictions with identical accuracy.
Its aggregate and per-layer CSVs were deleted with the harness artifacts.

Historical: quantile zeroing sweep (MLSpace jobs, commit `0c11e87412`, full
5-shot letter-likelihood MMLU, all samples validated; artifacts deleted
2026-09-30). "Total zero" is the zero share after zeroing including
already-zero values; every layer of every run lies in [q, q + 0.11 pp]. Flips are
changed answers versus the local baseline (`mlp` epsilon 0, 81.98%).

| Experiment | MMLU % | Hum. | Soc. | STEM | Other | Total zero % | Flips (fixed/broken) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp q=0.25 | 82.15 | 75.47 | 90.02 | 83.03 | 83.55 | 25.05 | 200 (89/66) |
| mlp q=0.5 | 81.90 | 75.09 | 89.60 | 83.00 | 83.46 | 50.09 | 401 (153/165) |
| block q=0.25 | 81.47 | 74.54 | 89.47 | 82.24 | 83.26 | 25.02 | 774 (276/348) |
| block q=0.5 | 70.33 | 60.98 | 80.70 | 70.63 | 73.93 | 50.02 | 3231 (521/2157) |

No MLSpace baseline job was run; comparisons use the local baseline.

Generative chat check (MLSpace, commit `85b77482af`, 304 seeded STEM test
questions, greedy, 1024-token reasoning cap). "Changed" counts answers differing
from `chat_baseline`; `chat_baseline_repeat` gives the noise floor from batching
nondeterminism. The last column scores the same 304 questions with the historical 5-shot
letter-likelihood harness runs above.

| Arm | Accuracy % | Natural / forced after cap | Mean reasoning tokens | Changed (fixed/broken) | Total zero % | Same questions, 5-shot likelihood % |
|---|---:|---:|---:|---:|---:|---:|
| chat_baseline | 92.76 | 297 / 7 | 206 | — | 0 | 83.55 |
| chat_baseline_repeat | 93.09 | 299 / 5 | 199 | 9 (4/3) | 0 | — |
| chat_mlp_q0.25 | 92.11 | 298 / 6 | 208 | 10 (3/5) | 25.04–25.06 | 84.21 |
| chat_mlp_q0.5 | 92.76 | 299 / 5 | 201 | 14 (6/6) | 50.07–50.10 | 84.21 |
| chat_block_q0.25 | 91.78 | 297 / 7 | 208 | 17 (6/9) | 25.01–25.04 | 84.87 |
| chat_block_q0.5 | 83.88 | 284 / 20 | 288 | 42 (5/32) | 50.01–50.03 | 66.45 |

No answer was unparsed and no finished response lacked an answer line.
Summary: `rounding-chat-sweep-summary.csv` in the artifact root.

High mlp quantiles (commit `97a3a3e818`, same setup). "Forced" counts answers
forced after the reasoning cap plus finished responses without an answer line
(q0.99 only: 2).

| Arm | Accuracy % | Natural / forced | Mean reasoning tokens | Changed (fixed/broken) | Total zero % |
|---|---:|---:|---:|---:|---:|
| chat_mlp_q0.8 | 91.12 | 300 / 4 | 175 | 15 (4/9) | 80.06–80.10 |
| chat_mlp_q0.9 | 88.16 | 294 / 10 | 178 | 25 (3/17) | 90.03–90.07 |
| chat_mlp_q0.95 | 86.51 | 285 / 19 | 233 | 32 (4/23) | 95.00–95.03 |
| chat_mlp_q0.99 | 66.12 | 86 / 218 | 786 | 92 (3/84) | 99.00 |

Generative fixed-ε and moe_out arms (commit `6ef774f2e1`, same setup, all
samples validated by `chat_eval.py summarize`). "Forced" counts capped responses
plus finished responses without an answer line; "Total zero %" is the mean over
layers with the per-layer range. A changed count of 304 (0/282) means every
answer changed and all 282 baseline-correct answers broke.

| Arm | Accuracy % | Natural / forced | Unparsed | Mean reasoning tokens | Changed (fixed/broken) | Total zero %, mean (range) |
|---|---:|---:|---:|---:|---:|---:|
| chat_mlp_eps0.01 | 93.09 | 298 / 6 | 0 | 202 | 8 (4/3) | 16.7 (4.3–72.7) |
| chat_mlp_eps0.1 | 91.45 | 298 / 6 | 0 | 205 | 10 (2/6) | 62.5 (25.5–99.4) |
| chat_mlp_eps0.5 | 60.53 | 254 / 50 | 0 | 303 | 113 (4/102) | 95.3 (67.6–99.98) |
| chat_block_eps0.01 | 92.11 | 297 / 7 | 0 | 198 | 12 (4/6) | 7.8 (0.6–49.4) |
| chat_block_eps0.1 | 0.00 | 0 / 304 | 304 | 1024 | 304 (0/282) | 70.1 (6.2–99.5) |
| chat_block_eps0.5 | 0.00 | 0 / 304 | 303 | 1022 | 304 (0/282) | 92.4 (34.1–99.9) |
| chat_moe_out_eps0.01 | 92.76 | 296 / 8 | 0 | 211 | 9 (4/4) | 19.7 (1.4–68.3) |
| chat_moe_out_eps0.1 | 0.00 | 0 / 304 | 304 | 1021 | 304 (0/282) | 85.4 (12.5–99.99) |
| chat_moe_out_eps0.5 | 0.00 | 0 / 304 | 304 | 1024 | 304 (0/282) | 98.7 (59.4–100) |
| chat_moe_out_q0.4 | 93.42 | 299 / 5 | 0 | 204 | 16 (8/6) | 40.09 (40.09–40.10) |
| chat_moe_out_q0.5 | 91.78 | 296 / 8 | 0 | 184 | 15 (6/9) | 50.12 (50.12–50.12) |

The 0% arms loop until the cap (block ε 0.1: "1 1 1 …"; moe_out ε 0.1: repeated
option lines; moe_out ε 0.5: invisible characters) and never produce a letter,
so they score below chance.

Paired q comparison: moe_out and block at the same per-token share (token rows
of 2048 values at both sites). "Only moe_out / only block" counts questions
exactly one arm answers correctly; z = (only moe_out − only block) /
√(only moe_out + only block).

| Share | moe_out % | block % | moe_out changed (fixed/broken) | block changed (fixed/broken) | Capped moe_out / block | Only moe_out / only block | Paired z |
|---|---:|---:|---:|---:|---:|---:|---:|
| q = 0.4 | 93.42 | 90.13 | 16 (8/6) | 22 (5/13) | 5 / 14 | 13 / 3 | 2.5 |
| q = 0.5 | 91.78 | 83.88 | 15 (6/9) | 42 (5/32) | 8 / 20 | 29 / 5 | 4.1 |

Summary: `rounding-chat-sweep-summary.csv` (now with an `epsilon` column).

## Conclusions

Expert activations tolerate zeroing up to epsilon 1e-2 (about 15% of values)
with no measurable MMLU change; epsilon 1e-1 zeroes about 60% for a 0.38 pp
drop. The residual stream tolerates 1e-2 (about 7% zeroed) but collapses to
chance at 1e-1, where early layers are almost entirely zeroed.

With a uniform per-token share, expert activations tolerate 50% zeroing (-0.08
pp) in every layer, better than epsilon 1e-1 (60% overall, up to 99% in early
layers, -0.38 pp). The residual stream tolerates 25% (-0.51 pp) but loses 11.7
pp at 50%, although that is far less than epsilon 1e-1's collapse to chance.
Next steps: probe mlp above 50% (e.g. 0.7, 0.8) and block between 0.25 and 0.5,
and consider an MLSpace baseline job if job-to-local drift matters.

In the generative chat setting, mlp zeroing up to 50% and block zeroing at 25%
stay within the batching noise floor (9 changed answers between two identical
baseline runs; accuracy within 1 pp). Block at 50% is unsafe for the chat
model: -8.9 pp, 32 answers broken vs 5 fixed, 40% longer reasoning and nearly
3x as many responses hitting the reasoning cap.

Expert activations stay near the noise floor up to 80% zeroing in chat
(-1.6 pp), then degrade: -4.6 pp at 90%, -6.3 pp at 95% (reasoning 13% longer,
3x as many capped), and -26.6 pp at 99%, where 71% of responses hit the cap.

At the same per-token share, zeroing only the MoE output keeps the generative
check within 1 pp of chat_baseline (+0.66 pp at q0.4, -0.99 pp at q0.5; 15-16
changed answers against a noise floor of 9), with no longer reasoning and no more
capped responses. Block zeroing at the same shares loses 2.6 and 8.9 pp. Where
the paired arms disagree, moe_out is the correct one 13 vs 3 times at q0.4
(z = 2.5) and 29 vs 5 at q0.5 (z = 4.1).

Fixed ε in chat: ε 0.01 stays at the noise floor at every site (8-12 changed
answers, 8-20% zeroed). At ε 0.1 mlp holds (-1.3 pp at 62.5% zeroed, versus
-0.38 pp in 5-shot), while block and moe_out collapse into loops with no
parsable answer; both zero almost all values in some layers (up to 99.5% and
99.99%). At ε 0.5 only mlp still answers (60.5% at 95% zeroed). A constant ε on
the MoE output zeroes 12-99.99% per layer at ε 0.1, whereas quantile zeroing at
the same site gives every layer exactly its share.

## Changelog

- 2026-09-27: Added separate expert-activation and residual-output experiments,
  request-scoped counters, API reporting, and a local sweep launcher.
- 2026-09-28: MLP sweep (epsilon 0 to 1e-2) completed. The first block run
  (epsilon 0) was interrupted at 28% because its launching shell ended; that
  directory, `mmlu/20260928-050930`, is incomplete and excluded. Added
  sweep resume and `--detach`. Extended the MLP sweep with 1e-1 and set block to
  1e-2 and 1e-1.
- 2026-09-28: Completed mlp 1e-1 and block 1e-2/1e-1; recorded full results.
- 2026-09-28: Added per-token quantile zeroing and an MLSpace job launcher for
  {mlp, block} x {q = 0.25, 0.5}; live checks pass for mlp and block at q = 0.5.
- 2026-09-28: Added the generative chat-template STEM check (`chat_eval.py`) for
  the unzeroed baseline and the four quantile arms.
- 2026-09-28: Quantile sweep completed on MLSpace (a100.1gpu): mlp_q0.25
  lm-mpi-job-bc26f81b-4ca3-483b-b6b3-74dcea4b39c9, mlp_q0.5
  lm-mpi-job-8954aff6-caa6-4b87-8622-573832c2fb1f, block_q0.25
  lm-mpi-job-fa24bc7d-3388-46da-8969-5502420eb632, block_q0.5
  lm-mpi-job-e8dc6836-ab28-417c-a91f-3a5a951b6974; recorded results.
- 2026-09-28: Generative chat check completed (chat_baseline
  lm-mpi-job-162b1bf3-ea6e-465f-86dd-c553df5824aa, chat_baseline_repeat
  lm-mpi-job-151cd78d-5e62-4648-ba17-27e5e9556dbe, chat_mlp_q0.25
  lm-mpi-job-ff0a7928-70fc-4158-b9fa-a945280b0c03, chat_mlp_q0.5
  lm-mpi-job-0db04fd8-1316-4dfa-9f90-f5b916a890e0, chat_block_q0.25
  lm-mpi-job-bb10a42c-8804-4600-a2bd-d329ba69d723, chat_block_q0.5
  lm-mpi-job-b99d1149-fbd5-45ae-a317-89b2c99e655e); recorded results.
- 2026-09-30: Launched high-quantile generative chat arms (commit `97a3a3e818`,
  a100.1gpu): chat_mlp_q0.9 lm-mpi-job-5b1acad0-3145-4762-98f0-c7ea0e0f2962,
  chat_mlp_q0.95 lm-mpi-job-3f0ebfc0-b52d-4c90-b15e-3a971a1a085c, chat_mlp_q0.99
  lm-mpi-job-400412f9-dd28-4e8c-9dc0-bc9bebfbab30.
- 2026-09-30: High-quantile chat arms completed; recorded results.
- 2026-09-30: Added the `moe_out` zeroing site (MoE output alone, residual stream
  untouched) and generative arms: fixed ε ∈ {0.01, 0.1, 0.5} at mlp, block, and
  moe_out, and moe_out q ∈ {0.4, 0.5} (spec from a deep interview).
- 2026-09-30: moe_out live checks passed (ε = 0 parity, ε = 0.1, q = 0.5) and the
  smoke job lm-mpi-job-08967001-23ab-45f5-8d0f-b7b3138b1693 completed. Launched
  the generative ε and moe_out arms (commit `6ef774f2e1`, a100.1gpu):
  chat_mlp_eps0.01 lm-mpi-job-9c69aca7-d391-4795-8548-8fb4247c5c2e,
  chat_mlp_eps0.1 lm-mpi-job-23009a26-9474-47aa-8037-495f446f631e,
  chat_mlp_eps0.5 lm-mpi-job-0b782684-0e0a-4769-ac2e-93a71d9cb162,
  chat_block_eps0.01 lm-mpi-job-04cbd9f2-9cf4-43b8-9a7c-f1c417e2c59d,
  chat_block_eps0.1 lm-mpi-job-fdf17cbc-b7f7-49c6-875d-f3b555e69363,
  chat_block_eps0.5 lm-mpi-job-38bb021b-7295-4e1e-8624-2040d36f091e,
  chat_moe_out_eps0.01 lm-mpi-job-d3124c8c-e225-4a57-8f0d-c46d4550df51,
  chat_moe_out_eps0.1 lm-mpi-job-2b18341b-fc04-4f95-b140-cbaeffb90867,
  chat_moe_out_eps0.5 lm-mpi-job-61761a02-0ee5-4b3a-8ed8-6faaea446f3a,
  chat_moe_out_q0.4 lm-mpi-job-0f09bd35-c8bd-4d93-8162-5d3e167d570e,
  chat_moe_out_q0.5 lm-mpi-job-200ef4e8-ea26-4ef7-b212-5d63ad4c20fc.
- 2026-09-30: Generative ε and moe_out arms completed; recorded results.
- 2026-09-30: Launched moe_out high-quantile chat arms (commit `6eb25993ae`,
  a100.1gpu): chat_moe_out_q0.7 lm-mpi-job-30bf9ec4-eb14-4bd6-9c36-efa0ebb13134,
  chat_moe_out_q0.8 lm-mpi-job-f5e32351-04ca-4520-8d86-e33444a4f3ec,
  chat_moe_out_q0.9 lm-mpi-job-e4a48ec2-cc47-4e98-b3e6-13ac33a0d5f1,
  chat_moe_out_q0.95 lm-mpi-job-3d377f42-5e4b-4b11-95cc-9ced51b87ce6,
  chat_moe_out_q0.99 lm-mpi-job-7261f66a-c76a-4b81-9aee-6f2ef390b1b0.
- 2026-09-30: All evaluation is generative (`docs/adr/0004`). Removed the 5-shot
  likelihood harness path (local fixed-ε sweep, `mlp_q*`/`block_q*`/`adapt_*`
  rows, `summarize_rounding.py`, `run_jobs.py --collect`) and deleted its
  artifacts (`mmlu/`, `smoke/`, their jobs and smoke job, `rounding-sweep*`,
  `rounding-quantile-sweep*`, and the server databases and logs they used).
