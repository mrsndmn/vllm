# Activation zeroing experiment

## Hypothesis

Setting small intermediate activations to zero may create sparsity without a
material change in MMLU accuracy. Expert activations and complete transformer
block outputs are separate experiments. Quantile zeroing tests whether a fixed
per-token sparsity (25% or 50% of each token row) is tolerated at either site.

## Setup

`run.py:serve`, `run.py:sweep`, and `run.py:evaluate` implement the experiment.
Model selection, threshold sweep, and benchmark settings live in
`run.py:experiment_configs`. The wrapper is pinned to the installed vLLM version;
it does not modify the shared installation or import this checkout's vLLM.

From the repository root, using the documented Python environment:

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py restart --round_activations_eps 1e-4 --round-activations-site mlp
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py status
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py stop
```

Use `block` for the other location. Both hyphenated and underscored epsilon flags
are accepted. Omit the epsilon flag on restart to restore the original server.
The existing [serving README](README.md) describes the port, persistence, and logs.
`start` remains idempotent; use `restart` to change a running server's settings.

`mlp` zeros the post-SiLU-and-multiply intermediate values of selected experts,
before their down projection. Each token's denominator includes all its selected
experts. `block` tests the combined residual stream in FP32, matching the sum
inside fused add/RMSNorm. Both residual operands are zeroed at selected positions;
all other operands retain their original bit patterns. This avoids introducing
an extra BF16 rounding operation throughout the residual stream.

### Quantile zeroing (MLSpace jobs)

`--round-activations-quantile q` (q in [0, 1), exclusive with a nonzero
epsilon) replaces the constant with a per-token-row cutoff: the row's
floor(q * width)-th smallest magnitude (width 6144 for `mlp`, 2048 for `block`).
Nonzero finite values at or below it are zeroed, ties included, so each row ends
with at least a q share of zeros; values already zero count toward q. The rule is
independent of batching. See `docs/adr/0001-per-token-quantile-zeroing.md`.

Registry rows with `launcher="mls"` run as MLSpace jobs, one `a100.1gpu` job per
row. `run_jobs.py` needs the `mls` client environment; each job runs
`run.py job --name <row>` from a per-commit snapshot of this directory, serves
that row inside the container, evaluates MMLU with the same settings as local
runs, and writes `jobs/<row>/done.json` last. Job containers use the environment
path `/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126` because `~/.mlspace` is
not mounted there (see `docs/adr/0002-single-registry-and-subdir-staging-for-mls-jobs.md`).

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py restart --round-activations-quantile 0.5 --round-activations-site mlp
M=/home/jovyan/.mlspace/envs/arkhip/bin/python
$M experiments/qwen3_mmlu/run_jobs.py --dry                  # print payloads only
$M experiments/qwen3_mmlu/run_jobs.py --filter mlp_q0.5 --limit 1   # smoke job
$M experiments/qwen3_mmlu/run_jobs.py                        # submit pending rows
$M experiments/qwen3_mmlu/run_jobs.py --collect              # manifest of finished rows
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/summarize_rounding.py --manifest <artifacts>/rounding-quantile-sweep.json
```

The launcher refuses uncommitted tracked changes in this directory, skips rows
with a done marker (unless `--force`) or an identical queued/running job, and
stages `experiments/qwen3_mmlu` once per commit under `<artifacts>/staging/`.
Smoke runs (`--limit N`) use `jobs-smoke/limitN/` instead of `jobs/`.
Summaries report both the newly zeroed share and the total zero share (newly
zeroed plus already zero); quantile zeroing guarantees the latter is at least q.

### Generative chat check (MLSpace jobs)

Rows with `task="mmlu_stem_chat"` test whether zeroing is safe for the deployed
chat model. `chat_eval.py` samples 16 test questions per MMLU STEM subject (19
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
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py sweep --limit 1
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py sweep
# Same sweep in the detached tmux session vllm-qwen3-30b-a3b-sweep:
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py sweep --detach
```

The sweep restarts the local server between experiments and leaves the final
configuration running. Each successful evaluation records its output directory
in a manifest, plus active server configuration and statistics before/after.
Rerunning resumes: configurations whose stored `experiment.json` is already in
the manifest are skipped; a manifest entry that matches no configuration (or a
different `--limit`) stops the sweep. `--detach` appends output to
`rounding-sweep.log` (or `rounding-smoke.log`) in the artifact root; the tmux
session ends when the sweep finishes or fails. Evaluation uses the same
standard likelihood-scoring harness as the original baseline. No MLS jobs run.
Before/after benchmark counter snapshots cover all server traffic in that
interval. Run without unrelated inference when interpreting them as MMLU
sparsity; per-response counters remain isolated even under concurrent traffic.

`validate_live.py capture` preserves original-server reference completions.
`validate_live.py check` exercises the running instrumented server, including
chunked prompts, cached prefixes, concurrent requests, multiple prompts/choices,
chat, streaming, and errors. Fresh cache salts isolate epsilon-zero parity checks
from differences in cached versus uncached execution shapes.
`summarize_rounding.py` validates the saved samples and exports summary and
per-layer CSVs from the sweep manifest.

## Results

Thirteen CPU correctness/API/sweep-resume/session tests pass. Live GPU checks
pass for both locations with zero and nonzero thresholds, including exact
original completion/logprob parity at zero, chunked prefill, cached prefixes,
concurrent requests, multiple prompts/choices, chat, streaming, and error
handling. The implementation passed independent adversarial review.

Full 5-shot MMLU sweep completed on 2026-09-28 (14,042 questions per run, all
samples validated by `summarize_rounding.py`). "Zeroed" is the share of all
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
Aggregate and per-layer CSVs are `rounding-sweep-summary.csv` and
`rounding-sweep-layers.csv` in the artifact root.

Quantile zeroing sweep (MLSpace jobs, commit `0c11e87412`, full 5-shot MMLU,
all samples validated). "Total zero" is the zero share after zeroing including
already-zero values; every layer of every run lies in [q, q + 0.11 pp]. Flips are
changed answers versus the local baseline (`mlp` epsilon 0, 81.98%).

| Experiment | MMLU % | Hum. | Soc. | STEM | Other | Total zero % | Flips (fixed/broken) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp q=0.25 | 82.15 | 75.47 | 90.02 | 83.03 | 83.55 | 25.05 | 200 (89/66) |
| mlp q=0.5 | 81.90 | 75.09 | 89.60 | 83.00 | 83.46 | 50.09 | 401 (153/165) |
| block q=0.25 | 81.47 | 74.54 | 89.47 | 82.24 | 83.26 | 25.02 | 774 (276/348) |
| block q=0.5 | 70.33 | 60.98 | 80.70 | 70.63 | 73.93 | 50.02 | 3231 (521/2157) |

Summaries are `rounding-quantile-sweep-summary.csv` and
`rounding-quantile-sweep-layers.csv` in the artifact root. No MLSpace baseline
job was run; comparisons use the local baseline.

Generative chat check (MLSpace, commit `85b77482af`, 304 seeded STEM test
questions, greedy, 1024-token reasoning cap). "Changed" counts answers differing
from `chat_baseline`; `chat_baseline_repeat` gives the noise floor from batching
nondeterminism. The last column scores the same 304 questions with the 5-shot
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
