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

## Conclusions

Expert activations tolerate zeroing up to epsilon 1e-2 (about 15% of values)
with no measurable MMLU change; epsilon 1e-1 zeroes about 60% for a 0.38 pp
drop. The residual stream tolerates 1e-2 (about 7% zeroed) but collapses to
chance at 1e-1, where early layers are almost entirely zeroed. Next steps:
probe mlp between 1e-2 and 1e-1 and block between 1e-2 and 1e-1, or use
per-layer thresholds.

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
