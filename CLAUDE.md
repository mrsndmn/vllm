# Local experiment usage

## Usage arms

`experiments/qwen3_mmlu/run.py` uses the installed vLLM environment and cached
model; it does not import the development checkout's vLLM package. Run commands
with `/home/jovyan/.mlspace/envs/vllm_cu126/bin/python` from the repository root.

- `experiments/qwen3_mmlu/run.py start`: detached local tmux server.
- `experiments/qwen3_mmlu/run.py serve`: internal foreground server entry point.
- `experiments/qwen3_mmlu/run.py status`: session and HTTP model-list check.
- `experiments/qwen3_mmlu/run.py stop`: graceful server shutdown.
- `experiments/qwen3_mmlu/run.py restart`: stop followed by start.
- `experiments/qwen3_mmlu/run.py eval --limit 1`: MMLU smoke evaluation.
- `experiments/qwen3_mmlu/run.py eval`: full MMLU through the local HTTP API.
- `experiments/qwen3_mmlu/run.py restart --round_activations_eps 1e-4
  --round-activations-site mlp`: instrumented serving (`block` is the other site).
- `experiments/qwen3_mmlu/run.py sweep --limit 1`: all configured smoke runs.
- `experiments/qwen3_mmlu/run.py sweep`: sequential local full MMLU sweep;
  resumes by skipping configurations already recorded in its manifest.
- `experiments/qwen3_mmlu/run.py sweep --detach`: same sweep in the detached
  tmux session `vllm-qwen3-30b-a3b-sweep`, logging to `rounding-sweep.log`.
  `sweep` runs only local rows (rows without `launcher="mls"`).
- `experiments/qwen3_mmlu/run.py restart --round-activations-quantile 0.5
  --round-activations-site mlp`: per-token quantile zeroing (exclusive with eps).
- `experiments/qwen3_mmlu/run.py job --name <row> [--limit N]`: MLSpace container
  entry point; serves one `launcher="mls"` registry row as a subprocess (no tmux),
  evaluates, stops the server, and writes `jobs/<row>/done.json` last
  (`jobs-smoke/limitN/<row>/` for smoke runs).
- `/home/jovyan/.mlspace/envs/arkhip/bin/python
  experiments/qwen3_mmlu/run_jobs.py [--dry] [--filter S] [--limit N] [--force]`:
  submits one `a100.1gpu` job per pending quantile row; `--collect` writes
  `rounding-quantile-sweep.json` for `summarize_rounding.py`.
- `experiments/qwen3_mmlu/rounding_server.py`: internal wrapper CLI; forwards
  standard vLLM server arguments and selects `rounding_worker.RoundingWorker`.
- `rounding_worker.py`: GPU load, warmup, chunked prefill, decode, continuous
  batching, and finished-request cleanup; only installed vLLM 0.14.1, eager,
  unquantized Qwen3 MoE on one GPU with its Triton expert implementation.
- `rounding_api.py`: completion/chat JSON and SSE responses (including multiple
  prompts/choices), errors, `/rounding/config`, and `/rounding/stats`.
- `python -m pytest experiments/qwen3_mmlu/test_rounding.py
  --confcutdir=experiments/qwen3_mmlu -q`: local threshold/counter/API tests.
- `experiments/qwen3_mmlu/validate_live.py capture`: preserve original-server
  deterministic completion/logprob reference; `check` validates instrumented
  completion/chat, concurrent requests, multiple prompts/choices, SSE, and errors.
- `experiments/qwen3_mmlu/summarize_rounding.py --manifest <path>`: validate
  sample scores/counts and export aggregate benchmark and per-layer CSVs.

The rounding wrapper does not modify installed packages. Epsilon zero preserves
original operands; block rounding tests the FP32 sum of hidden and residual
operands before the next fused add/RMSNorm. MoE rounding tests the selected
experts' SiLU-and-multiply outputs before down projection. SQLite counters cover
executed token positions, exclude cached prefixes and warmup, and include any
recomputed positions. Each server startup gets a separate database. The wrapper
assigns unique request IDs to prevent statistics collisions.

Configuration lives in the script's `experiment_configs`; lifecycle instructions
and results live in `experiments/qwen3_mmlu/README.md` and `ROUNDING.md`.
Artifacts are outside the checkout. Fixed-epsilon rows run locally; quantile rows
run as MLSpace jobs from a per-commit snapshot of `experiments/qwen3_mmlu`, using
`/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126` (`~/.mlspace` is not mounted
in job containers). Domain terms are in `CONTEXT.md`; decisions in `docs/adr/`.
