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
  --round-activations-site mlp`: instrumented serving (other sites: `block`,
  `moe_out`; every flag, schedule, and eval path accepts all three).
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
  submits one `a100.1gpu` job per pending row (`--filter` is a name glob, e.g.
  `'chat_*'`); `--collect` writes `rounding-quantile-sweep.json` (harness rows
  only) for `summarize_rounding.py`.
- `experiments/qwen3_mmlu/run.py eval --name chat_baseline [--limit N]`: evaluate
  one registry row against the running server (it must serve the same zeroing).
  Rows with `task="mmlu_stem_chat"` run `chat_eval.py run` instead of the harness:
  chat template, 304 seeded STEM questions, greedy reasoning capped at 1024 tokens,
  then a forced `[reasoning interrupted] Answer:` letter if needed.
- `experiments/qwen3_mmlu/chat_eval.py summarize [--filter GLOB]`: validate
  finished chat jobs and write `rounding-chat-sweep-summary.csv` with answer
  changes vs `chat_baseline`.
- Generative fixed-ε and moe_out rows: `chat_{mlp,block,moe_out}_eps{0.01,0.1,0.5}`
  and `chat_moe_out_q{0.4,0.5}` (launch with `run_jobs.py --filter 'chat_*_eps*'`
  and `--filter 'chat_moe_out_q*'`).
- `experiments/qwen3_mmlu/run.py restart --round-activations-quantiles
  0,0,0.75,0,0,0,0,0 --round-activations-site block`: one quantile per equal
  consecutive layer group (8 groups of 6); exclusive with eps and quantile.
- Adaptive zeroing (`ADAPTIVE.md`): `prof_*` rows (launch with
  `run_jobs.py --filter 'prof_*'`) zero one group at a time on chat_baseline's
  questions and re-score its responses; `experiments/qwen3_mmlu/sensitivity.py`
  writes `adaptive-profile.csv`/`adaptive-schedules.json` and prints schedules to
  paste into `run.py` `ADAPTIVE_SCHEDULES`, which generates `adapt_*` (harness)
  and `chat_adapt_*` rows.
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
operands before the next fused add/RMSNorm. `moe_out` rounding tests and zeroes
the layer's MoE output (hidden) alone and never touches the residual stream
(ADR 0003). MoE rounding tests the selected experts' SiLU-and-multiply outputs
before down projection. SQLite counters cover
executed token positions, exclude cached prefixes and warmup, and include any
recomputed positions. Each server startup gets a separate database. The wrapper
assigns unique request IDs to prevent statistics collisions.

Configuration lives in the script's `experiment_configs`; lifecycle instructions
and results live in `experiments/qwen3_mmlu/README.md` and `ROUNDING.md`.
Artifacts are outside the checkout. Unnamed 5-shot fixed-epsilon rows run locally;
named (`launcher="mls"`) rows, including the generative fixed-epsilon rows, run as
MLSpace jobs from a per-commit snapshot of `experiments/qwen3_mmlu`, using
`/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126` (`~/.mlspace` is not mounted
in job containers). Domain terms are in `CONTEXT.md`; decisions in `docs/adr/`.
