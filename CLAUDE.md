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
- `experiments/qwen3_mmlu/run.py sweep`: sequential local full MMLU sweep.
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
and results live in `experiments/qwen3_mmlu/README.md`. Artifacts are outside the
checkout. No MLS jobs are used for this experiment.
