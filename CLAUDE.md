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

Configuration lives in the script's `experiment_configs`; lifecycle instructions
and results live in `experiments/qwen3_mmlu/README.md`. Artifacts are outside the
checkout. No MLS jobs are used for this experiment.
