# Local Qwen3 server and MMLU

## Hypothesis

The cached model can serve requests on the local GPU while supporting a full
MMLU evaluation through its HTTP API.

## Setup

`run.py:serve` launches inference; `run.py:evaluate` runs the evaluation.

The OpenAI-compatible API is at **http://127.0.0.1:8000/v1**, with model name
`Qwen/Qwen3-30B-A3B-Instruct-2507`. It binds only to this machine's loopback
interface. Forward port 8000 over SSH to access it from another machine.

Run these commands from the repository root. The explicit executable uses the
configured vLLM environment even when the shell PATH points elsewhere.

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py start
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py status
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py stop
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py restart
```

Startup loads the weights and can take several minutes. `status` succeeds once
the API is ready. `start` is idempotent while its tmux session exists.
The detached tmux session `vllm-qwen3-30b-a3b` survives closing the agent or
terminal. It does not automatically restart after a machine reboot or server
crash; use `start` again. `stop` sends Ctrl-C and waits for shutdown.

Logs and evaluation artifacts are outside the repository:

```text
/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/vllm-artifacts/qwen3_mmlu/
```

Follow `server.log` there for startup errors and runtime logs.

## Evaluation

The source of truth for model and evaluation settings is `run.py`.
Evaluation uses EleutherAI lm-evaluation-harness through the running server's
completions API, with plain completion prompts and answer-choice likelihoods.
It does not apply a chat template. Full results contain all MMLU subjects,
aggregate accuracy, per-subject accuracy, and individual samples.

```bash
# Small end-to-end check, saved separately from full evaluations:
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py eval --limit 1
# Full evaluation (foreground; leave the terminal/session open):
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py eval
```

Dependencies: the existing vLLM environment, tmux, `lm_eval[api]==0.4.9.2`,
and `datasets==3.6.0`. Evaluation uses the prepared local MMLU caches without
Hub requests. These caches must be retained for future runs. The server stays
running after evaluation finishes.

## Results

Pending execution.

## Conclusions

Pending evaluation.

## Changelog

- 2026-09-27: Added local serving lifecycle commands and MMLU evaluation.
