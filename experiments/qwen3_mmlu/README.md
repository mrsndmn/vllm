# Local Qwen3 server and MMLU

## Hypothesis

The cached model can serve requests on the local GPU while supporting a full
MMLU evaluation through its HTTP API.

## Setup

`run.py:serve` launches inference; `run.py:evaluate` runs the generative check
(`chat_eval.py`). All evaluation is generative (`docs/adr/0004`).

For the experimental activation-zeroing flag, response statistics, and the
zeroing arms, see [ROUNDING.md](ROUNDING.md).

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
Evaluation is generative: `run.py eval` runs `chat_eval.py` against the running
server with the model's chat template on 304 seeded MMLU STEM test questions
(16 per subject), greedy, with reasoning capped at 1024 tokens and the answer
letter parsed from the text (protocol in [ROUNDING.md](ROUNDING.md)).

```bash
# Small end-to-end check (1 question per subject), saved under chat-smoke/:
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py eval --limit 1
# Full generative check (foreground; leave the terminal/session open), under chat/:
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/run.py eval
```

Dependencies: the existing vLLM environment, tmux, and `datasets==3.6.0`.
Evaluation uses the prepared local `cais/mmlu` cache without Hub requests; it must
be retained for future runs. The server stays running after evaluation finishes.

## Hidden-state viewer

`hidden_viz.py` is local analysis, not an evaluation or registry row. It feeds three
chat_baseline reference responses back through the HF model (one forward pass
each, no new generation). For each token it computes the nine token stats (signed
min/max/mean/median, |x| q10–q90) at five sites: `block` (all 49 hidden states),
and per layer `residual` (r, before the MoE add), `moe_out`, `experts` (each
selected expert's down_proj output before its routing weight), and `mlp` (the
experts' SiLU·up before down_proj). It writes one HTML heatmap with the data
embedded:

```bash
/home/jovyan/.mlspace/envs/vllm_cu126/bin/python experiments/qwen3_mmlu/hidden_viz.py
```

It needs the local GPU free of the vLLM server (~62 GB for bf16 weights). The
HTML controls pick the sample, the site, the stat, and the states as a range expression
(default `0,8,16,24,32,40,48`). To the right of each shown state is the distribution
of all its values over the sample's tokens, as |x| (log axis) or signed x (symlog).

## Results

Historical (5-shot letter-likelihood harness, removed 2026-09-30 with its
artifacts; see `docs/adr/0004`). Completed on 2026-09-27: **81.9826% MMLU
accuracy**, with **11,512 correct out of 14,042 questions** across all 57
subjects.

| Overall | Humanities | Social sciences | STEM | Other |
|---:|---:|---:|---:|---:|
| 81.98% | 75.11% | 90.06% | 82.84% | 83.52% |

Verified all saved sample counts, unique question IDs within each subject,
finite choice scores, predicted answers, and aggregate accuracy. The full run
finished with exit code zero and no errors or context-truncation warnings.
Server start/status/stop/restart, chat generation, and the smoke evaluation were
also exercised. Ruff checks and independent implementation review passed.

## Conclusions

Local serving and full MMLU evaluation succeeded on the single GPU. The server
was checked after evaluation and remains running in its detached tmux session.

## Changelog

- 2026-09-27: Added local serving lifecycle commands and MMLU evaluation.
- 2026-09-27: Completed and validated full MMLU; recorded results and confirmed
  the server remains available.
- 2026-09-30: Evaluation is generative only; `run.py eval` runs the generative
  check, and local serving uses the check's `max_num_seqs` of 16 (was 8).
  Removed the likelihood harness path and its artifacts (`mmlu/`, `smoke/`,
  `mmlu-summary.*`, harness logs).
- 2026-09-30: Added the hidden-state viewer (`hidden_viz.py`).
- 2026-09-30: Viewer shows per-state value distributions (|x| or signed).
- 2026-09-30: Viewer covers five sites (block, residual, moe_out, experts, mlp).
