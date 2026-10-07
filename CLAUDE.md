# Local experiment usage

## Evaluation policy

All further evaluation must be generative: the model writes its answer through
the chat template and the answer is parsed from the generated text
(`experiments/qwen3_mmlu/chat_eval.py`, MMLU and GSM8K). Do not add or
revive likelihood-based evaluation (lm-evaluation-harness `loglikelihood` tasks,
5-shot answer-letter or choice log-probability scoring). The harness path, its
rows, and its artifacts were removed on 2026-09-30 (`docs/adr/0004`); 5-shot
numbers in the experiment notes are history. New arms are `chat_*` registry rows
compared with `chat_baseline` and its noise floor. Log-probabilities are still
used, but not as an evaluation, by `validate_live.py` server-parity checks and by
the `prof_*` re-scoring of `chat_baseline`'s generated responses.

## Usage arms

- `watch_jobs.py --manifest FILE --state-dir DIR [--resume-config FILE]`:
  poll full MLS job IDs, persist status snapshots, and stop on the first terminal
  failure or complete success. With an Arkhip resume config, resume the bound
  conversation, retrying raised callback errors and recovering pending events;
  bot credentials arrive via environment or a private FIFO,
  never the manifest. Run in the MLS client environment, detached in tmux;
  `--once` performs one check. See `experiments/qwen3_mmlu/GSM8K.md`.
- `run.py eval --name gsm8k_baseline [--limit N]`: generative GSM8K evaluation
  against the matching local server; `gsm8k.py` parses numeric answers.
- `run_jobs.py --filter 'gsm8k_*' [--dry] [--limit N]`: GSM8K baseline/repeat
  and selected uniform/adaptive zeroing arms transferred from MMLU,
  through the shared MLS launcher. Dataset and protocol settings are in
  `run.py:experiment_configs`; GSM8K may fetch the pinned dataset into HF cache.
- `chat_eval.py summarize --task gsm8k_chat [--filter GLOB]`: validate full
  GSM8K jobs and write their own summary against `gsm8k_baseline`, separately
  from the default MMLU summary. See `experiments/qwen3_mmlu/GSM8K.md`.
- `experiments/qwen3_mmlu/compression.py {all,capture,screen,plot}`: checkpoint
  compression screening on routed corpus activations; no model accuracy eval.
  `all` captures separate calibration/held-out splits, screens selected expert
  projections and complete expert outputs, plots, and writes `done.json` last.
  Individual stages resume from source/config-validated captures.
  Configuration is in `compression.py:experiment_configs`; artifacts are under
  the configured external output. No sparse serving kernels are implemented.
- `experiments/qwen3_mmlu/run_jobs.py --analysis [--dry]`: submit the compression
  registry through existing committed-code staging and deduplication. This path
  uses the MLS client environment like the existing MMLU launcher; `--limit` is
  unavailable for analysis. Unit checks extend `test_rounding.py`.

`experiments/qwen3_mmlu/run.py` uses the installed vLLM environment and cached
model; it does not import the development checkout's vLLM package. Run commands
with `/home/jovyan/.mlspace/envs/vllm_cu126/bin/python` from the repository root.

- `experiments/qwen3_mmlu/run.py start`: detached local tmux server.
- `experiments/qwen3_mmlu/run.py serve`: internal foreground server entry point.
- `experiments/qwen3_mmlu/run.py status`: session and HTTP model-list check.
- `experiments/qwen3_mmlu/run.py stop`: graceful server shutdown.
- `experiments/qwen3_mmlu/run.py restart`: stop followed by start.
- `experiments/qwen3_mmlu/run.py eval --limit 1`: generative-check smoke run
  (1 question per STEM subject) against the running local server.
- `experiments/qwen3_mmlu/run.py eval`: full 304-question generative check
  through the local HTTP API; outputs under `<artifacts>/chat[-smoke]/<time>/`.
- `experiments/qwen3_mmlu/run.py restart --round_activations_eps 1e-4
  --round-activations-site mlp`: instrumented serving (other sites: `block`,
  `moe_out`; every flag, schedule, and eval path accepts all three).
- `experiments/qwen3_mmlu/run.py restart --round-activations-quantile 0.5
  --round-activations-site mlp`: per-token quantile zeroing (exclusive with eps).
- `experiments/qwen3_mmlu/run.py job --name <row> [--limit N]`: MLSpace container
  entry point; serves one `launcher="mls"` registry row as a subprocess (no tmux),
  evaluates, stops the server, and writes `jobs/<row>/done.json` last
  (`jobs-smoke/limitN/<row>/` for smoke runs).
- `/home/jovyan/.mlspace/envs/arkhip/bin/python
  experiments/qwen3_mmlu/run_jobs.py [--dry] [--filter S] [--limit N] [--force]`:
  submits one `a100.1gpu` job per pending row (`--filter` is a name glob, e.g.
  `'chat_*'`).
- `experiments/qwen3_mmlu/run.py eval --name chat_baseline [--limit N]`: evaluate
  one registry row against the running server (it must serve the same zeroing).
  Every MMLU row and the default config run `chat_eval.py run`: chat template, 304
  seeded STEM questions, greedy reasoning capped at 1024 tokens, then a forced
  `[reasoning interrupted] Answer:` letter if needed.
- `experiments/qwen3_mmlu/chat_eval.py summarize [--filter GLOB]`: validate
  finished chat jobs and write `rounding-chat-sweep-summary.csv` with answer
  changes vs `chat_baseline`.
- Generative fixed-ε and moe_out rows: `chat_{mlp,block,moe_out}_eps{0.01,0.1,0.5}`
  and `chat_moe_out_q{0.4,0.5,0.7,0.8,0.9,0.95,0.99}` (launch with
  `run_jobs.py --filter 'chat_*_eps*'` and `--filter 'chat_moe_out_q*'`).
- `experiments/qwen3_mmlu/run.py restart --round-activations-quantiles
  0,0,0.75,0,0,0,0,0 --round-activations-site block`: one quantile per equal
  consecutive layer group (8 groups of 6); exclusive with eps and quantile.
- Adaptive zeroing (`ADAPTIVE.md`): `prof_*` rows (launch with
  `run_jobs.py --filter 'prof_*'`) zero one group at a time on chat_baseline's
  questions and re-score its responses; `experiments/qwen3_mmlu/sensitivity.py`
  writes `adaptive-profile.csv`/`adaptive-schedules.json` and prints schedules to
  paste into `run.py` `ADAPTIVE_SCHEDULES`, which generates `chat_adapt_*` rows.
- `experiments/qwen3_mmlu/hidden_viz.py [--device cuda:0] [--output DIR]`: local
  analysis outside the registry (no server; needs ~62 GB free on the local GPU).
  Teacher-forces three chat_baseline reference responses (abstract_algebra #11,
  anatomy #14, astronomy #6) through the HF model (transformers, bf16). For every
  token it computes token stats at five sites: `block` (the 49 hidden states:
  embedding output plus the pre-norm residual after each layer), and per layer
  `residual` (r), `moe_out`, `experts` (unweighted expert outputs), and `mlp`
  (experts' SiLU-and-multiply outputs, recomputed from each MoE block's input and
  checked to reproduce its output exactly). It writes
  `<artifacts>/hidden-viz/<time>/hidden-states.html` (heatmap from the
  `hidden_viz.html` template, data embedded, Plotly from a CDN; ~13 MB) plus
  `stats.npz`. The heatmap has a site selector and a per-state value distribution
  on its right (all tokens): signed x on a linear axis over ±the row's 99.5th
  percentile of |x| (default), or |x| on a log axis.
- `experiments/qwen3_mmlu/rounding_server.py`: internal wrapper CLI; forwards
  standard vLLM server arguments and selects `rounding_worker.RoundingWorker`.
- `experiments/qwen3_mmlu/weight_viz.py [--device cuda:0] [--output DIR]
  [--name qwen3_weights|llama32_1b_weights|pythia160m_weights]
  [--render-only]`: checkpoint-only
  weight analysis, no inference/server or MLS launch. Supports indexed shards
  and a single safetensors file. Reads the configured safetensors, caches characteristics
  and distributions under the analysis output, and writes offline `weights.html`.
  `--render-only` requires all shard caches. The HTML supports layer/family/expert
  selection, metric maps, pooled expert distributions and layer comparison.
  Configuration is in the script's `experiment_configs`; approximate quantiles
  and stable rank are labelled. The actual HTML byte limit is enforced before
  replacing the output. Unit and Node helper checks extend `test_rounding.py`.
  Dense models hide expert controls; the title and layer ranges reflect the model.
  GPT-NeoX layer weights and biases are separate families; global embeddings and
  final norm also support distribution comparison.
  Pythia attention-mask and rotary-frequency buffers are excluded via its registry
  row and listed in report provenance; projection and norm biases remain included.
- `rounding_worker.py`: GPU load, warmup, chunked prefill, decode, continuous
  batching, and finished-request cleanup; only installed vLLM 0.14.1, eager,
  unquantized Qwen3 MoE on one GPU with its Triton expert implementation.
- `rounding_api.py`: completion/chat JSON and SSE responses (including multiple
  prompts/choices), errors, `/rounding/config`, and `/rounding/stats`.
- `python -m pytest experiments/qwen3_mmlu/test_rounding.py
  --confcutdir=experiments/qwen3_mmlu -q`: local threshold/counter/API and
  hidden-state viewer tests (the viewer's JS range parser runs under `node`, and
  that test is skipped without it).
- `experiments/qwen3_mmlu/validate_live.py capture`: preserve original-server
  deterministic completion/logprob reference; `check` validates instrumented
  completion/chat, concurrent requests, multiple prompts/choices, SSE, and errors.

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
Artifacts are outside the checkout. Every registry row is named
(`launcher="mls"`) and runs as an MLSpace job from a per-commit snapshot of
`experiments/qwen3_mmlu`, using
`/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126` (`~/.mlspace` is not mounted
in job containers). Domain terms are in `CONTEXT.md`; decisions in `docs/adr/`.
