# Activation zeroing experiment

## Hypothesis

Setting small intermediate activations to zero may create sparsity without a
material change in MMLU accuracy. Expert activations and complete transformer
block outputs are separate experiments.

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
```

The sweep restarts the local server between experiments and leaves the final
configuration running. Each successful evaluation records its output directory
in a manifest, plus active server configuration and statistics before/after.
An existing manifest prevents accidental reruns. Evaluation uses the same
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

Nine CPU correctness/API tests pass. Live GPU checks pass for both locations
with zero and nonzero thresholds, including exact original completion/logprob
parity at zero, chunked prefill, cached prefixes, concurrent requests, multiple
prompts/choices, chat, streaming, and error handling. The implementation passed
independent adversarial review. Full threshold-sweep results are pending.

## Conclusions

Await correctness validation and full benchmark results before judging sparsity
or accuracy effects.

## Changelog

- 2026-09-27: Added separate expert-activation and residual-output experiments,
  request-scoped counters, API reporting, and a local sweep launcher.
