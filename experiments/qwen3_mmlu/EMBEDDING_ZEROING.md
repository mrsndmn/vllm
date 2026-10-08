# Per-token embedding weight zeroing

## Hypothesis

Small input/output token-vector weights may be removed with limited degradation
in generative benchmark accuracy. Removing both projections may affect results
differently from removing input weights alone.

## Setup

`embedding_zeroing.py:apply` runs after checkpoint loading in
`rounding_worker.py:RoundingWorker.load_model`, before warmup and inference.
The experiment registry is `run.py:experiment_configs`; baseline rows define
the benchmark protocols. `run_jobs.py` stages committed source and submits jobs.

## Results

All 60 evaluations completed. Finished configurations, question sets, stored
metrics, baseline prompts, and realized pruning reports passed validation.
The loaded input/output matrices were untied in every run.

Input-only zeroing at the highest tested fraction preserved MMLU STEM and GSM8K
scores (93.09% and 95.30%, versus 92.76% and 95.15% baseline), while IFEval,
HumanEval+, and MBPP+ fell by 3.33, 4.27, and 3.17 percentage points.
The input matrix retained 44.67% of its sum-of-squares energy.

Zeroing both matrices at the middle tested fraction reduced MATH500 from 89.20%
to 70.80%, IFEval from 83.55% to 51.76%, GPQA from 44.95% to 18.18%, and
RULER 32k from 94.13% to 80.27%. At the highest fraction, MMLU STEM scored
2.63%, GSM8K 21.15%, and both code benchmarks 0%. The output matrix retained
43.88% of its energy at that fraction.

Detailed summaries are stored under the configured artifact root in
`benchmarks-summary.json`, `rounding-chat-sweep-summary.json`, and
`rounding-gsm8k-sweep-summary.json`; each job retains its weight report.

## Conclusions

Input token vectors tolerate substantially more magnitude sparsity than joint
input/output pruning. MMLU and GSM8K alone miss degradation visible in instruction
following, code, and other benchmarks. Joint pruning already has large losses
at the middle fraction, despite retaining about 78% energy in each matrix.
Small changes at the lower fractions require more repetitions: the existing
baseline repeat varies by up to 2.44 percentage points across these benchmarks.
Next, isolate output-only pruning to measure its effect directly; these runs
compare input-only against joint pruning and do not establish output-only scores.

## Changelog

- 2026-10-08: Added per-token magnitude zeroing and the full evaluation registry.
- 2026-10-08: Validated all completed evaluations and recorded benchmark findings.
- 2026-10-08: Extended the input-only sparsity sweep; additional results pending.
