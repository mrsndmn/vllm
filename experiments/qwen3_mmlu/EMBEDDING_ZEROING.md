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

Pending. Each evaluation records requested and realized weight-zeroing details,
including loaded-storage tying and retained energy. Existing benchmark baselines
and repeats provide comparisons.

## Conclusions

Compare generative accuracy and changes against each benchmark's baseline and
repeat. The weights are modified only in process memory; checkpoints are reused.

## Changelog

- 2026-10-08: Added per-token magnitude zeroing and the full evaluation registry.
