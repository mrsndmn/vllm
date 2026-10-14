# Keep common large norm channels

## Hypothesis

Channels with consistently large normalization scales may retain enough
information for useful generation when the other normalized channels are zeroed.

## Setup

`post_norm_channels.apply` implements the intervention; `run.py:job` runs the
paired preliminary evaluations. Experiment definitions and mask selection are
in `run.py:experiment_configs` (`normkeep_preview_*`). The same questions and
generation protocol are used for each baseline/intervention pair.

## Results

Pending preliminary evaluations.

## Conclusions

Compare paired accuracy, unparsable answers and reasoning-cap frequency before
deciding whether a larger evaluation is warranted.

## Changelog

- 2026-10-14: Implemented pinned post-normalization channel masking and paired
  preliminary evaluation configurations.
