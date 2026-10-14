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

The initial paired preliminary MLSpace jobs completed. The original model scored 69/76
on MMLU STEM (90.79%) and 61/64 on GSM8K (95.31%). The channel-masked model scored
0/76 and 0/64 with the initial mask; every masked answer was unparsable and reached the reasoning cap.
Reasoning consisted only of whitespace in 75/76 MMLU and 45/64 GSM8K samples.

Verified identical question sets, prompts and generation settings within each
pair. Runtime reports match the pinned channel mask and selected norm sites.
Sample counts, correctness, accuracy and missing-answer counts were checked
against the individual saved records. Comparison artifacts are saved under
`run/normkeep-preview/results.json` and `run/normkeep-preview/results.csv`.

Follow-up masks with greater channel retention are defined in the same registry;
their preliminary evaluations reuse the original matched baselines. All follow-up
jobs have been submitted, with durable observation registered; results are pending.

## Conclusions

Simultaneous hard masking at the selected norm sites destroys useful generation
on these preliminary subsets. Consistently large norm scales alone do not justify
discarding all other normalized channels. A follow-up should reduce the layer
scope or soften the intervention before considering a larger evaluation.

## Changelog

- 2026-10-14: Implemented pinned post-normalization channel masking and paired
  preliminary evaluation configurations.
- 2026-10-14: Independent review approved the implementation after fixing preview
  exclusion from full paper tables; submitted the paired preliminary jobs.
- 2026-10-14: All jobs completed; validated paired results and runtime masks and
  recorded complete generation collapse under the intervention.
- 2026-10-14: Added larger-retention follow-up configurations with pinned masks
  and unchanged question selection and generation protocols.
- 2026-10-14: Independent review approved the follow-up configurations; submitted
  all follow-up evaluations concurrently and registered durable observation.
