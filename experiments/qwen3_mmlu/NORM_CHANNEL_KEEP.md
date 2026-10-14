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
jobs completed. `keep1980` scored 63/76 on MMLU STEM (82.89%) and 37/64 on GSM8K
(57.81%), with no unparsable answers. Against the baseline it broke eight and
fixed two MMLU answers, and broke 25 and fixed one GSM8K answer. Reasoning-cap
counts increased to 14/76 and 15/64 respectively.

Both `keep1828` and `keep1539` scored 0/76 and 0/64. Every answer in both arms was
unparsable and reached the reasoning cap. Runtime mask hashes, question IDs,
prompts and non-intervention settings were validated against the original
baselines. Follow-up comparison artifacts, including the original arms, are saved
under `run/normkeep-expanded/results.json` and `run/normkeep-expanded/results.csv`.

Correction comparisons test preserving each token's original weighted norm-output
energy and computing routing on the full current-layer norm output before masking
expert inputs. The combined arm applies both corrections. These evaluations reuse
the original question sets and baselines; correction results are pending.

## Conclusions

The initial mask and two stronger follow-up masks destroy useful generation on
these preliminary subsets. The mildest follow-up preserves answer formatting but
loses 7.89 percentage points on MMLU STEM and 37.50 on GSM8K. Consistently large
norm scales alone do not justify discarding all other normalized channels.
The same shared mask across the selected norm sites is sensitive even to modest
channel removal; a follow-up should reduce layer scope or use a softer
intervention before considering a larger evaluation.

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
- 2026-10-14: Validated all completed follow-ups against the original baselines;
  recorded substantial accuracy loss in the mildest mask and generation collapse
  in the other follow-up masks.
- 2026-10-14: Added weighted-RMS restoration, external-routing preservation and
  combined correction arms, using the same mask and preview evaluation protocols.
