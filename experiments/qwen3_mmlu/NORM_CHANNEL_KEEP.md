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
the original question sets and baselines. All correction jobs completed.

RMS restoration alone scored 64/76 on MMLU STEM (84.21%) and 36/64 on GSM8K
(56.25%). Preserving the router's full current-layer input scored 68/76 (89.47%)
and 49/64 (76.56%). The combined correction scored 66/76 (86.84%) and 48/64
(75.00%). All correction answers were parsable. Reasoning-cap counts were 13/76
and 17/64 for RMS restoration, 3/76 and 2/64 for routing preservation, and 5/76
and 5/64 for the combined correction.

Validated identical question IDs, prompts, choices, answers and non-intervention
settings against the original baselines. Recomputed correctness and aggregate
counts from saved samples; runtime reports match the correction flags, mask hash,
norm sites and preserved residual. Comparisons and complete job provenance are
saved in `run/normkeep-corrections/results.json` and
`run/normkeep-corrections/results.csv`.

## Conclusions

The initial mask and two stronger follow-up masks destroy useful generation on
these preliminary subsets. The mildest follow-up preserves answer formatting but
loses 7.89 percentage points on MMLU STEM and 37.50 on GSM8K. Consistently large
norm scales alone do not justify discarding all other normalized channels.
Preserving the full current-layer router input recovers five MMLU and twelve GSM8K
answers in aggregate relative to plain masking, and substantially reduces
reasoning-cap events. Routing disruption contributes to the degradation on these
subsets. RMS restoration alone shows no consistent benefit; combining it with
routing preservation performs slightly worse than preserving routing alone.
The router-only correction still falls below the original GSM8K baseline by
18.75 percentage points, so masked expert inputs remain a material limitation.
These small preliminary subsets cannot establish the ranking on the full tasks.
A useful next comparison would preserve routing while reducing layer scope or
softening the expert-input mask, before considering a larger evaluation.

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
- 2026-10-14: Independent adversarial review approved all correction arms; focused
  tests and installed-vLLM CUDA checks passed. Submitted the correction evaluations
  and registered durable observation. Job provenance is saved in
  `run/normkeep-corrections/jobs.json`.
- 2026-10-14: Validated all completed correction evaluations and recorded paired
  results. Routing preservation improves accuracy and reasoning termination;
  weighted-RMS restoration adds no consistent benefit on these subsets.
