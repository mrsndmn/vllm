# Research paper

This directory follows `../compression_horizon`'s paper workflow: LaTeX sources,
separate generated tables, `latexmk` output under `build/`, and lint before build.
The manuscript reports baseline workload statistics and completed joint-zeroing
quality results. `sections/joint-zeroing.tex` discusses all thirty full evaluations;
`tables/manual/joint_zeroing.tex` transcribes their validated summary. Result
provenance and single-run limitations are documented in that section.

- Edit prose in `article.tex`, `sections/`, and `appendix.tex`.
- Edit generated table logic in `../scripts/paper/token_lengths.py`, then run
  `make tables`. Never edit generated `.tex` tables by hand.
- Keep hand-authored tables in `tables/manual/`, figures in `figures/`, and
  optional local styles in `styles/`. Reference each attachment from the sources.
- Keep experiment settings in `experiments/qwen3_mmlu/run.py`.

From the repository root:

```sh
make tables       # completed baseline and repeat arms, offline tokenizer
make paper-check  # missing references, raw tables, NaNs, unused attachments
make paper        # lint, then PDF at paper/build/article.pdf
make paper-clean  # remove LaTeX build output
```

`PYTHON` and `LATEXMK` can be overridden for another environment. The default
Python is this project's configured environment; TinyTeX supplies `latexmk` here.
PDF builds use the checked-in tables and do not require access to eval artifacts.
Table regeneration needs the external artifacts and cached tokenizer configured
in the source registry; it never starts a server or submits jobs.

The generator also saves CSV, JSON, and Markdown under the external experiment
artifact directory's `paper-token-lengths/`. To select other generative arms:

```sh
python scripts/paper/token_lengths.py --filter 'gsm8k_*' \
  --tables /tmp/gsm8k-paper-tables --output /tmp/gsm8k-token-lengths
```

Columns are benchmarks and rows are experiment arms. Missing completed runs are
listed as pending in JSON; absent cells in a partial matrix show `--`.
Output is retokenized saved model-written text, including forced continuations,
excluding injected suffixes and invisible stop tokens. Input is only the initial
saved prompt. These counts are not total API usage across requests. Provenance
and checksums are checked in alongside the generated tables.
