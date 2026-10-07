# Embedding and unembedding weight analysis

## Hypothesis

Input and output token vectors may have different marginal distributions and
channel concentration despite sharing the same hidden coordinates.

## Setup

`embedding_viz.py:analyze` reads checkpoint rows without inference and writes
an offline interactive report. The experiment design and lexical selection
are defined in the script's `experiment_configs` and `select_tokens`.

## Results

The output weights have approximately 31% greater standard deviation. Both
pooled distributions are close to centered, with embedding showing heavier
tails (excess kurtosis 0.189 versus 0.023). Ten highest-energy channels account
for about 1% of either matrix's sampled energy. Median cosine similarity of
paired input/output token vectors is 0.055.

## Conclusions

The sample has modest channel concentration and weak direct alignment between
input and output vectors. The lexical filter and uniform vocabulary sampling
do not represent corpus frequencies. These are checkpoint descriptions, not
model quality measurements. Compare with corpus-weighted tokens or other token
families before generalizing to serving behavior.

## Changelog

- 2026-10-07: Added matched token analysis, channel statistics, offline HTML,
  exact selected weights, CSV exports and source provenance.
