# Evaluation is generative only

From 2026-09-30 every evaluation of the activation-zeroing experiments is
generative: the chat model writes its reasoning and an answer through its chat
template (`chat_eval.py`, the generative check), and the answer is parsed from the
text. The 5-shot lm-evaluation-harness path, which scored answer letters by
log-likelihood, was removed together with its registry rows, local sweep, summary
script, and artifacts (user decision). The generative check exercises the model
the way it is deployed; letter likelihoods cannot show generation failures such as
the reasoning loops without an answer seen at block and moe_out ε 0.1.

## Consequences

- `run.py eval`, `run.py job`, and every registry row run `chat_eval.py`;
  `evaluate` rejects any other task.
- Log-probabilities remain in two non-evaluation places: `validate_live.py`
  compares completion log-probabilities for server parity, and profiling rows
  (`chat_reference`) re-score a reference run's generated responses as a
  sensitivity cost for adaptive allocation.
- Recorded 5-shot numbers in `ROUNDING.md`, `ADAPTIVE.md`, and the serving README
  are history; their raw artifacts no longer exist.
