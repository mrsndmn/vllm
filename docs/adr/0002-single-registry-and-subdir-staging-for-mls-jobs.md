# MLSpace jobs reuse run.py's registry and stage only the experiment directory

The MLSpace launcher for the activation-zeroing experiments deliberately
deviates from the standard job scaffold: it has no separate `experiments.py`,
and staging copies only `experiments/qwen3_mmlu/` at the submitted commit, not
the whole vLLM repository. `run.py`'s `experiment_configs` is the project's single
experiments-as-code registry, which the local tmux flow also uses, so a second
registry would split the source of truth. Jobs run the installed vLLM 0.14.1
environment and never import this checkout's `vllm/` package, so staging the
305 MB repository would copy nothing the job uses.

## Consequences

Jobs must use the environment path `/workspace-SR004.nfs2/d.tarasov/envs/`,
because `/home/jovyan/.mlspace/envs` (the same environments via symlink) is not
mounted in job containers.
