# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plot benchmark scores, coloring changes against each benchmark's baseline."""

import argparse
import fnmatch
import json
import warnings
from pathlib import Path

import numpy as np

import run

TASK_LABELS = {
    "mmlu_stem_chat": "MMLU STEM",
    "gsm8k_chat": "GSM8K",
    "math500": "MATH500",
    "ifeval": "IFEval",
    "gpqa": "GPQA",
    "humaneval_plus": "HumanEval+",
    "mbpp_plus": "MBPP+",
    "ruler_8k": "RULER 8k",
    "ruler_16k": "RULER 16k",
    "ruler_32k": "RULER 32k",
    "mmlu_pro": "MMLU-Pro",
}


def arm(name, task):
    prefix = {
        "mmlu_stem_chat": "chat_",
        "gsm8k_chat": "gsm8k_",
    }.get(task, "suite_" + task + "_")
    return name.removeprefix(prefix)


def score_matrix(rows, expected=()):
    """Return labels, scores and baseline deltas; missing cells remain NaN."""
    tasks = list(
        dict.fromkeys([r["task"] for r in rows] + [c["task"] for c in expected])
    )
    tasks.sort(key=lambda t: list(TASK_LABELS).index(t) if t in TASK_LABELS else 99)
    arms = list(
        dict.fromkeys(
            [arm(r["experiment"], r["task"]) for r in rows]
            + [arm(c["name"], c["task"]) for c in expected]
        )
    )

    def order(a):
        if a in ("baseline", "baseline_repeat"):
            return (0 if a == "baseline" else 1, 0)
        if a.startswith("weight_"):
            scope, q = a.removeprefix("weight_").rsplit("_q", 1)
            return (2 if scope == "embedding" else 3, float(q))
        return (4, a)

    arms.sort(key=order)
    scores = np.full((len(arms), len(tasks)), np.nan)
    for row in rows:
        value = (
            row["accuracy_percent"]
            if "accuracy_percent" in row
            else 100 * row["accuracy"]
        )
        scores[
            arms.index(arm(row["experiment"], row["task"])), tasks.index(row["task"])
        ] = value
    baseline = (
        scores[arms.index("baseline")]
        if "baseline" in arms
        else np.full(len(tasks), np.nan)
    )
    return arms, tasks, scores, scores - baseline


def label(name):
    if name.startswith("weight_"):
        scope, q = name.removeprefix("weight_").rsplit("_q", 1)
        scope = "Embedding" if scope == "embedding" else "Embedding + output"
        return f"{scope} {100 * float(q):g}%"
    return name.replace("_", " ").capitalize()


def save(rows, path, expected=()):
    """Save a headless PNG; scores are text, baseline differences set the color."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    arms, tasks, scores, delta = score_matrix(rows, expected)
    if not arms or not tasks:
        raise ValueError("No benchmark scores to plot")
    finite = delta[np.isfinite(delta)]
    low = min(-5.0, float(finite.min())) if finite.size else -5.0
    high = max(5.0, float(finite.max())) if finite.size else 5.0
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad="#dedede")
    fig, ax = plt.subplots(
        figsize=(max(8, 1.35 * len(tasks) + 3), 2 + 0.52 * len(arms))
    )
    try:
        im = ax.imshow(
            np.ma.masked_invalid(delta),
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=low, vcenter=0, vmax=high),
            aspect="auto",
        )
        ax.set_xticks(
            range(len(tasks)),
            [TASK_LABELS.get(t, t) for t in tasks],
            rotation=20,
            ha="right",
        )
        ax.set_yticks(range(len(arms)), [label(a) for a in arms])
        for i in range(len(arms)):
            for j in range(len(tasks)):
                text = f"{scores[i, j]:.2f}" if np.isfinite(scores[i, j]) else "pending"
                color = (
                    "white"
                    if np.isfinite(delta[i, j]) and delta[i, j] < 0.6 * low
                    else "black"
                )
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
        ax.set_title(
            "Benchmark scores (%)\nColor: change vs baseline (pp); "
            "gray: missing score or baseline",
            pad=15,
        )
        fig.colorbar(im, ax=ax, label="Change vs baseline (pp)", shrink=0.85)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170)
    finally:
        plt.close(fig)
    return path


def combined(extra_rows=(), pattern="*weight_*"):
    """Combine canonical summaries; include baseline and unfinished selected arms."""
    keyed = {}
    for name in (
        "rounding-chat-sweep-summary.json",
        "rounding-gsm8k-sweep-summary.json",
        "benchmarks-summary.json",
    ):
        path = run.ARTIFACTS / name
        if path.exists():
            try:
                saved = json.loads(path.read_text())
            except json.JSONDecodeError:
                warnings.warn(f"Skipping incomplete summary: {path}", stacklevel=2)
                continue
            for row in saved:
                keyed[row["experiment"]] = row
    keyed.update((r["experiment"], r) for r in extra_rows)
    configs = [
        c for c in run.experiment_configs if fnmatch.fnmatchcase(c["name"], pattern)
    ]
    if not configs:
        return None
    tasks = {c["task"] for c in configs}
    configs += [
        c
        for c in run.experiment_configs
        if c["task"] in tasks
        and arm(c["name"], c["task"]) in ("baseline", "baseline_repeat")
    ]
    names = {c["name"] for c in configs}
    rows = [r for name, r in keyed.items() if name in names]
    return save(rows, run.ARTIFACTS / "benchmarks-heatmap.png", configs)


def summary(rows, stem, expected=None):
    """Write a PNG beside each summary and refresh the combined weight heatmap."""
    tasks = {r["task"] for r in rows}
    arms = {arm(r["experiment"], r["task"]) for r in rows}
    if expected is None:
        expected = [
            c
            for c in run.experiment_configs
            if c["task"] in tasks and arm(c["name"], c["task"]) in arms
        ]
    if rows or expected:
        save(rows, Path(str(stem) + ".png"), expected)
    combined(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="*weight_*")
    args = parser.parse_args()
    path = combined(pattern=args.filter)
    if path is None:
        parser.error("No experiments match the filter")
    print(path)
