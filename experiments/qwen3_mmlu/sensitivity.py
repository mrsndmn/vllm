"""Adaptive zeroing stages A/B: per-group sensitivity and budgeted schedules.

Reads finished `prof_*` chat runs (one 6-layer group zeroed at a time on the
chat_baseline questions). A group's cost at a quantile is the increase in
per-token NLL of chat_baseline's responses versus `prof_baseline`, both re-scored
inside their jobs; accuracy and answer changes versus chat_baseline are reported
alongside. For each site and budget (mean zero share over the equal groups) it
picks one profiled level per group minimizing the summed cost, assuming group
effects add up. Paste the printed schedules into run.py's ADAPTIVE_SCHEDULES.
"""

import argparse
import csv
import itertools
import json
import statistics

import chat_eval

import run


def nll(result):
    return json.loads((result[1] / "results.json").read_text())["reference"][
        "nll_per_token"
    ]


def profile(runs, reference):
    """One row per profiling run; `reference` maps questions to baseline records."""
    base = nll(runs["prof_baseline"])
    rows = []
    for name, (config, output, records) in sorted(runs.items()):
        result = json.loads((output / "results.json").read_text())
        schedule = config["round_activations_quantiles"]
        active = [index for index, value in enumerate(schedule) if value]
        group = active[0] if active else None
        size = len(chat_eval.layer_zero_shares(output)) // len(schedule)
        shares = chat_eval.layer_zero_shares(output)
        by_group = [
            statistics.mean(shares[g * size : (g + 1) * size])
            for g in range(len(schedule))
        ]
        rows.append(
            {
                "experiment": name,
                "site": config["round_activations_site"],
                "group": group,
                "layers": f"{group * size}-{(group + 1) * size - 1}" if active else "",
                "quantile": schedule[group] if active else 0.0,
                "delta_nll_per_token": result["reference"]["nll_per_token"] - base,
                "accuracy_percent": 100 * result["accuracy"],
                "changed_vs_baseline": sum(
                    records[k]["prediction"] != reference[k]["prediction"]
                    for k in records
                ),
                "broken_vs_baseline": sum(
                    reference[k]["correct"] and not records[k]["correct"]
                    for k in records
                ),
                "fixed_vs_baseline": sum(
                    records[k]["correct"] and not reference[k]["correct"]
                    for k in records
                ),
                "forced_cap": result["modes"]["forced_cap"],
                "mean_reasoning_tokens": result["reasoning_tokens"]["mean"],
                "group_zero_percent": by_group[group] if active else 0.0,
                "other_groups_max_zero_percent": max(
                    share for g, share in enumerate(by_group) if g != group
                ),
            }
        )
    return rows


def allocate(costs, levels, budget, groups, raw=None):
    """Cheapest per-group levels whose mean reaches the budget (exhaustive).

    Ties (costs within noise are clamped to zero) go to the lower unclamped
    loss, then the smaller mean, then the most even schedule, then the order.
    """
    raw = raw or costs
    best = None
    for choice in itertools.product((0.0, *levels), repeat=groups):
        total = round(sum(choice), 9)
        if total < round(budget * groups, 9):
            continue
        key = (
            round(sum(costs[(g, q)] for g, q in enumerate(choice) if q), 12),
            round(sum(raw[(g, q)] for g, q in enumerate(choice) if q), 12),
            total,
            max(choice) - min(choice),
            choice,
        )
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError(f"Budget {budget} is unreachable with levels {levels}")
    cost, _, total, _, choice = best
    return {"schedule": list(choice), "mean": total / groups, "predicted_cost": cost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    runs = chat_eval.finished_runs("prof_*")
    missing = {
        c["name"]
        for c in run.experiment_configs
        if c.get("name", "").startswith("prof_")
    } - set(runs)
    if missing:
        raise SystemExit(f"Unfinished profiling runs: {sorted(missing)}")
    reference = chat_eval.finished_runs("chat_baseline")["chat_baseline"][2]
    rows = profile(runs, reference)
    path = run.ARTIFACTS / "adaptive-profile.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    noise = [r for r in rows if r["experiment"] == "prof_baseline_repeat"]
    schedules = {}
    for site, budgets in run.ADAPTIVE_BUDGETS.items():
        # Changes within the unzeroed repeat's difference count as noise.
        floor = abs(noise[0]["delta_nll_per_token"]) if noise else 0.0
        raw = {
            (r["group"], r["quantile"]): r["delta_nll_per_token"]
            for r in rows
            if r["site"] == site and r["group"] is not None
        }
        costs = {key: max(value - floor, 0.0) for key, value in raw.items()}
        for budget in budgets:
            schedules[f"{site}_b{budget:g}"] = allocate(
                costs, run.PROFILE_LEVELS[site], budget, run.GROUPS, raw
            ) | {"site": site, "budget": budget}
    output = {
        "noise_floor": noise[0] if noise else None,
        "schedules": schedules,
    }
    (run.ARTIFACTS / "adaptive-schedules.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print("ADAPTIVE_SCHEDULES = {")
    for item in schedules.values():
        print(f'    ("{item["site"]}", {item["budget"]}): {item["schedule"]},')
    print("}")


if __name__ == "__main__":
    main()
