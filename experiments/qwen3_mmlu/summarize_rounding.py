"""Validate MMLU samples and export the completed rounding sweep as CSV/JSON."""

import argparse
import csv
import json
import math
from pathlib import Path

from rounding_stats import FIELDS, format_layers

from run import ARTIFACTS


def summarize_entry(entry):
    output = Path(entry["output"])
    files = list(output.glob("*/results_*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one result file: {output}")
    result = json.loads(files[0].read_text())
    questions = correct = subjects = 0
    for samples in files[0].parent.glob("samples_*.jsonl"):
        subject = samples.name.rsplit("_", 1)[0].removeprefix("samples_")
        ids = set()
        subject_correct = 0
        with samples.open() as stream:
            for line in stream:
                sample = json.loads(line)
                scores = [float(choice[0][0]) for choice in sample["resps"]]
                assert len(scores) == 4 and all(math.isfinite(x) for x in scores)
                prediction = max(range(4), key=scores.__getitem__)
                hit = int(prediction == sample["doc"]["answer"])
                assert hit == float(sample["acc"])
                assert sample["doc_id"] not in ids
                ids.add(sample["doc_id"])
                subject_correct += hit
        assert len(ids) == result["n-samples"][subject]["effective"]
        correct += subject_correct
        questions += len(ids)
        subjects += 1
    assert subjects == 57
    accuracy = result["results"]["mmlu"]["acc,none"]
    assert abs(accuracy - correct / questions) < 1e-12
    before = json.loads((output / "rounding-before.json").read_text())
    after = json.loads((output / "rounding-after.json").read_text())
    old = before["layers"] or [dict.fromkeys(FIELDS, 0) for _ in after["layers"]]
    counts = [
        [new[key] - prev[key] for key in FIELDS]
        for new, prev in zip(after["layers"], old)
    ]
    assert len(counts) == 48 and all(x >= 0 for row in counts for x in row)
    layers = format_layers(counts)
    totals = [sum(row[i] for row in counts) for i in range(len(FIELDS))]
    row = {
        "experiment": entry.get("name") or f"{entry['site']}_eps_{entry['epsilon']:g}",
        "site": entry["site"],
        "epsilon": entry["epsilon"],
        "quantile": entry.get("quantile", 0.0),
        "mmlu_percent": 100 * accuracy,
        "correct": correct,
        "questions": questions,
    }
    for name in ("humanities", "social_sciences", "stem", "other"):
        row[f"mmlu_{name}_percent"] = (
            100 * result["results"][f"mmlu_{name}"]["acc,none"]
        )
    row["newly_zeroed_float_percent"] = 100 * totals[1] / totals[0]
    row["affected_token_layer_percent"] = 100 * totals[4] / totals[3]
    row["executed_tokens_per_layer"] = counts[0][3]
    row["initial_cached_tokens_excluded"] = (
        after["initial_cached_tokens_excluded"]
        - before["initial_cached_tokens_excluded"]
    )
    row["result_file"] = str(files[0])
    return row, layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ARTIFACTS / "rounding-sweep.json"
    )
    args = parser.parse_args()
    rows, per_layer = [], []
    for entry in json.loads(args.manifest.read_text()):
        row, layers = summarize_entry(entry)
        rows.append(row)
        per_layer.extend({"experiment": row["experiment"], **layer} for layer in layers)
    stem = args.manifest.with_suffix("")
    for suffix, data in (("-summary", rows), ("-layers", per_layer)):
        path = stem.with_name(stem.name + suffix).with_suffix(".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    stem.with_name(stem.name + "-summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
