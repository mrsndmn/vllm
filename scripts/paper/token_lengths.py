# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate paper tables from completed generative evaluation samples."""

import argparse
import csv
import fnmatch
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/qwen3_mmlu"))
import run  # noqa: E402

LABELS = {
    "mmlu_stem_chat": "MMLU STEM",
    "gsm8k_chat": "GSM8K",
    "math500": "MATH500",
    "ifeval": "IFEval",
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA",
    "humaneval_plus": "HumanEval+",
    "mbpp_plus": "MBPP+",
    "ruler_8k": "RULER 8k",
    "ruler_16k": "RULER 16k",
    "ruler_32k": "RULER 32k",
}

METHOD = (
    "Input: saved initial prompt including chat template, add_special_tokens=False. "
    "Output: retokenized saved response text; for chat, reasoning and forced_text "
    "are encoded separately and summed. Injected forced_suffix and invisible stop "
    "tokens are excluded. These are text lengths, not API usage or total prefill "
    "work across forced-answer requests. Each mean weights questions equally."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_questions(config):
    if "benchmark" in config:
        return config["benchmark"]["expected_questions"]
    if config["task"] == "gsm8k_chat":
        return config["chat_expected_questions"]
    return len(config["chat_subjects"]) * config["chat_per_subject"]


def summarize(records, config, count):
    """Validate a complete sample set and count initial input and output text."""
    if len(records) != expected_questions(config):
        raise ValueError(f"Incomplete samples: {config['name']}")
    keys = [
        r["id"] if "benchmark" in config else (r["subject"], r["index"])
        for r in records
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate sample IDs: {config['name']}")
    if config["task"] == "mmlu_stem_chat":
        coverage = Counter(record["subject"] for record in records)
        expected = Counter(
            dict.fromkeys(config["chat_subjects"], config["chat_per_subject"])
        )
        if coverage != expected:
            raise ValueError(f"MMLU subject coverage differs: {config['name']}")
    inputs, outputs = [], []
    for record in records:
        inputs.append(count(record["prompt"]))
        if "input_tokens" in record and record["input_tokens"] != inputs[-1]:
            raise ValueError(f"Input token count mismatch: {config['name']}")
        if "benchmark" in config:
            outputs.append(count(record["response"]))
        else:
            forced = record["mode"] in ("forced_cap", "forced_missing")
            if forced and not {"forced_text", "forced_suffix"} <= record.keys():
                raise ValueError(f"Missing forced continuation: {config['name']}")
            outputs.append(
                count(record["reasoning"])
                + (count(record["forced_text"]) if forced else 0)
            )
    return dict(
        experiment=config["name"],
        task=config["task"],
        questions=len(records),
        mean_input_tokens=statistics.mean(inputs),
        mean_output_text_tokens=statistics.mean(outputs),
    )


def arm(row):
    name, task = row["experiment"], row["task"]
    prefix = (
        "chat_"
        if task == "mmlu_stem_chat"
        else "gsm8k_"
        if task == "gsm8k_chat"
        else f"suite_{task}_"
    )
    return name.removeprefix(prefix)


def pivot(rows, metric):
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    arms = list(dict.fromkeys(arm(r) for r in rows))
    values = {(arm(r), r["task"]): r[metric] for r in rows}
    return tasks, [
        [name, *[values.get((name, task)) for task in tasks]] for name in arms
    ]


def tex_escape(text):
    return "".join(
        {
            "\\": r"\textbackslash{}",
            "_": r"\_",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
            "$": r"\$",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }.get(char, char)
        for char in text
    )


def render(rows, metric, latex=False):
    tasks, values = pivot(rows, metric)
    header = ["Experiment", *[LABELS.get(task, task) for task in tasks]]
    formatted = [
        [
            row[0],
            *[
                "--" if v is None else (str(v) if metric == "questions" else f"{v:.1f}")
                for v in row[1:]
            ],
        ]
        for row in values
    ]
    if not latex:
        return (
            "\n".join(
                [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * len(header)) + " |",
                    *["| " + " | ".join(row) + " |" for row in formatted],
                ]
            )
            + "\n"
        )
    lines = [
        "% Generated by scripts/paper/token_lengths.py; do not edit.",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(tasks) + "}",
        r"\toprule",
        " & ".join(map(tex_escape, header)) + r" \\",
        r"\midrule",
        *[" & ".join(map(tex_escape, row)) + r" \\" for row in formatted],
        r"\bottomrule",
        r"\end{tabular}}",
    ]
    return "\n".join(lines) + "\n"


def generate(pattern, tables, artifacts):
    from transformers import AutoTokenizer

    rows, sources, pending = [], [], []
    tokenizers, counts = {}, {}
    configs = [
        c
        for c in run.experiment_configs
        if fnmatch.fnmatchcase(c["name"], pattern)
        and not c.get("chat_reference")
        and c.get("evaluation_limit") is None
    ]
    for config in configs:
        marker = run.job_root(config["name"], None) / "done.json"
        if not marker.exists():
            pending.append(config["name"])
            continue
        output = Path(json.loads(marker.read_text())["output"])
        saved_path, samples = output / "experiment.json", output / "samples.jsonl"
        saved = json.loads(saved_path.read_text())
        # The retired MMLU baseline saved an unused zero-shot field.
        if saved.get("fewshot") == 0:
            saved.pop("fewshot")
        if saved != config:
            raise ValueError(f"Completed config differs from source: {config['name']}")
        snapshot = config["snapshot"]
        if snapshot not in tokenizers:
            tokenizers[snapshot] = AutoTokenizer.from_pretrained(
                snapshot, local_files_only=True
            )
        tokenizer = tokenizers[snapshot]

        def count(text, snapshot=snapshot, tokenizer=tokenizer):
            key = (snapshot, hashlib.sha256(text.encode()).digest())
            if key not in counts:
                counts[key] = len(tokenizer.encode(text, add_special_tokens=False))
            return counts[key]

        records = [json.loads(line) for line in samples.read_text().splitlines()]
        rows.append(summarize(records, config, count))
        sources.append(
            dict(
                experiment=config["name"],
                output=str(output),
                samples_sha256=digest(samples),
                config_sha256=digest(saved_path),
                marker_sha256=digest(marker),
                tokenizer_snapshot=snapshot,
            )
        )
        print(f"Counted {config['name']}: {len(records)} questions", flush=True)
    if not rows:
        raise ValueError("No completed generative runs matched the filter")
    tokenizer_sources = {
        snapshot: {
            p.name: digest(p)
            for p in sorted(Path(snapshot).glob("*"))
            if p.is_file()
            and ("token" in p.name or p.name in ("vocab.json", "merges.txt"))
        }
        for snapshot in tokenizers
    }
    tables.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "token-lengths.json").write_text(
        json.dumps(
            dict(
                method=METHOD,
                filter=pattern,
                rows=rows,
                pending=pending,
                sources=sources,
                tokenizers=tokenizer_sources,
            ),
            indent=2,
        )
        + "\n"
    )
    with (artifacts / "token-lengths.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [METHOD, ""]
    for metric in ("mean_input_tokens", "mean_output_text_tokens", "questions"):
        (tables / f"{metric}.tex").write_text(render(rows, metric, latex=True))
        markdown.extend([metric, "", render(rows, metric)])
    (artifacts / "token-lengths.md").write_text("\n".join(markdown))
    (tables / "token-lengths-provenance.json").write_text(
        (artifacts / "token-lengths.json").read_text()
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="*baseline*")
    parser.add_argument("--tables", type=Path, default=ROOT / "paper/tables")
    parser.add_argument(
        "--output", type=Path, default=run.ARTIFACTS / "paper-token-lengths"
    )
    args = parser.parse_args()
    generate(args.filter, args.tables, args.output)
