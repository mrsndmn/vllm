"""Generative chat-format MMLU STEM evaluation through the local rounding server.

Each sampled question is asked zero-shot with the model's chat template (thinking
disabled). The model reasons briefly and ends with "Answer: X". Reasoning is
capped; a capped or answer-less response is continued inside the same assistant
turn with a forced-answer suffix and a few tokens, from which the letter is
parsed. `summarize` validates finished MLSpace runs and compares them.
"""

import argparse
import concurrent.futures
import csv
import json
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

TASK = "mmlu_stem_chat"
PROMPT = (
    "Answer the following multiple choice question. Think step by step, but keep "
    "your reasoning short. The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
    "ABCD.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
INTERRUPTED = "\n\n[reasoning interrupted] Answer:"
MISSING = "\n\nAnswer:"
# Letter possibly wrapped in markdown, quotes, parentheses, $...$ or \boxed{...}.
LETTER = r"(?:\\boxed\{|\\text\{|[\s*_`'\"$(\[])*([ABCD])(?![A-Za-z])"
FINAL = re.compile(r"(?i:answer)\s*[:：]" + LETTER)
FORCED = re.compile(LETTER)


def final_answer(text):
    """Letter from the last line containing "Answer:", if it names one."""
    lines = [
        line for line in text.splitlines() if re.search(r"(?i)answer\s*[:：]", line)
    ]
    matches = FINAL.findall(lines[-1]) if lines else []
    return matches[-1] if matches else None


def forced_answer(text):
    """Letter at the start of a forced-answer continuation."""
    match = FORCED.match(text)
    return match[1] if match else None


def load_test_split(subject):
    from datasets import load_dataset

    return load_dataset("cais/mmlu", subject, split="test")


def select_questions(config, per_subject, load=load_test_split):
    """Seeded sample of per_subject test questions from every configured subject."""
    picked = []
    for subject in config["chat_subjects"]:
        rows = load(subject)
        rng = random.Random(f"{config['chat_seed']}:{subject}")
        for index in sorted(rng.sample(range(len(rows)), per_subject)):
            row = rows[index]
            picked.append(
                {
                    "subject": subject,
                    "index": index,
                    "question": row["question"].strip(),
                    "choices": [str(choice) for choice in row["choices"]],
                    "answer": "ABCD"[int(row["answer"])],
                }
            )
    return picked


def reference_records(name, limit=None):
    """Samples of a finished chat run; limit keeps the first N per subject."""
    import run

    marker = run.job_root(name, None) / "done.json"
    output = Path(json.loads(marker.read_text())["output"])
    records, kept = [], {}
    for record in map(json.loads, (output / "samples.jsonl").open()):
        kept[record["subject"]] = kept.get(record["subject"], 0) + 1
        if not limit or kept[record["subject"]] <= limit:
            records.append(record)
    return records


def reference_response(record):
    """The assistant text a user saw: reasoning plus any forced answer."""
    return (
        record["reasoning"]
        + record.get("forced_suffix", "")
        + record.get("forced_text", "")
    )


def score_reference(prompt_ids, response_ids, post, config):
    """Log-likelihood of a fixed response under the served (zeroed) model."""
    result = post(
        {
            "model": config["model"],
            "prompt": prompt_ids + response_ids,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
        }
    )
    logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
    scored = logprobs[len(prompt_ids) : len(prompt_ids) + len(response_ids)]
    if len(scored) != len(response_ids) or None in scored:
        raise RuntimeError("Missing prompt log-probabilities for reference response")
    return {"reference_tokens": len(response_ids), "reference_logprob": sum(scored)}


def chat_prompt(tokenizer, question):
    content = PROMPT.format(
        question=question["question"], **dict(zip("ABCD", question["choices"]))
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def answer(question, prompt, post, config):
    """Two-stage greedy answer: capped reasoning, then a forced letter if needed."""
    body = {"model": config["model"], "temperature": 0}
    first = post(
        {**body, "prompt": prompt, "max_tokens": config["chat_max_reasoning_tokens"]}
    )
    choice = first["choices"][0]
    capped = choice["finish_reason"] == "length"
    record = {
        **question,
        "prompt": prompt,
        "reasoning": choice["text"],
        "reasoning_tokens": first["usage"]["completion_tokens"],
        "finish_reason": choice["finish_reason"],
    }
    letter = None if capped else final_answer(choice["text"])
    if letter:
        record["mode"] = "natural"
    else:
        suffix = INTERRUPTED if capped else MISSING
        second = post(
            {
                **body,
                "prompt": prompt + choice["text"] + suffix,
                "max_tokens": config["chat_max_answer_tokens"],
            }
        )
        record["forced_suffix"] = suffix
        record["forced_text"] = second["choices"][0]["text"]
        record["mode"] = "forced_cap" if capped else "forced_missing"
        letter = forced_answer(record["forced_text"])
    record["prediction"] = letter
    record["correct"] = letter == question["answer"]
    return record


def poster(url, attempts=3):
    def post(body):
        request = urllib.request.Request(
            url + "/v1/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    return json.load(response)
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == attempts - 1:
                    raise
                time.sleep(10 * (attempt + 1))

    return post


def summarize_records(records, config):
    tokens = [r["reasoning_tokens"] for r in records]
    subjects = sorted({r["subject"] for r in records})
    modes = ("natural", "forced_cap", "forced_missing")
    reference = {}
    if records and "reference_tokens" in records[0]:
        total = sum(r["reference_tokens"] for r in records)
        logprob = sum(r["reference_logprob"] for r in records)
        reference = {
            "reference": {
                "name": config.get("chat_reference"),
                "tokens": total,
                "logprob": logprob,
                "nll_per_token": -logprob / total,
            }
        }
    return {
        "task": TASK,
        "questions": len(records),
        "correct": sum(r["correct"] for r in records),
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "modes": {mode: sum(r["mode"] == mode for r in records) for mode in modes},
        "unparsed": sum(r["prediction"] is None for r in records),
        "reasoning_tokens": {
            "mean": statistics.mean(tokens),
            "median": statistics.median(tokens),
            "max": max(tokens),
        },
        "per_subject_accuracy": {
            s: statistics.mean(r["correct"] for r in records if r["subject"] == s)
            for s in subjects
        },
        "prompt": PROMPT,
        "interrupted_suffix": INTERRUPTED,
        "missing_suffix": MISSING,
        **{key: config[key] for key in config if key.startswith("chat_")},
        **reference,
    }


def run_eval(
    config, output, url, limit=None, load=load_test_split, post=None, tokenizer=None
):
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["snapshot"])
    reference = config.get("chat_reference")
    if reference:  # Same questions as the reference run, in its order.
        references = reference_records(reference, limit)
        fields = ("subject", "index", "question", "choices", "answer")
        questions = [{key: r[key] for key in fields} for r in references]
    else:
        questions = select_questions(config, limit or config["chat_per_subject"], load)
    post = post or poster(url)
    prompts = [chat_prompt(tokenizer, question) for question in questions]
    with concurrent.futures.ThreadPoolExecutor(config["chat_concurrency"]) as pool:
        records = list(
            pool.map(lambda q, p: answer(q, p, post, config), questions, prompts)
        )
        if reference:
            # Tokenize in this thread; fast tokenizers are not safe to share.
            pairs = [
                (
                    tokenizer.encode(r["prompt"], add_special_tokens=False),
                    tokenizer.encode(reference_response(r), add_special_tokens=False),
                )
                for r in references
            ]
            scores = pool.map(lambda pair: score_reference(*pair, post, config), pairs)
            for record, score in zip(records, scores):
                record.update(score)
    with (output / "samples.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    result = summarize_records(records, config)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("accuracy", "modes", "unparsed")}))
    return result


def layer_zero_shares(output):
    before = json.loads((output / "rounding-before.json").read_text())["layers"]
    after = json.loads((output / "rounding-after.json").read_text())["layers"]
    shares = []
    for index, layer in enumerate(after):
        old = before[index] if before else dict.fromkeys(layer, 0)
        floats = layer["floats"] - old["floats"]
        zero = (
            layer["newly_zeroed_floats"]
            + layer["already_zero_floats"]
            - old["newly_zeroed_floats"]
            - old["already_zero_floats"]
        )
        shares.append(100 * zero / floats)
    return shares


def finished_runs(pattern="*"):
    """(config, output, records by question) of finished chat runs matching a glob."""
    import fnmatch

    import run

    runs = {}
    for config in run.experiment_configs:
        name = config.get("name", "")
        marker = run.job_root(name, None) / "done.json"
        if config.get("task") != TASK or not fnmatch.fnmatchcase(name, pattern):
            continue
        if not marker.exists():
            continue
        output = Path(json.loads(marker.read_text())["output"])
        records = [json.loads(line) for line in (output / "samples.jsonl").open()]
        keys = {(r["subject"], r["index"]) for r in records}
        expected = len(config["chat_subjects"]) * config["chat_per_subject"]
        assert len(records) == len(keys) == expected, output
        assert all(r["correct"] == (r["prediction"] == r["answer"]) for r in records)
        runs[config["name"]] = (
            config,
            output,
            {(r["subject"], r["index"]): r for r in records},
        )
    return runs


def summarize(pattern="*", baseline="chat_baseline"):
    """Validate finished chat runs and write summary CSV/JSON with answer changes."""
    import run

    runs = finished_runs(pattern)
    if baseline not in runs:
        runs = {**finished_runs(baseline), **runs}
    reference = runs.get(baseline, (None, None, None))[2]
    rows = []
    for name, (config, output, records) in runs.items():
        result = json.loads((output / "results.json").read_text())
        shares = layer_zero_shares(output)
        row = {
            "experiment": name,
            "site": config["round_activations_site"],
            "quantile": config["round_activations_quantile"],
            "accuracy_percent": 100 * result["accuracy"],
            "correct": result["correct"],
            "questions": result["questions"],
            **{f"{mode}": count for mode, count in result["modes"].items()},
            "unparsed": result["unparsed"],
            "mean_reasoning_tokens": result["reasoning_tokens"]["mean"],
            "min_layer_total_zero_percent": min(shares),
            "max_layer_total_zero_percent": max(shares),
        }
        if config.get("round_activations_quantiles"):
            row["quantiles"] = ",".join(map(str, config["round_activations_quantiles"]))
        if "reference" in result:
            row["reference_nll_per_token"] = result["reference"]["nll_per_token"]
        if reference and name != baseline:
            row["changed_vs_baseline"] = sum(
                records[k]["prediction"] != reference[k]["prediction"] for k in records
            )
            row["fixed_vs_baseline"] = sum(
                records[k]["correct"] and not reference[k]["correct"] for k in records
            )
            row["broken_vs_baseline"] = sum(
                reference[k]["correct"] and not records[k]["correct"] for k in records
            )
        rows.append(row)
    if not rows:
        raise SystemExit("No completed chat runs")
    stem = run.ARTIFACTS / "rounding-chat-sweep"
    with stem.with_name(stem.name + "-summary.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    stem.with_name(stem.name + "-summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "summarize"])
    parser.add_argument("--config", type=Path, help="experiment.json for run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--limit", type=int, help="Questions per subject")
    parser.add_argument("--filter", default="*", help="Run-name glob for summarize")
    args = parser.parse_args()
    if args.action == "summarize":
        summarize(args.filter)
    else:
        config = json.loads(args.config.read_text())
        run_eval(config, args.output, args.url, args.limit)


if __name__ == "__main__":
    main()
