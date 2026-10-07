# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generative transfer benchmarks with official programmatic grading."""

import argparse
import concurrent.futures
import csv
import functools
import gzip
import importlib.metadata
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import benchmark_data
import chat_eval
import regex as re

import run


def content(question, task):
    if task == "ifeval":
        return question["question"]
    if task in ("humaneval_plus", "mbpp_plus"):
        return (
            "Write a complete Python solution for the following task. Return only "
            "Python code, including the function definition and any imports.\n\n"
            + question["question"]
        )
    if task == "math500":
        return (
            "Solve the following mathematics problem. Show your reasoning and "
            "put the final answer in \\boxed{...}.\n\n" + question["question"]
        )
    options = "\n".join(
        f"{chr(65 + i)}) {text}" for i, text in enumerate(question["choices"])
    )
    return (
        "Answer the following multiple choice question. Think step by step. "
        "End with 'Answer: LETTER'.\n\n" + question["question"] + "\n\n" + options
    )


def choice_answer(text, count):
    lines = [
        line for line in text.splitlines() if re.search(r"(?i)answer\s*[:：]", line)
    ]
    if not lines:
        return None
    match = re.fullmatch(
        r"\s*[*_`'\"$(\[]*([A-J])[*_`'\"$)\].]*\s*",
        re.split(r"(?i)answer\s*[:：]", lines[-1])[-1],
    )
    return match[1] if match and ord(match[1]) - 65 < count else None


def generate(question, prompt, config, post):
    response = post(
        dict(
            model=config["model"],
            prompt=prompt,
            temperature=0,
            max_tokens=config["generation_max_tokens"],
        )
    )
    choice = response["choices"][0]
    return dict(
        **question,
        prompt=prompt,
        response=choice["text"],
        finish_reason=choice["finish_reason"],
        completion_tokens=response["usage"]["completion_tokens"],
    )


@functools.lru_cache
def load_code_grader(path, checksum):
    if benchmark_data.digest(Path(path)) != checksum:
        raise ValueError("Code grader checksum mismatch")
    name = "evalplus_transfer_" + checksum
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def code_grade(record, config, problem, oracle):
    from evalplus.sanitize import sanitize

    grader = load_code_grader(
        config["code_grader"]["path"], config["code_grader"]["sha256"]
    )
    solution = sanitize(record["response"], entrypoint=problem["entry_point"])
    statuses = {}
    for kind in ("base", "plus"):
        values, times = oracle[kind]
        status, _ = grader.untrusted_check(
            config["benchmark"]["dataset"],
            solution,
            problem[kind + "_input"],
            problem["entry_point"],
            expected=values,
            atol=problem["atol"],
            ref_time=times,
            fast_check=True,
            min_time_limit=config["code_min_time_limit"],
            gt_time_limit_factor=config["code_time_limit_factor"],
        )
        statuses[kind + "_passed"] = status == grader.PASS
    return dict(
        **record,
        solution=solution,
        **statuses,
        correct=all(statuses.values()) and record["finish_reason"] != "length",
    )


def grade(record, task):
    correct, prediction = False, None
    extra = {}
    if task == "math500":
        from math_verify import LatexExtractionConfig, parse, verify

        gold = parse(
            "$" + record["answer"] + "$", extraction_config=[LatexExtractionConfig()]
        )
        if not gold:
            raise ValueError(f"Unparsable MATH gold: {record['id']}")
        parsed = parse(record["response"], extraction_config=[LatexExtractionConfig()])
        prediction = str(parsed) if parsed else None
        correct = bool(verify(gold, parsed))
    elif task == "ifeval":
        from instruction_following_eval import evaluation_lib

        random.seed(int(record["id"]))
        inp = evaluation_lib.InputExample(
            key=int(record["id"]),
            instruction_id_list=record["instruction_id_list"],
            prompt=record["question"],
            kwargs=record["kwargs"],
        )
        responses = {record["question"]: record["response"]}
        strict = evaluation_lib.test_instruction_following_strict(inp, responses)
        loose = evaluation_lib.test_instruction_following_loose(inp, responses)
        correct = strict.follow_all_instructions
        extra = dict(
            strict_instructions=strict.follow_instruction_list,
            loose_instructions=loose.follow_instruction_list,
            loose_correct=loose.follow_all_instructions,
        )
    else:
        prediction = choice_answer(record["response"], len(record["choices"]))
        correct = prediction is not None and prediction == record["answer"]
    return dict(
        **record,
        prediction=prediction,
        correct=bool(correct) and record["finish_reason"] != "length",
        **extra,
    )


def code_grade_saved(record, config):
    path = benchmark_data.oracle_path(Path(config["benchmark_data"]), record["id"])
    with gzip.open(path, "rt") as stream:
        problem, oracle = benchmark_data.decode_value(json.load(stream))
    return code_grade(record, config, problem, oracle)


def metrics(records, task):
    result = dict(
        task=task,
        questions=len(records),
        correct=sum(r["correct"] for r in records),
        accuracy=sum(r["correct"] for r in records) / len(records),
        truncated=sum(r["finish_reason"] == "length" for r in records),
        mean_completion_tokens=statistics.mean(r["completion_tokens"] for r in records),
    )
    if task == "ifeval":
        result.update(
            strict_prompt_accuracy=result["accuracy"],
            loose_prompt_accuracy=sum(
                r["loose_correct"] and r["finish_reason"] != "length" for r in records
            )
            / len(records),
        )
        for mode in ("strict", "loose"):
            outcomes = [
                passed and r["finish_reason"] != "length"
                for r in records
                for passed in r[mode + "_instructions"]
            ]
            result[mode + "_instruction_accuracy"] = sum(outcomes) / len(outcomes)
    elif task in ("humaneval_plus", "mbpp_plus"):
        result["pass_at_1_plus"] = result["accuracy"]
        result["pass_at_1_base"] = sum(
            r["base_passed"] and r["finish_reason"] != "length" for r in records
        ) / len(records)
    else:
        result["unparsed"] = sum(r["prediction"] is None for r in records)
    return result


def run_eval(config, output, url, limit=None, tokenizer=None, post=None):
    task = config["task"]
    root = Path(config["benchmark_data"])
    manifest = benchmark_data.validate(root, config["benchmark"])
    if config.get("code_grader"):
        load_code_grader(config["code_grader"]["path"], config["code_grader"]["sha256"])
    for package, version in config["benchmark_packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Grader version mismatch: {package}")
    questions = [
        json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines()
    ]
    assert len(questions) == config["benchmark"]["expected_questions"]
    if limit is not None:
        if not 0 < limit <= len(questions):
            raise ValueError("Invalid benchmark limit")
        questions = questions[:limit]
    if task == "ifeval":
        import nltk
        from langdetect import DetectorFactory

        DetectorFactory.seed = 0
        sys.path.insert(0, str(root))
        nltk.data.path.insert(0, str(root / "nltk_data"))
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["snapshot"])
    post = post or chat_eval.poster(url)
    # Tokenize in the calling thread, as fast tokenizers cannot be shared safely.
    prompts = {
        q["id"]: tokenizer.apply_chat_template(
            [{"role": "user", "content": content(q, task)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in questions
    }

    with concurrent.futures.ThreadPoolExecutor(config["chat_concurrency"]) as pool:
        records = list(
            pool.map(lambda q: generate(q, prompts[q["id"]], config, post), questions)
        )
    (output / "generations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    if task in ("humaneval_plus", "mbpp_plus"):
        with concurrent.futures.ProcessPoolExecutor(config["grading_workers"]) as pool:
            futures = [pool.submit(code_grade_saved, r, config) for r in records]
            records = [f.result() for f in futures]
    else:
        records = [grade(r, task) for r in records]
    result = metrics(records, task)
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def summarize():
    rows = []
    for task in run.BENCHMARK_CONFIGS:
        runs = {}
        for config in run.experiment_configs:
            if config["task"] != task:
                continue
            marker = run.job_root(config["name"], None) / "done.json"
            if not marker.exists():
                continue
            output = Path(json.loads(marker.read_text())["output"])
            if json.loads((output / "experiment.json").read_text()) != config:
                raise ValueError("Completed config differs from source")
            root = Path(config["benchmark_data"])
            manifest = benchmark_data.validate(root, config["benchmark"])
            assert manifest == json.loads((output / "data-manifest.json").read_text())
            questions = {
                q["id"]: q
                for q in map(
                    json.loads, (root / "questions.jsonl").read_text().splitlines()
                )
            }
            records = [
                json.loads(line)
                for line in (output / "samples.jsonl").read_text().splitlines()
            ]
            keyed = {r["id"]: r for r in records}
            assert (
                len(records) == len(keyed) == config["benchmark"]["expected_questions"]
            )
            assert keyed.keys() == questions.keys()
            assert all(
                all(r[field] == value for field, value in questions[key].items())
                for key, r in keyed.items()
            )
            result = metrics(records, task)
            assert result == json.loads((output / "results.json").read_text())
            runs[config["name"]] = (result, keyed)
        baseline = runs.get("suite_" + task + "_baseline")
        repeat = runs.get("suite_" + task + "_baseline_repeat")
        for name, (result, records) in runs.items():
            row = dict(experiment=name, **result)
            shares = chat_eval.layer_zero_shares(
                Path(
                    json.loads((run.job_root(name, None) / "done.json").read_text())[
                        "output"
                    ]
                )
            )
            row.update(
                min_layer_total_zero_percent=min(shares),
                max_layer_total_zero_percent=max(shares),
            )
            for label, reference in (("baseline", baseline), ("repeat", repeat)):
                if reference is None:
                    continue
                reference_result, reference_records = reference
                assert records.keys() == reference_records.keys()
                assert all(
                    records[k]["prompt"] == reference_records[k]["prompt"]
                    for k in records
                )
                row["delta_" + label + "_pp"] = 100 * (
                    result["accuracy"] - reference_result["accuracy"]
                )
                row["fixed_vs_" + label] = sum(
                    r["correct"] and not reference_records[k]["correct"]
                    for k, r in records.items()
                )
                row["broken_vs_" + label] = sum(
                    reference_records[k]["correct"] and not r["correct"]
                    for k, r in records.items()
                )
            rows.append(row)
    path = run.ARTIFACTS / "benchmarks-summary.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    pivot = {}
    for row in rows:
        arm = row["experiment"].removeprefix("suite_" + row["task"] + "_")
        pivot.setdefault(arm, {"experiment": arm})[row["task"]] = 100 * row["accuracy"]
    with (run.ARTIFACTS / "benchmarks-accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["experiment", *run.BENCHMARK_CONFIGS]
        )
        writer.writeheader()
        writer.writerows(pivot.values())
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "summarize"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "summarize":
        summarize()
    else:
        run_eval(json.loads(args.config.read_text()), args.output, args.url, args.limit)
