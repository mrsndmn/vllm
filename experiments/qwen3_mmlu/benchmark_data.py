# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare pinned benchmark inputs and official graders on shared storage."""

import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import run


def encode_value(value):
    """Preserve Python oracle types without executable object serialization."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return {
            "type": "array",
            "dtype": str(value.dtype),
            "value": encode_value(value.tolist()),
        }
    if isinstance(value, np.generic):
        return encode_value(value.item())
    if isinstance(value, int) and value.bit_length() > 4096:
        return {"type": "int", "value": hex(value)}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": [[encode_value(k), encode_value(v)] for k, v in value.items()],
        }
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    if isinstance(value, (tuple, set, frozenset)):
        return {"type": type(value).__name__, "value": [encode_value(v) for v in value]}
    if isinstance(value, complex):
        return {"type": "complex", "value": [value.real, value.imag]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported oracle type: {type(value)}")


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "int":
        return int(value["value"], 16)
    if kind == "dict":
        return {decode_value(k): decode_value(v) for k, v in value["value"]}
    if kind == "array":
        import numpy as np

        return np.array(decode_value(value["value"]), dtype=value["dtype"])
    if kind == "complex":
        return complex(*value["value"])
    constructors = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
    return constructors[kind](decode_value(v) for v in value["value"])


def oracle_path(root, key):
    return root / "oracles" / (hashlib.sha256(key.encode()).hexdigest() + ".json.gz")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def code_problems(task):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    return (get_human_eval_plus if task == "humaneval_plus" else get_mbpp_plus)()


def prepare(task):
    """Freeze questions, evaluator files, and code oracles before MLS submission."""
    settings = run.BENCHMARK_CONFIGS[task]
    if task in ("humaneval_plus", "mbpp_plus"):
        grader = run.CODE_GRADER
        path = Path(grader["path"])
        if not path.exists():
            raw = download(
                "https://raw.githubusercontent.com/evalplus/evalplus/"
                + grader["revision"]
                + "/evalplus/eval/__init__.py"
            )
            if hashlib.sha256(raw).hexdigest() != grader["sha256"]:
                raise ValueError("Upstream code grader checksum mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        if digest(path) != grader["sha256"]:
            raise ValueError("Code grader checksum mismatch")
    for package, version in packages(settings).items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Grader version mismatch: {package}")
    root = run.ARTIFACTS / "benchmarks" / task
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        validate(root, settings)
        print(f"Validated prepared {task}", flush=True)
        return
    extra = []
    if task.startswith("ruler_"):
        import ruler

        questions, extra = ruler.prepare(root, settings)
    elif task in ("humaneval_plus", "mbpp_plus"):
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.gen.util import trusted_exec

        problems = code_problems(task)
        questions = []
        for key, problem in problems.items():
            questions.append(
                dict(
                    id=key,
                    question=problem["prompt"],
                    entry_point=problem["entry_point"],
                )
            )
            oracle = {}
            for kind in ("base", "plus"):
                values, times = trusted_exec(
                    problem["prompt"] + problem["canonical_solution"],
                    problem[kind + "_input"],
                    problem["entry_point"],
                    record_time=True,
                    output_not_none=(
                        task == "mbpp_plus"
                        and problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
                    ),
                )
                oracle[kind] = (values, times)
            target = oracle_path(root, key)
            target.parent.mkdir(exist_ok=True)
            with gzip.open(target, "wt", compresslevel=1) as stream:
                json.dump(encode_value((problem, oracle)), stream)
            extra.append(target)
    elif task == "gpqa":
        raw = download(
            "https://raw.githubusercontent.com/idavidrein/gpqa/"
            + settings["revision"]
            + "/dataset.zip"
        )
        archive = zipfile.ZipFile(io.BytesIO(raw))
        member = next(n for n in archive.namelist() if n.endswith("gpqa_diamond.csv"))
        rows = csv.DictReader(
            io.StringIO(
                archive.read(member, pwd=settings["archive_password"].encode()).decode()
            )
        )
        questions = []
        rng = random.Random(0)
        for index, row in enumerate(rows):
            options = [(row["Correct Answer"], True)] + [
                (row[f"Incorrect Answer {i}"], False) for i in range(1, 4)
            ]
            rng.shuffle(options)
            questions.append(
                dict(
                    id=str(index),
                    question=row["Question"],
                    choices=[text for text, _ in options],
                    answer="ABCD"[
                        next(i for i, (_, correct) in enumerate(options) if correct)
                    ],
                )
            )
    else:
        from datasets import load_dataset

        rows = load_dataset(
            settings["dataset"], revision=settings["revision"], split=settings["split"]
        )
        questions = []
        for index, row in enumerate(rows):
            if task == "math500":
                question = dict(
                    id=row["unique_id"],
                    question=row["problem"],
                    answer=row["answer"],
                    subject=row["subject"],
                )
            elif task == "mmlu_pro":
                question = dict(
                    id=str(row["question_id"]),
                    question=row["question"],
                    choices=row["options"],
                    answer=row["answer"],
                    subject=row["category"],
                )
            else:
                question = dict(
                    id=str(row["key"]),
                    question=row["prompt"],
                    instruction_id_list=row["instruction_id_list"],
                    kwargs=[
                        {k: v for k, v in args.items() if v is not None}
                        for args in row["kwargs"]
                    ],
                )
            questions.append(question)
        if task == "ifeval":
            package = root / "instruction_following_eval"
            package.mkdir(exist_ok=True)
            (package / "__init__.py").write_text("")
            extra.append(package / "__init__.py")
            for name in (
                "evaluation_lib.py",
                "instructions.py",
                "instructions_registry.py",
                "instructions_util.py",
            ):
                target = package / name
                target.write_bytes(
                    download(
                        "https://raw.githubusercontent.com/google-research/google-research/"
                        + settings["evaluator_revision"]
                        + "/instruction_following_eval/"
                        + name
                    )
                )
                extra.append(target)
            nltk_root = root / "nltk_data"
            archive = zipfile.ZipFile(
                io.BytesIO(
                    download(
                        "https://raw.githubusercontent.com/nltk/nltk_data/"
                        + settings["nltk_revision"]
                        + "/packages/tokenizers/punkt_tab.zip"
                    )
                )
            )
            for member in archive.namelist():
                if member.startswith("punkt_tab/english/") and not member.endswith("/"):
                    target = nltk_root / "tokenizers" / member
                    if not target.resolve().is_relative_to(nltk_root.resolve()):
                        raise ValueError("Invalid tokenizer archive path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
            extra.extend(p for p in nltk_root.rglob("*") if p.is_file())
    assert len(questions) == settings["expected_questions"], (task, len(questions))
    assert len({q["id"] for q in questions}) == len(questions)
    path = root / "questions.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in questions))
    extra.append(path)
    manifest = dict(
        benchmark=settings,
        packages=packages(settings),
        files={str(p.relative_to(root)): digest(p) for p in extra},
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {task}: {len(questions)} questions", flush=True)


def packages(settings):
    return run.BENCHMARK_PACKAGES | (
        run.RULER_PACKAGES if settings.get("dataset") == "NVIDIA/RULER" else {}
    )


def validate(root, settings):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["benchmark"] != settings:
        raise ValueError("Prepared benchmark differs from source config")
    if manifest["packages"] != packages(settings):
        raise ValueError("Prepared grader versions differ from source config")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Prepared artifact checksum mismatch: {name}")
    return manifest


if __name__ == "__main__":
    for task in run.BENCHMARK_CONFIGS:
        prepare(task)
