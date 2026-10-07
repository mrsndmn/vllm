# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pinned upstream RULER v1 generators, chat adapter, and substring grading."""

import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

import benchmark_data

import run


def source(settings):
    root = run.ARTIFACTS / "runtime" / ("ruler-" + settings["revision"])
    marker = root / "source-manifest.json"
    if marker.exists():
        provenance = json.loads(marker.read_text())
        if (
            provenance["revision"] != settings["revision"]
            or provenance["archive_sha256"] != settings["source_sha256"]
        ):
            raise ValueError("RULER source revision mismatch")
        for name, checksum in provenance["files"].items():
            if benchmark_data.digest(root / name) != checksum:
                raise ValueError(f"RULER source checksum mismatch: {name}")
        check_corpus(root)
        return root, provenance
    root.mkdir(parents=True, exist_ok=True)
    raw = benchmark_data.download(
        "https://codeload.github.com/NVIDIA/RULER/tar.gz/" + settings["revision"]
    )
    if hashlib.sha256(raw).hexdigest() != settings["source_sha256"]:
        raise ValueError("RULER archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(*Path(member.name).parts[1:])
            target = root / relative
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid RULER archive path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    corpus = root / "scripts/data/synthetic/json"
    for name, url in settings["corpus_urls"].items():
        target = corpus / name
        if not target.exists():
            target.write_bytes(benchmark_data.download(url))
    essay = corpus / "PaulGrahamEssays.json"
    failures = []
    if not essay.exists():
        import html2text
        from bs4 import BeautifulSoup

        urls = (corpus / "PaulGrahamEssays_URLs.txt").read_text().splitlines()

        def fetch(url):
            try:
                raw = benchmark_data.download(
                    url.replace("http://", "https://"), timeout=15
                )
                if ".html" in url:
                    html = raw.decode("unicode_escape")
                    converter = html2text.HTML2Text()
                    converter.ignore_images = converter.ignore_tables = True
                    converter.escape_all = True
                    converter.reference_links = converter.mark_code = False
                    value = converter.handle(
                        str(BeautifulSoup(html, "html.parser").find("font"))
                    )
                else:
                    value = raw.decode()
                return url, value, None
            except Exception as error:
                return url, None, str(error)

        with concurrent.futures.ThreadPoolExecutor(32) as pool:
            values = list(pool.map(fetch, urls))
        successes = [(url, text) for url, text, error in values if error is None]
        failures = [dict(url=url, error=error) for url, _, error in values if error]
        # Match upstream's ordering: sorted text files, followed by sorted HTML files.
        successes.sort(key=lambda item: (".html" in item[0], item[0].split("/")[-1]))
        text = "".join(value for _, value in successes)
        if len(text) < 500_000:
            raise ValueError("Incomplete RULER essay corpus")
        essay.write_text(json.dumps({"text": text}))
    check_corpus(root)
    provenance = dict(
        revision=settings["revision"],
        archive_sha256=settings["source_sha256"],
        corpus_download_failures=failures,
        files={
            str(p.relative_to(root)): benchmark_data.digest(p)
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    marker.write_text(json.dumps(provenance, indent=2) + "\n")
    return root, provenance


def check_corpus(root):
    for name, checksum in run.RULER_CORPUS_SHA256.items():
        if (
            benchmark_data.digest(root / "scripts/data/synthetic/json" / name)
            != checksum
        ):
            raise ValueError(f"RULER corpus checksum mismatch: {name}")


def prepare(root, settings):
    """Invoke unchanged upstream generators and freeze actual chat token counts."""
    import yaml
    from transformers import AutoTokenizer

    upstream, provenance = source(settings)
    data = upstream / "scripts/data"
    spec = importlib.util.spec_from_file_location(
        "ruler_generator_constants", data / "synthetic/constants.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    customized = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    tokenizer = AutoTokenizer.from_pretrained(run.BASE_CONFIG["snapshot"])
    benchmark_data.validate(
        run.ARTIFACTS / "benchmarks/ifeval", run.BENCHMARK_CONFIGS["ifeval"]
    )
    env = os.environ.copy()
    env["NLTK_DATA"] = str(run.ARTIFACTS / "benchmarks/ifeval/nltk_data")

    def generate_task(subtask):
        config = customized[subtask]
        base = module.TASKS[config["task"]]
        command = [
            sys.executable,
            str(data / "synthetic" / (config["task"] + ".py")),
            "--save_dir",
            str(root / "generated"),
            "--save_name",
            subtask,
            "--subset",
            "test",
            "--tokenizer_path",
            run.BASE_CONFIG["snapshot"],
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            str(settings["context_length"] - settings["template_reserve"]),
            "--tokens_to_generate",
            str(base["tokens_to_generate"]),
            "--num_samples",
            str(settings["samples_per_task"]),
            "--random_seed",
            str(settings["random_seed"]),
            "--template",
            base["template"] + base.get("answer_prefix", ""),
        ]
        for key, value in config["args"].items():
            command.extend(["--" + key, str(value)])
        print(f"Generating {root.name}/{subtask}", flush=True)
        with (root / (subtask + "-generation.log")).open("w") as log:
            subprocess.run(
                command,
                cwd=data,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        path = root / "generated" / subtask / "test.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != settings["samples_per_task"]:
            raise ValueError(f"Wrong RULER sample count: {subtask}")
        return base, rows

    with concurrent.futures.ThreadPoolExecutor(settings["preparation_workers"]) as pool:
        generated = list(pool.map(generate_task, settings["subtasks"]))
    questions = {}
    for subtask, (base, rows) in zip(settings["subtasks"], generated):
        questions[subtask] = []
        for index, row in enumerate(rows):
            question = dict(
                id=f"{subtask}/{index}",
                subtask=subtask,
                question=row["input"],
                answers=row["outputs"],
                answer_prefix=row.get("answer_prefix", ""),
                tokens_to_generate=base["tokens_to_generate"],
            )
            prompt = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + question["answer_prefix"]
            )
            question["input_tokens"] = check_budget(
                prompt,
                question,
                {"max_model_len": settings["context_length"]},
                tokenizer,
            )
            questions[subtask].append(question)
    # Round-robin order makes --limit 13 a smoke sample of every task.
    rows = [
        questions[task][i]
        for i in range(settings["samples_per_task"])
        for task in settings["subtasks"]
    ]
    path = root / "source-provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return rows, [path]


def check_budget(prompt, question, config, tokenizer):
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if tokens + question["tokens_to_generate"] > config["max_model_len"]:
        raise ValueError(f"RULER prompt exceeds context budget: {question['id']}")
    if "input_tokens" in question and tokens != question["input_tokens"]:
        raise ValueError(f"RULER tokenizer changed: {question['id']}")
    return tokens


def grade(record):
    """Official case-insensitive substring recall (any alias for QA)."""
    answers = record["answers"]
    if not answers or any(not answer for answer in answers):
        raise ValueError("Empty RULER gold answers")
    matches = [answer.lower() in record["response"].lower() for answer in answers]
    score = (
        float(any(matches))
        if record["subtask"].startswith("qa_")
        else statistics.mean(matches)
    )
    return dict(record, prediction=record["response"], score=score, correct=score == 1)


def metrics(records):
    groups = {}
    for record in records:
        groups.setdefault(record["subtask"], []).append(record["score"])
    by_task = {name: statistics.mean(scores) for name, scores in groups.items()}
    return dict(
        accuracy=statistics.mean(by_task.values()),
        subtask_scores=by_task,
        mean_input_tokens=statistics.mean(r["input_tokens"] for r in records),
        max_input_tokens=max(r["input_tokens"] for r in records),
    )
