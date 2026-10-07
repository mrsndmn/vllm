# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Local persistent Qwen3 serving and the generative MMLU STEM evaluation."""

import argparse
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

# Generative chat-format check on a seeded MMLU STEM subset (chat_eval.py).
STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]
# Local serving and evaluation defaults; every registry row derives from them.
# All evaluation is generative (ADR 0004).
experiment_configs = [
    {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "snapshot": "/workspace-SR004.nfs2/.cache/huggingface/hub/"
        "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/"
        "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "python": "/home/jovyan/.mlspace/envs/vllm_cu126/bin/python",
        "artifacts": "/mnt/virtual_ai0001053-00054_SR004-nfs2/"
        "d.tarasov/vllm-artifacts/qwen3_mmlu",
        "session": "vllm-qwen3-30b-a3b",
        "port": 8000,
        "max_model_len": 8192,
        "memory_fraction": 0.80,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 1024,
        "task": "mmlu_stem_chat",
        "offline_evaluation": True,
        "chat_subjects": STEM_SUBJECTS,
        "chat_per_subject": 16,
        "chat_seed": 0,
        "chat_max_reasoning_tokens": 1024,
        "chat_max_answer_tokens": 5,
        "chat_concurrency": 16,
    }
]
CONFIG = experiment_configs[0]
BASE_CONFIG = CONFIG.copy()
# MLSpace job containers do not mount ~/.mlspace; this path is the same env.
JOB_PYTHON = "/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126/bin/python"
# Registry rows run as MLSpace jobs (run_jobs.py).
CHAT_CONFIG = dict(
    BASE_CONFIG, launcher="mls", instance_type="a100.1gpu", python=JOB_PYTHON
)
experiment_configs = [
    dict(
        CHAT_CONFIG,
        name=name or f"chat_{site}_q{quantile:g}",
        round_activations_site=site,
        round_activations_eps=0.0,
        round_activations_quantile=quantile,
    )
    # The repeat measures answer changes from batching nondeterminism alone.
    for name, site, quantile in (
        ("chat_baseline", "mlp", 0.0),
        ("chat_baseline_repeat", "mlp", 0.0),
        (None, "mlp", 0.25),
        (None, "mlp", 0.5),
        (None, "block", 0.25),
        (None, "block", 0.5),
        (None, "mlp", 0.6),
        (None, "mlp", 0.7),
        (None, "mlp", 0.8),
        (None, "block", 0.3),
        (None, "block", 0.4),
        (None, "mlp", 0.9),
        (None, "mlp", 0.95),
        (None, "mlp", 0.99),
    )
]
# Adaptive zeroing: one quantile per group of 6 consecutive layers (8 groups).
# Stage A profiles one group at a time on the chat_baseline questions; each job
# also re-scores chat_baseline's responses (sensitivity.py turns this into a
# per-group cost and allocates budgets).
GROUPS = 8
PROFILE_LEVELS = {"mlp": (0.5, 0.75, 0.9), "block": (0.25, 0.5, 0.75)}
# Stage B targets: mean zero share over the 8 groups (layers are equal-sized).
ADAPTIVE_BUDGETS = {"mlp": (0.6, 0.7, 0.8), "block": (0.3, 0.4, 0.5)}
PROFILE_CONFIG = dict(
    CHAT_CONFIG,
    chat_reference="chat_baseline",
    round_activations_eps=0.0,
    round_activations_quantile=0.0,
)
experiment_configs += [
    dict(
        PROFILE_CONFIG,
        name=name,
        round_activations_site="mlp",
        round_activations_quantiles=[0.0] * GROUPS,
    )
    for name in ("prof_baseline", "prof_baseline_repeat")
] + [
    dict(
        PROFILE_CONFIG,
        name=f"prof_{site}_g{group}_q{quantile:g}",
        round_activations_site=site,
        round_activations_quantiles=[
            quantile if index == group else 0.0 for index in range(GROUPS)
        ],
    )
    for site, levels in PROFILE_LEVELS.items()
    for group in range(GROUPS)
    for quantile in levels
]
# Stage C: schedules chosen by sensitivity.py from Stage A (filled after Stage A),
# keyed by (site, budget = mean zero share); each gets a chat row.
# From sensitivity.py on the prof_* runs (commit 747cfb715b); see
# adaptive-schedules.json. Block at 0.5 allocated [0.5] * 8, i.e. the existing
# uniform chat_block_q0.5 row, so it has no separate adaptive row.
ADAPTIVE_SCHEDULES = {
    ("mlp", 0.6): [0.75, 0.75, 0.75, 0.5, 0.75, 0.5, 0.5, 0.5],
    ("mlp", 0.7): [0.75, 0.75, 0.75, 0.5, 0.75, 0.75, 0.75, 0.75],
    ("mlp", 0.8): [0.9, 0.75, 0.9, 0.75, 0.75, 0.75, 0.75, 0.9],
    ("block", 0.3): [0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5],
    ("block", 0.4): [0.5, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
}


def adaptive_rows(schedules):
    return [
        dict(
            CHAT_CONFIG,
            name=f"chat_adapt_{site}_b{budget:g}",
            round_activations_site=site,
            round_activations_eps=0.0,
            round_activations_quantile=0.0,
            round_activations_quantiles=list(schedule),
        )
        for (site, budget), schedule in schedules.items()
    ]


experiment_configs += adaptive_rows(ADAPTIVE_SCHEDULES)
# Generative fixed-eps arms at every site, and quantile arms at moe_out (the MoE
# output alone, residual stream untouched; ADR 0003) paired with chat_block_q0.4
# and chat_block_q0.5 at the same per-token share; 0.7-0.99 match the high mlp
# quantile arms.
experiment_configs += [
    dict(
        CHAT_CONFIG,
        name=f"chat_{site}_eps{eps:g}",
        round_activations_site=site,
        round_activations_eps=eps,
        round_activations_quantile=0.0,
    )
    for site in ("mlp", "block", "moe_out")
    for eps in (0.01, 0.1, 0.5)
] + [
    dict(
        CHAT_CONFIG,
        name=f"chat_moe_out_q{quantile:g}",
        round_activations_site="moe_out",
        round_activations_eps=0.0,
        round_activations_quantile=quantile,
    )
    for quantile in (0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
]
GSM8K_CONFIG = dict(
    CHAT_CONFIG,
    task="gsm8k_chat",
    chat_dataset="openai/gsm8k",
    chat_dataset_config="main",
    chat_dataset_revision="740312add88f781978c0658806c59bc2815b9866",
    chat_split="test",
    chat_expected_questions=1319,
    chat_max_answer_tokens=32,
    offline_evaluation=False,
    round_activations_site="mlp",
    round_activations_eps=0.0,
    round_activations_quantile=0.0,
)
for key in ("chat_subjects", "chat_per_subject"):
    del GSM8K_CONFIG[key]
experiment_configs += [
    dict(GSM8K_CONFIG, name=name)
    for name in ("gsm8k_baseline", "gsm8k_baseline_repeat")
]
ARTIFACTS = Path(CONFIG["artifacts"])
URL = f"http://127.0.0.1:{CONFIG['port']}"
SCRIPT = Path(__file__).resolve()


def session_exists():
    try:
        result = subprocess.run(
            # "=" disables tmux prefix matching (other sessions may share a prefix).
            ["tmux", "has-session", "-t", "=" + CONFIG["session"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:  # MLSpace containers may lack tmux.
        return False
    return result.returncode == 0


def environment():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return env


def server_command(database):
    command = [
        CONFIG["python"],
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        CONFIG["snapshot"],
        "--served-model-name",
        CONFIG["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(CONFIG["port"]),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(CONFIG["max_model_len"]),
        "--gpu-memory-utilization",
        str(CONFIG["memory_fraction"]),
        "--max-num-seqs",
        str(CONFIG["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(CONFIG["max_num_batched_tokens"]),
        "--enforce-eager",
        "--enable-prefix-caching",
        "--disable-log-requests",
    ]
    if "round_activations_eps" in CONFIG:
        command[1:3] = [
            str(SCRIPT.with_name("rounding_server.py")),
            *rounding_args(),
            "--rounding-db",
            str(database),
        ]
    return command


def serve():
    command = server_command(ARTIFACTS / "rounding" / f"{uuid.uuid4().hex}.sqlite")
    os.chdir(ARTIFACTS)
    with (ARTIFACTS / "server.log").open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        print(f"\nStarting at {time.ctime()}: {shlex.join(command)}", flush=True)
        os.execve(command[0], command, environment())


def start():
    if session_exists():
        print("Server session already exists; use status or restart.")
        return
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", CONFIG["port"]))
    command = shlex.join([CONFIG["python"], str(SCRIPT), "serve", *rounding_args()])
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            CONFIG["session"],
            "-c",
            str(ARTIFACTS),
            command,
        ],
        check=True,
    )
    print(f"Started {CONFIG['session']}; see {ARTIFACTS / 'server.log'}")


def stop():
    if not session_exists():
        print("Server session is not running.")
        return
    subprocess.run(
        [
            "tmux",
            "send-keys",
            "-t",
            f"={CONFIG['session']}:",
            "C-c",
        ],
        check=True,
    )
    for _ in range(60):
        if not session_exists():
            print("Stopped server.")
            return
        time.sleep(1)
    raise RuntimeError("Server did not stop within 60 seconds; inspect server.log")


def status():
    print(f"tmux session: {'running' if session_exists() else 'absent'}")
    with urllib.request.urlopen(URL + "/v1/models", timeout=10) as response:
        models = json.load(response)
    if CONFIG["model"] not in {model["id"] for model in models["data"]}:
        raise RuntimeError("Endpoint is serving a different model")
    print(json.dumps(models, indent=2))


def rounding_args():
    if "round_activations_eps" not in CONFIG:
        return []
    args = [
        "--round-activations-eps",
        str(CONFIG["round_activations_eps"]),
        "--round-activations-site",
        CONFIG["round_activations_site"],
    ]
    if "round_activations_quantile" in CONFIG:
        args += [
            "--round-activations-quantile",
            str(CONFIG["round_activations_quantile"]),
        ]
    if "round_activations_quantiles" in CONFIG:
        args += [
            "--round-activations-quantiles",
            ",".join(map(str, CONFIG["round_activations_quantiles"])),
        ]
    return args


def config_mismatch(actual):
    return actual is None or (
        actual["epsilon"] != CONFIG["round_activations_eps"]
        or actual["site"] != CONFIG["round_activations_site"]
        or actual.get("quantile", 0.0) != CONFIG.get("round_activations_quantile", 0.0)
        or actual.get("quantiles") != CONFIG.get("round_activations_quantiles")
    )


def get_json(path):
    with urllib.request.urlopen(URL + path, timeout=120) as response:
        return json.load(response)


def wait_ready(alive=session_exists):
    for _ in range(600):
        if not alive():
            raise RuntimeError("Server exited; inspect server.log")
        try:
            get_json("/v1/models")
            if config_mismatch(get_json("/rounding/config")):
                raise RuntimeError("Rounding server configuration mismatch")
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("Server not ready within startup timeout")


def evaluate(limit, output=None):
    """Run the generative check (chat_eval.py) against the running server."""
    if CONFIG["task"] not in ("mmlu_stem_chat", "gsm8k_chat"):
        raise RuntimeError("Only generative evaluation is supported (ADR 0004)")
    status()
    if output is None:
        output = ARTIFACTS / ("chat-smoke" if limit else "chat")
        if CONFIG["task"] == "gsm8k_chat":
            output = ARTIFACTS / ("gsm8k-smoke" if limit else "gsm8k")
        output /= time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    actual = None
    try:
        actual = get_json("/rounding/config")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    if "round_activations_eps" in CONFIG and config_mismatch(actual):
        raise RuntimeError("Evaluation configuration differs from running server")
    if actual:
        (output / "rounding-config.json").write_text(json.dumps(actual, indent=2))
        (output / "rounding-before.json").write_text(
            json.dumps(get_json("/rounding/stats"), indent=2)
        )
    command = [
        CONFIG["python"],
        str(SCRIPT.with_name("chat_eval.py")),
        "run",
        "--config",
        str(output / "experiment.json"),
        "--output",
        str(output),
        "--url",
        URL,
    ]
    if limit:
        command += ["--limit", str(limit)]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = environment()
    if CONFIG["offline_evaluation"]:
        env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    (output / "experiment.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    print(f"Results: {output}", flush=True)
    subprocess.run(command, cwd=output, env=env, check=True)
    if actual:
        (output / "rounding-after.json").write_text(
            json.dumps(get_json("/rounding/stats"), indent=2)
        )
    return output


def job_root(name, limit):
    if limit:
        return ARTIFACTS / "jobs-smoke" / f"limit{limit}" / name
    return ARTIFACTS / "jobs" / name


def select(name):
    matches = [c for c in experiment_configs if c.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one experiment named {name!r}")
    CONFIG.clear()
    CONFIG.update(matches[0])


def job(name, limit):
    """MLSpace container entry point: serve one registry row, evaluate, stop."""
    select(name)
    root = job_root(name, limit)
    attempt = root / time.strftime("%Y%m%d-%H%M%S")
    attempt.mkdir(parents=True)
    env = environment()
    if CONFIG["offline_evaluation"]:
        env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    with (attempt / "server.log").open("w") as log:
        server = subprocess.Popen(
            server_command(attempt / "rounding.sqlite"),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=attempt,
            env=env,
        )
        try:
            wait_ready(lambda: server.poll() is None)
            task_dir = "gsm8k" if CONFIG["task"] == "gsm8k_chat" else "mmlu"
            output = evaluate(limit, attempt / task_dir)
        finally:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    # Written last: run_jobs.py treats this marker as the completed result.
    marker = {
        "name": name,
        "site": CONFIG["round_activations_site"],
        "epsilon": CONFIG["round_activations_eps"],
        "quantile": CONFIG["round_activations_quantile"],
        "quantiles": CONFIG.get("round_activations_quantiles"),
        "output": str(output),
    } | ({"limit": limit} if limit else {})
    (root / "done.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(f"Job complete: {root / 'done.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
            "serve",
            "eval",
            "job",
        ],
    )
    parser.add_argument("--limit", type=int, help="Smoke-test questions per subject")
    parser.add_argument(
        "--round-activations-eps", "--round_activations_eps", type=float
    )
    parser.add_argument(
        "--round-activations-quantile", "--round_activations_quantile", type=float
    )
    parser.add_argument(
        "--round-activations-quantiles",
        "--round_activations_quantiles",
        help="Comma-separated quantile per equal consecutive layer group",
    )
    parser.add_argument(
        # Same values as rounding_stats.SITES (not imported: launchers lack torch).
        "--round-activations-site",
        choices=["mlp", "block", "moe_out"],
        default="mlp",
    )
    parser.add_argument("--name", help="Registry experiment name for job or eval")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    eps, quantile = args.round_activations_eps, args.round_activations_quantile
    schedule = args.round_activations_quantiles
    if schedule is not None:
        try:
            schedule = [float(value) for value in schedule.split(",")]
        except ValueError:
            parser.error("quantile schedule must be comma-separated numbers")
    zeroing = (eps, quantile, schedule)
    if args.action == "job" and args.name is None:
        parser.error("--name is required for job")
    if args.name is not None and args.action not in ("job", "eval"):
        parser.error("--name only applies to job and eval")
    if args.name is not None and any(value is not None for value in zeroing):
        parser.error("--name takes its zeroing settings from the registry")
    if eps is not None and (not math.isfinite(eps) or eps < 0):
        parser.error("epsilon must be finite and nonnegative")
    for value in [quantile, *(schedule or [])]:
        if value is not None and not (math.isfinite(value) and 0 <= value < 1):
            parser.error("quantiles must be in [0, 1)")
    if sum(map(bool, zeroing)) > 1:
        parser.error("use one of epsilon, quantile, or quantile schedule")
    if any(value is not None for value in zeroing):
        CONFIG.update(
            round_activations_eps=eps or 0.0,
            round_activations_site=args.round_activations_site,
        )
        if quantile is not None:
            CONFIG["round_activations_quantile"] = quantile
        if schedule is not None:
            CONFIG["round_activations_quantiles"] = schedule
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.action == "restart":
        stop()
        start()
    elif args.action == "eval":
        if args.name:
            select(args.name)
        evaluate(args.limit)
    elif args.action == "job":
        job(args.name, args.limit)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
