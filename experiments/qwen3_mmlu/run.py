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
experiment_configs += [
    dict(
        GSM8K_CONFIG,
        name=f"gsm8k_{site}_q{quantile:g}",
        round_activations_site=site,
        round_activations_quantile=quantile,
    )
    for site, quantile in (
        ("mlp", 0.5),
        ("mlp", 0.8),
        ("moe_out", 0.4),
        ("block", 0.25),
    )
]
experiment_configs.append(
    dict(
        GSM8K_CONFIG,
        name="gsm8k_adapt_mlp_b0.8",
        round_activations_quantiles=list(ADAPTIVE_SCHEDULES[("mlp", 0.8)]),
    )
)
BENCHMARK_CONFIGS = {
    "math500": dict(
        dataset="HuggingFaceH4/MATH-500",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        split="test",
        expected_questions=500,
    ),
    "ifeval": dict(
        dataset="google/IFEval",
        revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
        split="train",
        expected_questions=541,
        evaluator_revision="e49bbfe381c9c0e564b937f1c4e163a2273c65cc",
        nltk_revision="550b6625bcef1f2abff2ff770a5a0d272c9c6b2a",
    ),
    "mmlu_pro": dict(
        dataset="TIGER-Lab/MMLU-Pro",
        revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        split="test",
        expected_questions=12032,
    ),
    "gpqa": dict(
        dataset="idavidrein/gpqa",
        revision="56686c06f5e19865c153de0fdb11be3890014df7",
        split="gpqa_diamond",
        expected_questions=198,
        archive_password="deserted-untie-orchid",
    ),
    "humaneval_plus": dict(
        dataset="humaneval",
        revision="v0.1.10",
        expected_questions=164,
    ),
    "mbpp_plus": dict(
        dataset="mbpp",
        revision="v0.2.0",
        expected_questions=378,
    ),
}
RULER_CONFIG = dict(
    dataset="NVIDIA/RULER",
    revision="c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
    source_sha256="e8fb7531a2c16c7d3cec4c6b1c58958adecca365a8a08a7d052f6058b04400bc",
    samples_per_task=100,
    preparation_workers=4,
    random_seed=42,
    template_reserve=64,
    expected_questions=1300,
    subtasks=[
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_1",
        "qa_2",
    ],
    corpus_urls={
        "english_words.json": "https://media.githubusercontent.com/media/NVIDIA/RULER/"
        "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/"
        "scripts/data/synthetic/json/english_words.json",
        "squad.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "hotpotqa.json": "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
        "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
    },
)
RULER_CORPUS_SHA256 = {
    "english_words.json": (
        "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca"
    ),
    "squad.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    "hotpotqa.json": "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    "PaulGrahamEssays.json": (
        "8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd"
    ),
}
RULER_PACKAGES = {
    "wonderwords": "2.2.0",
    "html2text": "2025.4.15",
    "beautifulsoup4": "4.13.5",
}
for context_length in (8192, 16384, 32768):
    BENCHMARK_CONFIGS[f"ruler_{context_length // 1024}k"] = dict(
        RULER_CONFIG, context_length=context_length
    )
BENCHMARK_PACKAGES = {
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "evalplus": "0.3.1",
    "langdetect": "1.0.9",
    "nltk": "3.10.3",
    "tree-sitter": "0.26.0",
    "tree-sitter-python": "0.25.0",
}
CODE_GRADER = dict(
    revision="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
    sha256="3c833c39b842e33f251c83db4347e0a95191909f23b390c12d73ab29a28a4daf",
    path=str(
        Path(BASE_CONFIG["artifacts"])
        / "runtime"
        / "evalplus-26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
        / "evalplus_grader.py"
    ),
)
for benchmark, settings in BENCHMARK_CONFIGS.items():
    for arm in [c for c in experiment_configs if c.get("task") == "gsm8k_chat"]:
        experiment_configs.append(
            dict(
                {
                    key: value
                    for key, value in arm.items()
                    if not key.startswith("chat_")
                    or key in ("chat_concurrency", "chat_seed")
                },
                name="suite_" + benchmark + arm["name"].removeprefix("gsm8k"),
                task=benchmark,
                benchmark=settings,
                benchmark_packages=BENCHMARK_PACKAGES
                | (RULER_PACKAGES if benchmark.startswith("ruler_") else {}),
                benchmark_data=str(
                    Path(BASE_CONFIG["artifacts"]) / "benchmarks" / benchmark
                ),
                generation_max_tokens=4096,
                grading_workers=4,
                code_min_time_limit=1.0,
                code_time_limit_factor=4.0,
                code_grader=CODE_GRADER
                if benchmark in ("humaneval_plus", "mbpp_plus")
                else None,
                offline_evaluation=True,
            )
        )
# Weight zeroing arms inherit the exact baseline protocol for each benchmark.
WEIGHT_BASELINES = [
    row
    for row in experiment_configs
    if row.get("name", "").endswith("_baseline")
    and not row["name"].startswith("prof_")
    and row["task"] != "mmlu_pro"
]
experiment_configs += [
    dict(
        baseline,
        name=baseline["name"].removesuffix("baseline")
        + f"weight_{scope}_q{quantile:g}",
        weight_zeroing=dict(scope=scope, quantile=quantile, batch_rows=256),
    )
    for baseline in WEIGHT_BASELINES
    for scope in ("embedding", "both")
    for quantile in (
        (0.5, 0.7, 0.9, 0.95, 0.97, 0.99) if scope == "embedding" else (0.5, 0.7, 0.9)
    )
]
# Signed-epsilon sweep inherits each existing generative baseline protocol.
EPSILON_TASKS = (
    "mmlu_stem_chat",
    "gsm8k_chat",
    "math500",
    "gpqa",
    "ifeval",
    "humaneval_plus",
    "mbpp_plus",
)
EPSILON_BASELINES = [
    row
    for row in experiment_configs
    if row["name"].endswith("_baseline")
    and row["task"] in EPSILON_TASKS
    and not row["name"].startswith("prof_")
]
experiment_configs += [
    dict(
        baseline,
        name=baseline["name"].removesuffix("baseline") + f"mlp_signed_epsilon_q{q:g}",
        round_activations_quantile=q,
        round_activations_replacement="signed_epsilon_normal",
    )
    for baseline in EPSILON_BASELINES
    for q in (0.5, 0.8)
] + [
    dict(
        baseline,
        name=baseline["name"].removesuffix("baseline") + f"mlp_weights_{mode}_q{q:g}",
        mlp_weight_rounding=dict(quantile=q, replacement=mode, batch_rows=256),
    )
    for baseline in EPSILON_BASELINES
    for mode in ("zero", "signed_epsilon_normal")
    for q in (0.25, 0.5)
]


NORM_CHANNEL_MASK = dict(
    first_layer=4,
    last_layer=44,
    threshold=1.0,
    expected_kept=604,
    mask_sha256="444ac2a54fe9a671be9ad4d541be6c5b509409e04a436288179ee70c9efec6a5",
)
NORM_CHANNEL_MASKS = {
    604: NORM_CHANNEL_MASK,
    1980: dict(
        NORM_CHANNEL_MASK,
        threshold=0.2,
        expected_kept=1980,
        mask_sha256="60f2a02616bdf16eb53ca6f1ef031facc67f9888d591358109c796f16365da38",
    ),
    1828: dict(
        NORM_CHANNEL_MASK,
        threshold=0.4,
        expected_kept=1828,
        mask_sha256="03d2ef25128991deae28e17cfa4e674f2457f23e8a2a2802789328a170094965",
    ),
    1539: dict(
        NORM_CHANNEL_MASK,
        threshold=0.7,
        expected_kept=1539,
        mask_sha256="bb7ac7dc16ef5eb1a2ee3febbed3100d2928a73ce403c5945769f9c1898addf7",
    ),
}
for task, baseline_name, limit in (
    ("mmlu", "chat_baseline", 4),
    ("gsm8k", "gsm8k_baseline", 64),
):
    baseline = next(c for c in experiment_configs if c["name"] == baseline_name)
    for kept in (None, *NORM_CHANNEL_MASKS):
        arm = "baseline" if kept is None else f"keep{kept}"
        experiment_configs.append(
            dict(
                baseline,
                name=f"normkeep_preview_{task}_{arm}",
                evaluation_limit=limit,
                **(
                    {"post_norm_channels": NORM_CHANNEL_MASKS[kept].copy()}
                    if kept is not None
                    else {}
                ),
            )
        )

for task in ("mmlu", "gsm8k"):
    base = next(
        c
        for c in experiment_configs
        if c["name"] == f"normkeep_preview_{task}_keep1980"
    )
    for correction, rescale, preserve in (
        ("rms", True, False),
        ("router", False, True),
        ("rms_router", True, True),
    ):
        experiment_configs.append(
            dict(
                base,
                name=base["name"] + "_" + correction,
                post_norm_channels=dict(
                    base["post_norm_channels"],
                    rescale_rms=rescale,
                    preserve_routing=preserve,
                ),
            )
        )

for row in experiment_configs:
    if row["task"].startswith("ruler_"):
        row.update(
            max_model_len=row["benchmark"]["context_length"],
            memory_fraction=0.9,
            max_num_seqs=2,
            chat_concurrency=2,
            generation_max_tokens=128,
        )
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
    if CONFIG.get("round_activations_replacement"):
        args += [
            "--round-activations-replacement",
            CONFIG["round_activations_replacement"],
        ]
    if CONFIG.get("mlp_weight_rounding"):
        args += ["--mlp-weight-rounding", json.dumps(CONFIG["mlp_weight_rounding"])]
    if CONFIG.get("weight_zeroing"):
        args += ["--weight-zeroing", json.dumps(CONFIG["weight_zeroing"])]
    if CONFIG.get("post_norm_channels"):
        args += ["--post-norm-channels", json.dumps(CONFIG["post_norm_channels"])]
    return args


def config_mismatch(actual):
    return actual is None or (
        actual["epsilon"] != CONFIG["round_activations_eps"]
        or actual["site"] != CONFIG["round_activations_site"]
        or actual.get("quantile", 0.0) != CONFIG.get("round_activations_quantile", 0.0)
        or actual.get("quantiles") != CONFIG.get("round_activations_quantiles")
        or actual.get("replacement", "zero")
        != CONFIG.get("round_activations_replacement", "zero")
        or actual.get("mlp_weight_rounding") != CONFIG.get("mlp_weight_rounding")
        or actual.get("weight_zeroing") != CONFIG.get("weight_zeroing")
        or actual.get("post_norm_channels") != CONFIG.get("post_norm_channels")
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
    """Run the configured generative benchmark against the matching server."""
    if limit is None:
        limit = CONFIG.get("evaluation_limit")
    extended = CONFIG["task"] in BENCHMARK_CONFIGS
    if CONFIG["task"] not in ("mmlu_stem_chat", "gsm8k_chat") and not extended:
        raise RuntimeError("Only generative evaluation is supported (ADR 0004)")
    status()
    if output is None:
        output = ARTIFACTS / ("chat-smoke" if limit else "chat")
        if CONFIG["task"] == "gsm8k_chat":
            output = ARTIFACTS / ("gsm8k-smoke" if limit else "gsm8k")
        elif extended:
            output = ARTIFACTS / (CONFIG["task"] + ("-smoke" if limit else ""))
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
        str(SCRIPT.with_name("benchmark_eval.py" if extended else "chat_eval.py")),
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
    evaluation_limit = limit if limit is not None else CONFIG.get("evaluation_limit")
    root = job_root(name, limit)
    attempt = root / time.strftime("%Y%m%d-%H%M%S")
    attempt.mkdir(parents=True)
    env = environment()
    if CONFIG["offline_evaluation"]:
        env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    smoke_output = None
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
            task_dir = (
                CONFIG["task"]
                if CONFIG["task"] in BENCHMARK_CONFIGS
                else ("gsm8k" if CONFIG["task"] == "gsm8k_chat" else "mmlu")
            )
            if CONFIG["task"].startswith("ruler_") and limit is None:
                smoke_output = evaluate(
                    len(CONFIG["benchmark"]["subtasks"]), attempt / "ruler-smoke"
                )
            output = evaluate(evaluation_limit, attempt / task_dir)
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
    marker["replacement"] = CONFIG.get("round_activations_replacement", "zero")
    if evaluation_limit is not None:
        marker["evaluation_limit"] = evaluation_limit
    if CONFIG.get("post_norm_channels"):
        marker["post_norm_channels"] = CONFIG["post_norm_channels"]
    if CONFIG.get("mlp_weight_rounding"):
        marker["mlp_weight_rounding"] = CONFIG["mlp_weight_rounding"]
    if smoke_output is not None:
        marker["smoke_output"] = str(smoke_output)
    if CONFIG.get("weight_zeroing"):
        marker["weight_zeroing"] = CONFIG["weight_zeroing"]
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
