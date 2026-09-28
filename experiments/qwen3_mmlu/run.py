"""Local persistent Qwen3 serving and a reproducible MMLU evaluation."""

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

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
        "max_num_seqs": 8,
        "max_num_batched_tokens": 1024,
        "task": "mmlu",
        "fewshot": 5,
        "offline_evaluation": True,
    }
]
CONFIG = experiment_configs[0]
BASE_CONFIG = CONFIG.copy()
experiment_configs = [
    dict(BASE_CONFIG, round_activations_site=site, round_activations_eps=eps)
    for site, epsilons in (
        ("mlp", (0.0, 1e-4, 1e-3, 1e-2, 1e-1)),
        ("block", (1e-2, 1e-1)),
    )
    for eps in epsilons
]
ARTIFACTS = Path(CONFIG["artifacts"])
URL = f"http://127.0.0.1:{CONFIG['port']}"
SCRIPT = Path(__file__).resolve()


def session_exists(session=None):
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session or CONFIG["session"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


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


def serve():
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
        database = ARTIFACTS / "rounding" / f"{uuid.uuid4().hex}.sqlite"
        command[1:3] = [
            str(SCRIPT.with_name("rounding_server.py")),
            *rounding_args(),
            "--rounding-db",
            str(database),
        ]
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
            CONFIG["session"],
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
    return [
        "--round-activations-eps",
        str(CONFIG["round_activations_eps"]),
        "--round-activations-site",
        CONFIG["round_activations_site"],
    ]


def get_json(path):
    with urllib.request.urlopen(URL + path, timeout=120) as response:
        return json.load(response)


def wait_ready():
    for _ in range(600):
        if not session_exists():
            raise RuntimeError("Server exited; inspect server.log")
        try:
            get_json("/v1/models")
            actual = get_json("/rounding/config")
            if (
                actual["epsilon"] != CONFIG["round_activations_eps"]
                or actual["site"] != CONFIG["round_activations_site"]
            ):
                raise RuntimeError("Rounding server configuration mismatch")
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("Server not ready within startup timeout")


def evaluate(limit):
    status()
    output = ARTIFACTS / ("smoke" if limit else "mmlu") / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    actual = None
    try:
        actual = get_json("/rounding/config")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    if "round_activations_eps" in CONFIG and (
        actual is None
        or actual["epsilon"] != CONFIG["round_activations_eps"]
        or actual["site"] != CONFIG["round_activations_site"]
    ):
        raise RuntimeError("Evaluation configuration differs from running server")
    if actual:
        (output / "rounding-config.json").write_text(json.dumps(actual, indent=2))
        (output / "rounding-before.json").write_text(
            json.dumps(get_json("/rounding/stats"), indent=2)
        )
    model_args = ",".join(
        [
            f"model={CONFIG['model']}",
            f"base_url={URL}/v1/completions",
            f"tokenizer={CONFIG['snapshot']}",
            "tokenizer_backend=huggingface",
            "tokenized_requests=True",
            "num_concurrent=8",
            "max_retries=3",
            f"max_length={CONFIG['max_model_len']}",
        ]
    )
    command = [
        CONFIG["python"],
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        CONFIG["task"],
        "--num_fewshot",
        str(CONFIG["fewshot"]),
        "--batch_size",
        "1",
        "--seed",
        "0,1234,1234,1234",
        "--output_path",
        str(output),
        "--log_samples",
    ]
    if limit:
        command += ["--limit", str(limit)]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = environment()
    if CONFIG["offline_evaluation"]:
        env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    (output / "experiment.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    print(f"Results: {output}", flush=True)
    # The harness treats a local directory named "mmlu" as a task definition.
    workdir = ARTIFACTS / "runtime"
    workdir.mkdir(exist_ok=True)
    subprocess.run(command, cwd=workdir, env=env, check=True)
    if actual:
        (output / "rounding-after.json").write_text(
            json.dumps(get_json("/rounding/stats"), indent=2)
        )
    return output


def sweep(limit):
    manifest = ARTIFACTS / ("rounding-smoke.json" if limit else "rounding-sweep.json")
    # Resume: configurations already recorded for this limit are not rerun.
    results = json.loads(manifest.read_text()) if manifest.exists() else []
    completed = []
    for entry in results:
        stored = json.loads((Path(entry["output"]) / "experiment.json").read_text())
        if entry.get("limit") != limit or stored not in experiment_configs:
            raise RuntimeError(
                f"{manifest} records an unconfigured run: {entry['output']}"
            )
        completed.append(stored)
    for config in experiment_configs:
        if config in completed:
            continue
        CONFIG.clear()
        CONFIG.update(config)
        stop()
        start()
        wait_ready()
        output = evaluate(limit)
        results.append(
            {
                "site": config["round_activations_site"],
                "epsilon": config["round_activations_eps"],
                "output": str(output),
            }
            | ({"limit": limit} if limit else {})
        )
        completed.append(config)
        manifest.write_text(json.dumps(results, indent=2) + "\n")
        print(
            f"Completed {len(completed)}/{len(experiment_configs)}: {output}",
            flush=True,
        )
    print(f"Sweep complete: {manifest}", flush=True)


def launch_sweep(limit):
    """Run the sweep in its own tmux session so it outlives this shell."""
    session = CONFIG["session"] + "-sweep"
    if session_exists(session):
        raise RuntimeError(f"Sweep session already exists: {session}")
    log = ARTIFACTS / ("rounding-smoke.log" if limit else "rounding-sweep.log")
    command = [CONFIG["python"], str(SCRIPT), "sweep"]
    if limit:
        command += ["--limit", str(limit)]
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(ARTIFACTS),
            f"{shlex.join(command)} >> {shlex.quote(str(log))} 2>&1",
        ],
        check=True,
    )
    print(f"Started {session}; see {log}")


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
            "sweep",
        ],
    )
    parser.add_argument("--limit", type=int, help="Smoke-test items per subject")
    parser.add_argument(
        "--round-activations-eps", "--round_activations_eps", type=float
    )
    parser.add_argument(
        "--round-activations-site", choices=["mlp", "block"], default="mlp"
    )
    parser.add_argument(
        "--detach", action="store_true", help="Run sweep in a tmux session"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.detach and args.action != "sweep":
        parser.error("--detach only applies to sweep")
    if args.round_activations_eps is not None:
        import math

        if (
            not math.isfinite(args.round_activations_eps)
            or args.round_activations_eps < 0
        ):
            parser.error("epsilon must be finite and nonnegative")
        CONFIG.update(
            round_activations_eps=args.round_activations_eps,
            round_activations_site=args.round_activations_site,
        )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.action == "restart":
        stop()
        start()
    elif args.action == "eval":
        evaluate(args.limit)
    elif args.action == "sweep" and args.detach:
        launch_sweep(args.limit)
    elif args.action == "sweep":
        sweep(args.limit)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
