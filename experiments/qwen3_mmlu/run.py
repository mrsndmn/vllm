"""Local persistent Qwen3 serving and a reproducible MMLU evaluation."""

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
import urllib.request
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
ARTIFACTS = Path(CONFIG["artifacts"])
URL = f"http://127.0.0.1:{CONFIG['port']}"
SCRIPT = Path(__file__).resolve()


def session_exists():
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", CONFIG["session"]],
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
        probe.bind(("127.0.0.1", CONFIG["port"]))
    command = shlex.join([CONFIG["python"], str(SCRIPT), "serve"])
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


def evaluate(limit):
    status()
    output = ARTIFACTS / ("smoke" if limit else "mmlu") / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
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
        ],
    )
    parser.add_argument("--limit", type=int, help="Smoke-test items per subject")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.action == "restart":
        stop()
        start()
    elif args.action == "eval":
        evaluate(args.limit)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
