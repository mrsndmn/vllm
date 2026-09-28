"""Submit MLSpace jobs for run.py registry rows with launcher="mls".

Each job serves one row inside its container (`run.py job`) and evaluates MMLU.
Run with the environment that provides the shared `mls` client, e.g.
/home/jovyan/.mlspace/envs/arkhip/bin/python experiments/qwen3_mmlu/run_jobs.py --dry
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import run

SUBDIR = "experiments/qwen3_mmlu"
STAGING_ROOT = run.ARTIFACTS / "staging"
MANIFEST = run.ARTIFACTS / "rounding-quantile-sweep.json"
PROFILE = "default"
QUEUE = "fusionbrainlab-job"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
TAGS = "#ARKHIP #rnd #notify_completed @mrsndmn"
REPO = Path(__file__).resolve().parents[2]


def job_configs(pattern=None):
    return [
        config
        for config in run.experiment_configs
        if config.get("launcher") == "mls"
        and (not pattern or pattern in config["name"])
    ]


def job_desc(config, limit):
    smoke = f" smoke{limit}" if limit else ""
    return f"qwen3-mmlu zeroing {config['name']}{smoke} {TAGS}"


def normalize(desc):
    return " ".join(re.sub(r"[#@]\S+", "", desc).split())


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def committed_head():
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", SUBDIR)
    if dirty:
        raise SystemExit(f"Commit or stash tracked changes first:\n{dirty}")
    return git("rev-parse", "HEAD").strip()


def stage(commit):
    """Materialize the committed experiment directory once per commit."""
    target = STAGING_ROOT / commit
    if not target.exists():
        tmp = STAGING_ROOT / f"{commit}.tmp.{os.getpid()}"
        tmp.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "-C", str(REPO), "archive", commit, SUBDIR], stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", str(tmp)], stdin=archive.stdout, check=True)
        if archive.wait():
            raise RuntimeError("git archive failed")
        try:
            os.rename(tmp, target)
        except OSError:  # A concurrent launch staged the same commit first.
            if not target.exists():
                raise
            shutil.rmtree(tmp)
    return target / SUBDIR


def payload(config, workdir, limit, region):
    command = [config["python"], "run.py", "job", "--name", config["name"]]
    if limit:
        command += ["--limit", str(limit)]
    return {
        "script": f"cd {shlex.quote(str(workdir))} && {shlex.join(command)}",
        "job_desc": job_desc(config, limit),
        "env_variables": {"HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"},
        "instance_type": config["instance_type"],
        "queue_name": QUEUE,
        "region": region,
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": BASE_IMAGE,
        "n_workers": 1,
        "processes_per_worker": 1,
    }


def connect():
    """Return (region, normalized in-progress descriptions, submit function)."""
    from mls.manager.job.utils import (
        get_in_progress_jobs,
        run_job_with_retry,
        training_job_api_from_profile,
    )

    client, extra = training_job_api_from_profile(PROFILE)
    queued = {normalize(job.get("job_desc", "")) for job in get_in_progress_jobs()}

    def submit(body):
        return run_job_with_retry(client, body, profile=PROFILE)

    return extra["region"], queued, submit


def collect():
    """Write a summarize_rounding.py manifest from completed full-run markers."""
    entries = []
    for config in job_configs():
        marker = run.job_root(config["name"], None) / "done.json"
        if marker.exists():
            entries.append(json.loads(marker.read_text()))
    MANIFEST.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"{len(entries)}/{len(job_configs())} completed: {MANIFEST}")


def launch(args):
    configs = job_configs(args.filter)
    if not configs:
        raise SystemExit("No MLSpace experiments match")
    commit = committed_head()
    region, queued, submit = connect()
    workdir = STAGING_ROOT / commit / SUBDIR
    launched = []
    for config in configs:
        desc = job_desc(config, args.limit)
        marker = run.job_root(config["name"], args.limit) / "done.json"
        if marker.exists() and not args.force:
            print(f"Skip (done): {config['name']} ({marker})")
            continue
        if normalize(desc) in queued:
            print(f"Skip (queued or running): {desc}")
            continue
        body = payload(config, workdir, args.limit, region)
        if args.dry:
            print(json.dumps(body, indent=2))
            continue
        stage(commit)
        result = submit(body)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if not job_name:
            raise RuntimeError(f"Submission failed for {config['name']}: {result}")
        print(f"Submitted {config['name']}: {job_name}")
        launched.append({"name": config["name"], "job_name": job_name, "desc": desc})
    print("__QWEN3_MMLU_JOBS_JSON__")
    print(json.dumps({"commit": commit, "dry": args.dry, "jobs": launched}))
    return launched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="Print; submit nothing")
    parser.add_argument("--force", action="store_true", help="Ignore done markers")
    parser.add_argument("--filter", help="Substring of experiment names")
    parser.add_argument("--limit", type=int, help="Smoke-test items per subject")
    parser.add_argument(
        "--collect", action="store_true", help="Write the completed-runs manifest"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.collect:
        collect()
    else:
        launch(args)


if __name__ == "__main__":
    main()
