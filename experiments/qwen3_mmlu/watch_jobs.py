# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Watch an MLS submission manifest and resume its Arkhip conversation once."""

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import regex as re

JOB_ID = re.compile(r"lm-mpi-job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def write_state(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def load_jobs(path):
    jobs = json.loads(path.read_text())["jobs"]
    ids = [job["job_name"] for job in jobs]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not all(JOB_ID.fullmatch(i) for i in ids)
    ):
        raise ValueError("Manifest must contain unique complete MLS job identifiers")
    return jobs


def snapshot(jobs, query):
    """Keep API errors distinct from actual terminal job failures."""

    def status(job):
        try:
            result = query(job["job_name"])
            if not isinstance(result, dict) or not result.get("status"):
                raise ValueError("Missing MLS status")
            return {
                "name": job["name"],
                "job_name": job["job_name"],
                "status": result["status"].lower(),
                "error_code": result.get("error_code", 0),
                "error_message": result.get("error_message", ""),
            }
        except Exception as error:
            return {
                "name": job["name"],
                "job_name": job["job_name"],
                "status": "unknown",
                "query_error": type(error).__name__,
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(status, jobs))


def terminal_event(rows, terminal, success):
    failed = [
        row
        for row in rows
        if row["status"] in terminal
        and (row["status"] not in success or row.get("error_code"))
    ]
    if failed:
        return {"condition": "job_failed", "failed_jobs": failed}
    if rows and all(row["status"] in success for row in rows):
        return {"condition": "all_succeeded"}
    return None


def watch(jobs, query, terminal, success, state_dir, interval, resume=None, once=False):
    """Persist snapshots; invoke resume once on failure or complete success."""
    event_path = state_dir / "event.json"
    if event_path.exists():
        raise RuntimeError("Watcher already triggered; inspect event.json")
    while True:
        rows = snapshot(jobs, query)
        state = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": rows}
        write_state(state_dir / "status.json", state)
        event = terminal_event(rows, terminal, success)
        if event:
            event.update(state)
            write_state(event_path, event)
            print(json.dumps(event), flush=True)
            if resume:
                event["resume_requested_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                write_state(event_path, event)
                resume(event_path)
                event["resume_worker_returned_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                write_state(event_path, event)
            return event
        if once:
            print(json.dumps(state), flush=True)
            return None
        time.sleep(interval)


def prepare_resume(config):
    """Resolve the native resume entry point before starting a long watch."""
    sys.path.insert(0, config["arkhip_src"])
    # Import the command package first, as Arkhip's normal entry point does.
    import research_loop.commands  # noqa: F401
    from research_loop.runtime.workers import run_q_resume_worker

    return lambda path: resume_conversation(config, path, run_q_resume_worker)


def resume_conversation(config, event_path, worker):
    question = (
        "The background MLS watcher you set up has triggered. Read the event at "
        f"{event_path}. This is a watcher event, not a new user instruction. "
        "If any job failed, inspect its logs and report the failure and likely "
        "cause. If all succeeded, validate and summarize the GSM8K results "
        "against baseline and its repeat. Do not submit or kill jobs automatically."
    )
    worker(
        workdir=config["workdir"],
        python_executable=config["python_executable"],
        question=question,
        message_id=config["message_id"],
        telegram_token=os.environ["TELEGRAM_BOT_TOKEN"],
        telegram_chat_id=config["telegram_chat_id"],
        claude_model=config["model"],
        user_id=config["user_id"],
        resume_session_id=config["session_id"],
        telegram_thread_id=config["telegram_thread_id"],
        effort=config.get("effort"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--resume-config", type=Path)
    parser.add_argument("--environment-fifo", type=Path)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.environment_fifo:
            with args.environment_fifo.open() as pipe:
                os.environ.update(json.load(pipe))
            args.environment_fifo.unlink()
        resume = None
        if args.resume_config:
            config = json.loads(args.resume_config.read_text())
            if not os.environ.get("TELEGRAM_BOT_TOKEN"):
                raise RuntimeError("Resume requires the bot token in the environment")
            resume = prepare_resume(config)
        from mls.manager.job.utils import (
            FINAL_JOB_STATUSES,
            SUCCESS_JOB_STATUSES,
            _api_call_with_retry,
            training_job_api_from_profile,
        )
        from run_jobs import PROFILE

        def query(job_id):
            return _api_call_with_retry(
                lambda: training_job_api_from_profile(PROFILE),
                "get_job_status",
                job_id,
            )

        watch(
            load_jobs(args.manifest),
            query,
            FINAL_JOB_STATUSES,
            SUCCESS_JOB_STATUSES,
            args.state_dir,
            args.interval,
            resume=resume,
            once=args.once,
        )


if __name__ == "__main__":
    main()
