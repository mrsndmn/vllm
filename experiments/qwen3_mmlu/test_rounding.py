"""Behavior checks for threshold boundaries and isolated API statistics."""

import asyncio
import json
import subprocess

import chat_eval
import pytest
import run_jobs
import torch
from rounding_api import RoundingMiddleware
from rounding_stats import (
    connect,
    read_stats,
    threshold,
    validate_eps,
    validate_quantile,
)

import run


def test_threshold_counts_new_zeros_and_affected_tokens():
    values = torch.tensor([[0, 0.125, -0.125, 0.25], [0, 0, 1, -1]])
    counts = threshold(values, 0.125)
    assert values.tolist() == [[0, 0, 0, 0.25], [0, 0, 1, -1]]
    assert counts.tolist() == [[4, 2, 1, 1, 1, 0], [4, 0, 2, 1, 0, 0]]


def test_zero_epsilon_is_bitwise_noop_and_nonfinite_values_are_retained():
    values = torch.tensor([[0.0, -0.0, 1e-5, float("nan"), float("inf")]])
    original = values.view(torch.int32).clone()
    threshold(values, 0)
    assert torch.equal(values.view(torch.int32), original)
    counts = threshold(values, 1e-4)
    assert counts.tolist() == [[5, 1, 2, 1, 1, 0]]
    assert torch.isnan(values[0, 3]) and torch.isinf(values[0, 4])
    threshold(values, 1e300)
    assert torch.isnan(values[0, 3]) and torch.isinf(values[0, 4])


def test_block_threshold_preserves_unselected_residual_operands():
    hidden = torch.tensor([[1.0, 1.0, 0.5]], dtype=torch.bfloat16)
    residual = torch.tensor([[-0.99609375, 0.0078125, -0.5]], dtype=torch.bfloat16)
    counts = threshold(hidden, 0.004, residual)
    assert counts.tolist() == [[3, 1, 1, 1, 1, 0]]
    assert hidden.tolist() == [[0, 1.0, 0.5]]
    assert residual.tolist() == [[0, 0.0078125, -0.5]]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(value):
    with pytest.raises(ValueError):
        validate_eps(value)


def test_request_counts_are_isolated_and_aggregate_across_chunks(tmp_path):
    path = tmp_path / "stats.sqlite"
    with connect(path) as db:
        db.executemany(
            "INSERT INTO requests VALUES (?, ?, ?, ?)",
            [
                ("a-0", "a", "[[8, 2, 1, 2, 1, 0]]", 4),
                ("a-1", "a", "[[4, 1, 0, 1, 1, 0]]", 0),
                ("b-0", "b", "[[100, 0, 0, 25, 0, 0]]", 0),
            ],
        )
    result = read_stats(path, "mlp", 0.1, "a")
    assert result["engine_sequences"] == 2
    assert result["initial_cached_tokens_excluded"] == 4
    assert result["layers"][0]["newly_zeroed_float_percent"] == 25
    assert result["layers"][0]["tokens"] == 3
    assert read_stats(path, "mlp", 0.1, "missing")["layers"] == []


@pytest.mark.parametrize("streaming", [False, True])
def test_api_attaches_statistics_after_fragmented_response(
    tmp_path, monkeypatch, streaming
):
    path = tmp_path / "stats.sqlite"
    connect(path).close()
    monkeypatch.setenv("ROUND_ACTIVATIONS_DB", str(path))
    monkeypatch.setenv("ROUND_ACTIVATIONS_SITE", "mlp")
    monkeypatch.setenv("ROUND_ACTIVATIONS_EPS", "0.1")
    messages = []

    async def app(scope, receive, send):
        uid = dict(scope["headers"])[b"x-request-id"].decode()
        response_id = "cmpl-" + uid
        with connect(path) as db:
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?)",
                (response_id + "-0", response_id, "[[4, 1, 0, 1, 1, 0]]", 0),
            )
        content_type = b"text/event-stream" if streaming else b"application/json"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", content_type), (b"content-length", b"0")],
            }
        )
        body = (
            b'data: {"choices": []}\n\ndata: [DONE]\n\n'
            if streaming
            else json.dumps({"id": response_id, "choices": []}).encode()
        )
        for byte in body[:-1]:
            await send(
                {"type": "http.response.body", "body": bytes([byte]), "more_body": True}
            )
        await send({"type": "http.response.body", "body": body[-1:]})

    async def send(message):
        messages.append(message)

    asyncio.run(
        RoundingMiddleware(app)(
            {"type": "http", "path": "/v1/completions", "headers": []}, None, send
        )
    )
    body = b"".join(message.get("body", b"") for message in messages)
    if streaming:
        events = body.split(b"\n\n")
        result = json.loads(events[-3][len(b"data: ") :])
        assert events[-2] == b"data: [DONE]"
    else:
        result = json.loads(body)
    assert result["rounding_statistics"]["layers"][0]["newly_zeroed_floats"] == 1
    assert not messages[-1].get("more_body", False)


def fake_sweep(tmp_path, monkeypatch):
    evaluated = []

    def evaluate(limit):
        output = tmp_path / f"run-{len(evaluated)}"
        output.mkdir()
        (output / "experiment.json").write_text(json.dumps(run.CONFIG))
        evaluated.append(dict(run.CONFIG))
        return output

    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    for name in ("stop", "start", "wait_ready"):
        monkeypatch.setattr(run, name, lambda: None)
    monkeypatch.setattr(run, "evaluate", evaluate)
    return evaluated


def record(tmp_path, name, config, limit=None):
    output = tmp_path / name
    output.mkdir()
    (output / "experiment.json").write_text(json.dumps(config))
    entry = {
        "site": config["round_activations_site"],
        "epsilon": config["round_activations_eps"],
        "output": str(output),
    }
    return entry | ({"limit": limit} if limit else {})


def test_sweep_resumes_only_unrecorded_configs(tmp_path, monkeypatch):
    evaluated = fake_sweep(tmp_path, monkeypatch)
    done = run.experiment_configs[1]
    manifest = tmp_path / "rounding-sweep.json"
    manifest.write_text(json.dumps([record(tmp_path, "old", done)]))
    run.sweep(None)
    assert done not in evaluated
    local = [c for c in run.experiment_configs if c.get("launcher") != "mls"]
    assert evaluated == [c for c in local if c != done]
    entries = json.loads(manifest.read_text())
    assert entries[0]["output"] == str(tmp_path / "old")
    stored = [
        json.loads((run.Path(e["output"]) / "experiment.json").read_text())
        for e in entries
    ]
    assert sorted(map(json.dumps, stored)) == sorted(map(json.dumps, local))
    run.sweep(None)
    assert len(evaluated) == len(local) - 1


@pytest.mark.parametrize("mismatch", ["config", "limit"])
def test_sweep_rejects_foreign_manifest_entries(tmp_path, monkeypatch, mismatch):
    evaluated = fake_sweep(tmp_path, monkeypatch)
    config = dict(run.experiment_configs[0])
    if mismatch == "config":
        config["round_activations_eps"] = 0.5
    entry = record(tmp_path, "old", config, limit=1 if mismatch == "config" else 2)
    (tmp_path / "rounding-smoke.json").write_text(json.dumps([entry]))
    with pytest.raises(RuntimeError, match="unconfigured"):
        run.sweep(1)
    assert evaluated == []


def test_server_session_checks_ignore_prefixed_sweep_session(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path))
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG, session="srv"))
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "srv-sweep", "sleep 600"], check=True
    )
    try:
        assert run.session_exists("srv-sweep")
        assert not run.session_exists()
        run.stop()
        assert run.session_exists("srv-sweep")
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", "srv", "sleep 600"], check=True
        )
        run.stop()
        assert not run.session_exists()
        assert run.session_exists("srv-sweep")
    finally:
        subprocess.run(["tmux", "kill-server"], check=False)


def test_quantile_zeroes_each_row_up_to_its_own_cutoff():
    rows = torch.tensor([[4.0, -1, 3, 2, 8, -7, 6, 5], [0.5, 0.25, 9, 9, 9, 9, 9, 9]])
    alone = rows[:1].clone()
    counts = threshold(rows, 0, quantile=0.25)
    threshold(alone, 0, quantile=0.25)
    assert rows[0].tolist() == [4, 0, 3, 0, 8, -7, 6, 5]
    assert rows[1].tolist() == [0, 0, 9, 9, 9, 9, 9, 9]
    assert torch.equal(alone, rows[:1])
    assert counts[:, 1].tolist() == [2, 2]


def test_quantile_zeroes_ties_and_keeps_nonfinite_values():
    values = torch.tensor([[1.0, -1, 1, 5], [0, 0, 3, 4], [1, 2, float("nan"), 7]])
    counts = threshold(values, 0, quantile=0.5)
    assert values[0].tolist() == [0, 0, 0, 5]
    assert values[1].tolist() == [0, 0, 3, 4]
    assert values[2, :2].tolist() == [0, 0] and values[2, 2].isnan()
    assert counts[:, 1].tolist() == [3, 0, 2]
    assert counts[:, 2].tolist() == [0, 2, 0]
    nan_rows = torch.tensor([[float("nan"), float("nan"), float("nan"), 2.0]])
    threshold(nan_rows, 0, quantile=0.75)
    assert nan_rows[0, 3] == 0 and nan_rows[0, :3].isnan().all()


def test_zero_quantile_is_bitwise_noop_and_block_zeroes_both_operands():
    values = torch.tensor([[0.0, -0.0, 1e-5, float("nan")]])
    original = values.view(torch.int32).clone()
    threshold(values, 0, quantile=0.0)
    assert torch.equal(values.view(torch.int32), original)
    hidden = torch.tensor([[1.0, 1.0, 0.5, 4.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[-0.99609375, 2.0, -0.5, 1.0]], dtype=torch.bfloat16)
    counts = threshold(hidden, 0, residual, quantile=0.5)
    # Position 2 sums to zero already, so its operands keep their bit patterns.
    assert hidden.tolist() == [[0, 1.0, 0.5, 4.0]]
    assert residual.tolist() == [[0, 2.0, -0.5, 1.0]]
    assert counts.tolist() == [[4, 1, 1, 1, 1, 0]]
    assert threshold(torch.empty(0, 4), 0, quantile=0.5).shape == (0, 6)


@pytest.mark.parametrize("value", [-0.1, 1, float("nan")])
def test_invalid_quantile_is_rejected(value):
    with pytest.raises(ValueError):
        validate_quantile(value)


def test_registry_keeps_fixed_eps_rows_and_adds_named_quantile_jobs():
    jobs = [c for c in run_jobs.job_configs() if c["task"] == "mmlu"]
    assert [c["name"] for c in jobs] == [
        "mlp_q0.25",
        "mlp_q0.5",
        "block_q0.25",
        "block_q0.5",
    ]
    for config in jobs:
        assert config["round_activations_eps"] == 0
        assert config["python"] == run.JOB_PYTHON
    fixed = [c for c in run.experiment_configs if c.get("launcher") != "mls"]
    assert all(
        set(c)
        == set(run.BASE_CONFIG) | {"round_activations_site", "round_activations_eps"}
        for c in fixed
    )


class FakeServer:
    def __init__(self, command, **kwargs):
        self.command = command
        self.signals = []

    def poll(self):
        return None

    def send_signal(self, value):
        self.signals.append(value)

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize("fail", [False, True])
def test_job_stops_server_and_writes_marker_last(tmp_path, monkeypatch, fail):
    servers = []

    def popen(command, **kwargs):
        servers.append(FakeServer(command, **kwargs))
        return servers[-1]

    def evaluate(limit, output, workdir):
        assert not (run.job_root("mlp_q0.5", 1) / "done.json").exists()
        if fail:
            raise RuntimeError("evaluation failed")
        return output

    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    monkeypatch.setattr(run.subprocess, "Popen", popen)
    monkeypatch.setattr(run, "wait_ready", lambda alive: alive())
    monkeypatch.setattr(run, "evaluate", evaluate)
    marker = run.job_root("mlp_q0.5", 1) / "done.json"
    assert marker.parent != run.job_root("mlp_q0.5", 2)
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            run.job("mlp_q0.5", 1)
        assert not marker.exists()
    else:
        run.job("mlp_q0.5", 1)
        done = json.loads(marker.read_text())
        assert done["quantile"] == 0.5 and done["site"] == "mlp"
        assert done["limit"] == 1 and done["output"].endswith("/mmlu")
    command = servers[0].command
    assert command[0] == run.JOB_PYTHON
    assert command[command.index("--round-activations-quantile") + 1] == "0.5"
    assert servers[0].signals == [run.signal.SIGINT]


def fake_mls(monkeypatch, tmp_path, queued=()):
    submitted = []
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run_jobs, "STAGING_ROOT", tmp_path / "staging")
    monkeypatch.setattr(run_jobs, "committed_head", lambda: "abc123")
    monkeypatch.setattr(run_jobs, "stage", lambda commit: None)

    def submit(body):
        submitted.append(body)
        return {"job_name": f"lm-mpi-job-{len(submitted)}"}

    queued = {run_jobs.normalize(desc) for desc in queued}
    monkeypatch.setattr(run_jobs, "connect", lambda: ("SR004", queued, submit))
    return submitted


def launch_args(**kwargs):
    defaults = dict(dry=False, force=False, filter=None, limit=None)
    return run_jobs.argparse.Namespace(**(defaults | kwargs))


def test_launcher_skips_done_and_queued_and_dry_submits_nothing(tmp_path, monkeypatch):
    queued = [run_jobs.job_desc(run_jobs.job_configs("block_q0.5")[0], None)]
    submitted = fake_mls(monkeypatch, tmp_path, [queued[0] + " #other @someone"])
    done = run.job_root("mlp_q0.25", None)
    done.mkdir(parents=True)
    (done / "done.json").write_text("{}")
    assert run_jobs.launch(launch_args(dry=True)) == [] and submitted == []
    launched = run_jobs.launch(launch_args())
    chat = [c["name"] for c in run_jobs.job_configs() if c["task"] != "mmlu"]
    assert [job["name"] for job in launched] == ["mlp_q0.5", "block_q0.25", *chat]
    body = submitted[0]
    assert body["instance_type"] == "a100.1gpu" and body["region"] == "SR004"
    workdir = tmp_path / "staging" / "abc123" / run_jobs.SUBDIR
    assert body["script"] == (
        f"cd {workdir} && {run.JOB_PYTHON} run.py job --name mlp_q0.5"
    )
    smoke = run_jobs.launch(launch_args(filter="mlp_q0.25", limit=1))
    assert submitted[-1]["script"].endswith("--name mlp_q0.25 --limit 1")
    assert "smoke1" in smoke[0]["desc"]
    forced = run_jobs.launch(launch_args(filter="mlp_q0.25", force=True))
    assert [job["name"] for job in forced] == ["mlp_q0.25"]
    assert [c["name"] for c in run_jobs.job_configs("chat_*_q0.5")] == [
        "chat_mlp_q0.5",
        "chat_block_q0.5",
    ]


def test_launcher_refuses_dirty_experiment_tree(monkeypatch):
    monkeypatch.setattr(run_jobs, "git", lambda *args: " M experiments/x.py\n")
    with pytest.raises(SystemExit, match="Commit or stash"):
        run_jobs.committed_head()


def test_stage_extracts_only_the_committed_experiment_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(run_jobs, "STAGING_ROOT", tmp_path)
    commit = run_jobs.git("rev-parse", "HEAD").strip()
    workdir = run_jobs.stage(commit)
    assert workdir == tmp_path / commit / run_jobs.SUBDIR
    assert (workdir / "run.py").exists()
    assert [p.name for p in (tmp_path / commit).iterdir()] == ["experiments"]
    assert run_jobs.stage(commit) == workdir
    assert [p.name for p in tmp_path.iterdir()] == [commit]

    def racing_rename(src, dst):  # another launcher finished staging first
        run_jobs.shutil.copytree(src, dst)
        raise OSError(39, "Directory not empty", dst)

    monkeypatch.setattr(run_jobs.os, "rename", racing_rename)
    other = run_jobs.git("rev-parse", "HEAD~1").strip()
    assert run_jobs.stage(other) == tmp_path / other / run_jobs.SUBDIR
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([commit, other])


def test_summary_reports_total_zero_share_and_rejects_empty_manifest(
    tmp_path, monkeypatch
):
    import summarize_rounding

    manifest = tmp_path / "empty.json"
    manifest.write_text("[]")
    monkeypatch.setattr("sys.argv", ["summarize", "--manifest", str(manifest)])
    with pytest.raises(SystemExit, match="No completed runs"):
        summarize_rounding.main()


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("Short reasoning.\nAnswer: B", "B"),
        ("**Answer:** (C)", "C"),
        ("Answer: A is tempting.\nFinal answer: D", "D"),
        ("answer: a simple case, so Answer: C.", "C"),
        ("The answer depends on B.", None),
        ("The answer: A is wrong.\nAnswer: $C$", "C"),
        ("Answer: \\boxed{D}", "D"),
        ("Answer: `B`", "B"),
        ("Answer: 'A'", "A"),
        ("Answer: _B_", "B"),
        ("Answer: B.\nThe answer: I am not sure", None),
    ],
)
def test_final_answer_takes_last_answer_line(text, letter):
    assert chat_eval.final_answer(text) == letter


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        (" B", "B"),
        (" **C**", "C"),
        (" (A)", "A"),
        (" $B$", "B"),
        (" \\boxed{C}", "C"),
        (" The", None),
        (" Both", None),
    ],
)
def test_forced_answer_reads_leading_letter(text, letter):
    assert chat_eval.forced_answer(text) == letter


def fake_post(first_text, finish, forced=" B"):
    calls = []

    def post(body):
        calls.append(body)
        text, reason = (first_text, finish) if len(calls) == 1 else (forced, "length")
        return {
            "choices": [{"text": text, "finish_reason": reason}],
            "usage": {"completion_tokens": 7},
        }

    return post, calls


CHAT = dict(run.CHAT_CONFIG)
QUESTION = {
    "subject": "s",
    "index": 3,
    "question": "Q",
    "choices": list("wxyz"),
    "answer": "B",
}


def test_chat_answer_natural_and_forced_modes():
    post, calls = fake_post("Reason.\nAnswer: B", "stop")
    record = chat_eval.answer(QUESTION, "<p>", post, CHAT)
    assert (record["mode"], record["prediction"], record["correct"]) == (
        "natural",
        "B",
        True,
    )
    assert len(calls) == 1 and calls[0]["max_tokens"] == 1024
    assert calls[0]["temperature"] == 0 and calls[0]["prompt"] == "<p>"
    post, calls = fake_post("Reasoning that ran out", "length", forced=" **B**")
    record = chat_eval.answer(QUESTION, "<p>", post, CHAT)
    assert record["mode"] == "forced_cap" and record["prediction"] == "B"
    assert calls[1]["prompt"] == (
        "<p>Reasoning that ran out\n\n[reasoning interrupted] Answer:"
    )
    assert calls[1]["max_tokens"] == 5
    post, calls = fake_post("Answer: C but more text follows", "length")
    assert chat_eval.answer(QUESTION, "<p>", post, CHAT)["mode"] == "forced_cap"
    post, calls = fake_post("No letter given.", "stop", forced=" I think")
    record = chat_eval.answer(QUESTION, "<p>", post, CHAT)
    assert record["mode"] == "forced_missing"
    assert calls[1]["prompt"].endswith("No letter given.\n\nAnswer:")
    assert record["prediction"] is None and record["correct"] is False


def test_chat_prompt_uses_template_without_thinking():
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(messages=messages, **kwargs)
            return "templated"

    assert chat_eval.chat_prompt(Tokenizer(), QUESTION) == "templated"
    content = seen["messages"][0]["content"]
    assert seen["enable_thinking"] is False and seen["add_generation_prompt"]
    assert "keep your reasoning short" in content and "\nA) w\nB) x" in content


def test_question_sample_is_seeded_per_subject():
    data = {
        f"s{i}": [
            {"question": f"q{i}-{j} ", "choices": [1, 2, 3, 4], "answer": j % 4}
            for j in range(40)
        ]
        for i in range(3)
    }
    config = dict(CHAT, chat_subjects=["s0", "s1", "s2"])
    first = chat_eval.select_questions(config, 5, data.__getitem__)
    again = chat_eval.select_questions(
        dict(config, chat_subjects=["s2", "s0", "s1"]), 5, data.__getitem__
    )
    assert len(first) == 15 and sorted(map(str, first)) == sorted(map(str, again))
    for row in first:
        assert row["question"] == f"q{row['subject'][1]}-{row['index']}"
        assert row["answer"] == "ABCD"[row["index"] % 4]
    other = chat_eval.select_questions(dict(config, chat_seed=1), 5, data.__getitem__)
    assert [r["index"] for r in other] != [r["index"] for r in first]


def test_chat_registry_rows_and_collect_filter(tmp_path, monkeypatch):
    chat = [c for c in run.experiment_configs if c.get("task") == "mmlu_stem_chat"]
    assert [c["name"] for c in chat] == [
        "chat_baseline",
        "chat_baseline_repeat",
        "chat_mlp_q0.25",
        "chat_mlp_q0.5",
        "chat_block_q0.25",
        "chat_block_q0.5",
    ]
    for config in chat:
        assert len(config["chat_subjects"]) * config["chat_per_subject"] == 304
        assert config["launcher"] == "mls" and config["max_num_seqs"] == 16
    assert chat[0]["round_activations_quantile"] == 0
    assert {k: v for k, v in chat[1].items() if k != "name"} == {
        k: v for k, v in chat[0].items() if k != "name"
    }
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run_jobs, "MANIFEST", tmp_path / "manifest.json")
    for name in ("mlp_q0.5", "chat_mlp_q0.5"):
        run.job_root(name, None).mkdir(parents=True)
        (run.job_root(name, None) / "done.json").write_text(json.dumps({"n": name}))
    run_jobs.collect()
    assert json.loads((tmp_path / "manifest.json").read_text()) == [{"n": "mlp_q0.5"}]


def test_evaluate_routes_chat_rows_to_chat_eval(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    run.select("chat_block_q0.5")
    served = {"site": "block", "epsilon": 0.0, "quantile": 0.5}
    monkeypatch.setattr(run, "status", lambda: None)
    monkeypatch.setattr(
        run, "get_json", lambda path: served if "config" in path else {"layers": []}
    )
    monkeypatch.setattr(
        run.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    output = run.evaluate(2, tmp_path / "out", tmp_path / "runtime")
    command = commands[0]
    assert command[:3] == [
        run.JOB_PYTHON,
        str(run.SCRIPT.with_name("chat_eval.py")),
        "run",
    ]
    assert command[command.index("--limit") + 1] == "2"
    stored = json.loads((output / "experiment.json").read_text())
    assert stored["name"] == "chat_block_q0.5"
    assert (output / "rounding-after.json").exists()


def write_chat_run(root, name, predictions):
    output = root / "jobs" / name / "attempt" / "mmlu"
    output.mkdir(parents=True)
    records = [
        {
            "subject": subject,
            "index": 0,
            "answer": "A",
            "prediction": prediction,
            "correct": prediction == "A",
            "mode": "natural",
            "reasoning_tokens": 5,
        }
        for subject, prediction in zip(("s0", "s1"), predictions)
    ]
    lines = "".join(json.dumps(record) + "\n" for record in records)
    (output / "samples.jsonl").write_text(lines)
    config = dict(CHAT, chat_subjects=["s0", "s1"])
    summary = chat_eval.summarize_records(records, config)
    (output / "results.json").write_text(json.dumps(summary))
    layer = {"floats": 100, "newly_zeroed_floats": 50, "already_zero_floats": 1}
    (output / "rounding-before.json").write_text(json.dumps({"layers": []}))
    (output / "rounding-after.json").write_text(json.dumps({"layers": [layer]}))
    marker = root / "jobs" / name / "done.json"
    marker.write_text(json.dumps({"output": str(output)}))


def test_chat_summary_validates_runs_and_counts_changes(tmp_path, monkeypatch):
    small = [
        dict(c, chat_subjects=["s0", "s1"], chat_per_subject=1)
        for c in run.experiment_configs
        if c.get("task") == "mmlu_stem_chat"
    ]
    monkeypatch.setattr(run, "experiment_configs", small)
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    write_chat_run(tmp_path, "chat_baseline", ["A", "A"])
    write_chat_run(tmp_path, "chat_block_q0.5", ["A", None])
    rows = {row["experiment"]: row for row in chat_eval.summarize()}
    assert rows["chat_baseline"]["accuracy_percent"] == 100
    worse = rows["chat_block_q0.5"]
    assert worse["accuracy_percent"] == 50 and worse["unparsed"] == 1
    assert (worse["changed_vs_baseline"], worse["broken_vs_baseline"]) == (1, 1)
    assert worse["min_layer_total_zero_percent"] == 51
    assert (tmp_path / "rounding-chat-sweep-summary.csv").exists()
    write_chat_run(tmp_path, "chat_mlp_q0.5", ["A"])
    with pytest.raises(AssertionError):
        chat_eval.summarize()


def test_poster_retries_transient_failures(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": 1}'

    def urlopen(request, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise chat_eval.urllib.error.URLError("reset")
        return Response()

    monkeypatch.setattr(chat_eval.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(chat_eval.time, "sleep", lambda seconds: None)
    assert chat_eval.poster("http://x")({"a": 1}) == {"ok": 1} and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(
        chat_eval.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(ConnectionError()),
    )
    with pytest.raises(ConnectionError):
        chat_eval.poster("http://x")({"a": 1})
