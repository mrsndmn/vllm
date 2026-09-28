"""Behavior checks for threshold boundaries and isolated API statistics."""

import asyncio
import json
import subprocess

import pytest
import torch
from rounding_api import RoundingMiddleware
from rounding_stats import connect, read_stats, threshold, validate_eps

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
    assert evaluated == [c for c in run.experiment_configs if c != done]
    entries = json.loads(manifest.read_text())
    assert entries[0]["output"] == str(tmp_path / "old")
    stored = [
        json.loads((run.Path(e["output"]) / "experiment.json").read_text())
        for e in entries
    ]
    assert sorted(map(json.dumps, stored)) == sorted(
        map(json.dumps, run.experiment_configs)
    )
    run.sweep(None)
    assert len(evaluated) == len(run.experiment_configs) - 1


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
