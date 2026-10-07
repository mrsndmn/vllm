# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Behavior checks for threshold boundaries and isolated API statistics."""

import asyncio
import itertools
import json
import math
import shutil
import subprocess
import sys

import chat_eval
import compression
import hidden_viz
import pytest
import run_jobs
import sensitivity
import torch
import weight_viz
from rounding_api import RoundingMiddleware
from rounding_stats import (
    connect,
    group_quantile,
    layer_operands,
    read_stats,
    threshold,
    validate_eps,
    validate_quantile,
    validate_schedule,
)

import run


def test_compression_reservoir_keeps_uniform_priorities_and_total_routed_count():
    rows = torch.arange(60).reshape(20, 3).bfloat16()
    sample = compression.Reservoir(5, 42)
    sample.add(rows[:7])
    sample.add(rows[7:])
    expected = (
        torch.rand(20, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
        .topk(5)
        .indices
    )
    assert sample.seen == 20 and len(sample.rows) == 5
    assert torch.equal(sample.rows, rows[expected])


def test_compression_capture_matches_actual_hf_expert_routing():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    config = Qwen3MoeConfig(
        hidden_size=16, moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2
    )
    module = Qwen3MoeSparseMoeBlock(config).eval()
    samples = [compression.Reservoir(24, e) for e in range(4)]
    actual = {}

    def hook(expert):
        def save(module, args):
            actual[expert] = args[0].clone()

        return save

    handles = [module.register_forward_pre_hook(compression.capture_hook(samples))]
    handles += [
        expert.register_forward_pre_hook(hook(e))
        for e, expert in enumerate(module.experts)
    ]
    with torch.inference_mode():
        module(torch.randn(1, 12, 16))
    for handle in handles:
        handle.remove()
    assert sum(s.seen for s in samples) == 24
    for e, rows in actual.items():
        assert samples[e].seen == len(rows)
        sampled = samples[e].rows[samples[e].rows[:, 0].argsort()]
        actual_sorted = rows[rows[:, 0].argsort()]
        assert torch.equal(sampled, actual_sorted)


def test_compression_storage_budgets_include_csr_indices_and_factor_values():
    config = compression.experiment_configs[0]
    dense, budget, rank, nnz = compression.storage_budget((64, 96), 0.5, config)
    assert dense == 64 * 96 * 2
    assert rank * (64 + 96) * 2 <= budget
    assert nnz * 6 + 65 * 4 <= budget < (nnz + 1) * 6 + 65 * 4
    weight = torch.ones(64, 96)
    compressed = compression.prune(weight, weight, nnz, per_output=True)
    assert int(compressed.count_nonzero()) == nnz
    per_row = compressed.count_nonzero(dim=1)
    assert int(per_row.max() - per_row.min()) <= 1


def test_compression_whitening_preserves_high_energy_input_direction():
    weight = torch.diag(torch.tensor([10.0, 1.0]))
    calibration = torch.diag(torch.tensor([0.01, 100.0]))
    ordinary, whitened = compression.svd_basis(weight, calibration, 1e-4)
    reference = calibration @ weight.T
    plain_error = compression.errors(
        reference, compression.apply(compression.factors(ordinary, 1), calibration)
    )["relative_frobenius"]
    aware_error = compression.errors(
        reference, compression.apply(compression.factors(whitened, 1), calibration)
    )["relative_frobenius"]
    assert plain_error > 0.99 and aware_error < 0.02
    full = compression.apply(compression.factors(ordinary, 2), calibration)
    assert compression.errors(reference, full)["relative_frobenius"] < 0.02


def test_compression_whitening_handles_rank_deficient_outlier_activations():
    generator = torch.Generator().manual_seed(73)
    direction = torch.randn(1, 2048, generator=generator)
    calibration = torch.tensor([[1.0], [10000.0]]) @ direction
    weight = torch.randn(4, 2048, generator=generator)
    _, whitened = compression.svd_basis(weight, calibration, 1e-4)
    reference = calibration @ weight.T
    replay = compression.apply(compression.factors(whitened, 4), calibration)
    assert torch.isfinite(replay).all()
    assert compression.errors(reference, replay)["relative_frobenius"] < 0.02


def test_compression_expert_replay_evaluates_nonlinearity_and_down_projection():
    weights = {f: torch.eye(2) for f in ["gate_proj", "up_proj", "down_proj"]}
    x = torch.tensor([[1.0, -1.0], [2.0, 3.0]])
    reference = compression.expert_output(weights, x)
    expected = torch.nn.functional.silu(x) * x
    assert torch.equal(reference, expected)
    changed = compression.expert_output(weights, x, "gate_proj", torch.zeros(2, 2))
    assert torch.equal(changed, torch.zeros_like(reference))
    assert compression.errors(reference, changed)["relative_frobenius"] == 1


def test_compression_job_payload_uses_analysis_entrypoint_and_absolute_output():
    config = compression.experiment_configs[0]
    jobs = run_jobs.job_configs("*compression*", analysis=True)
    assert jobs == [config]
    payload = run_jobs.payload(
        config, run_jobs.STAGING_ROOT / "commit" / run_jobs.SUBDIR, None, "SR004"
    )
    assert "compression.py all --name expert_compression_screen" in payload["script"]
    assert config["python"] in payload["script"]
    assert "run.py job" not in payload["script"]
    assert str(run_jobs.STAGING_ROOT) in payload["script"]
    assert run_jobs.job_desc(config, None).startswith("qwen3-mmlu compression ")


def test_compression_pipeline_captures_and_screens_a_tiny_checkpoint(
    tmp_path, monkeypatch
):
    import numpy as np
    from safetensors.torch import save_file
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
        )
    ).eval()
    weights = model.state_dict()
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in weights}})
    )
    records = [
        [name, 0, int(name.split(".")[5]), "expert." + name.split(".")[6]]
        for name in weights
        if ".experts." in name
    ]
    np.savez(
        tmp_path / "prior.npz",
        records=json.dumps(records),
        stats=np.ones((len(records), len(weight_viz.STATS))),
    )
    config = dict(
        compression.experiment_configs[0],
        snapshot=str(tmp_path),
        weight_stats=str(tmp_path),
        layers=[0],
        sequence_length=8,
        min_rows={"calibration": 1, "heldout": 1},
        max_rows={"calibration": 16, "heldout": 16},
        storage_fractions=[0.5],
    )
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(
        compression,
        "corpus_tokens",
        lambda c, s, t: (
            list(
                range(1 if s == "calibration" else 17, 17 if s == "calibration" else 33)
            ),
            s,
        ),
    )
    metadata = compression.capture(config, "cpu", tmp_path)
    assert metadata["splits"]["calibration"]["tokens"] == 16
    assert metadata["splits"]["heldout"]["tokens"] == 16
    assert compression.capture(config, "cpu", tmp_path) == metadata
    with torch.inference_mode():
        rows = compression.screen(config, "cpu", tmp_path)
    assert {r["method"] for r in rows} == set(config["methods"])
    assert {r["family"] for r in rows} == set(config["families"])
    assert all(r["stored_bytes"] <= r["dense_bytes"] * 0.5 for r in rows)
    assert all(math.isfinite(r["expert_relative_frobenius"]) for r in rows)
    compression.plot(config, tmp_path)
    assert (tmp_path / "compression-error.png").exists()
    with pytest.raises(ValueError, match="Stale screening"):
        compression.plot(dict(config, covariance_damping=0.01), tmp_path)
    path = tmp_path / "results.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Stale screening"):
        compression.plot(config, tmp_path)


def test_compression_launcher_rejects_stale_completion(tmp_path):
    config = compression.experiment_configs[0]
    marker = tmp_path / "done.json"
    assert not run_jobs.complete(config, marker)
    marker.write_text(json.dumps({"fingerprint": compression.fingerprint(config)}))
    assert run_jobs.complete(config, marker)
    with pytest.raises(SystemExit, match="Stale completed"):
        run_jobs.complete(dict(config, covariance_damping=0.01), marker)


def test_weight_report_moments_and_stable_rank_match_known_matrices():
    matrices = torch.stack(
        [torch.diag(torch.tensor([3.0, 1.0])), torch.eye(2), torch.zeros(2, 2)]
    )
    stats, linear, log, _ = weight_viz.summarize(matrices, weight_viz.CONFIG, 42)
    values = dict(zip(weight_viz.STATS, stats.T))
    assert values["frobenius"] == pytest.approx([10**0.5, 2**0.5, 0])
    assert values["sigma_max_est"] == pytest.approx([3, 1, 0])
    assert values["stable_rank_est"] == pytest.approx([10 / 9, 2, 0])
    assert values["mean"] == pytest.approx([1, 0.5, 0])
    assert values["zero_share"] == pytest.approx([0.5, 0.5, 1])
    assert values["std"][0] == pytest.approx((1.5) ** 0.5)
    assert values["row_norm_cv"][0] == pytest.approx(0.5)
    assert values["abs_q50_est"][0] == pytest.approx(0.5)
    assert linear.sum(1).tolist() == [4, 4, 4]
    assert log.sum((1, 2)).tolist() == [4, 4, 4]
    assert not torch.isfinite(torch.tensor(values["excess_kurtosis"][2]))


def test_weight_histograms_cover_tails_zero_and_sign_without_vector_rank():
    vectors = torch.tensor([[-100.0, -0.5, -1e-8, 0, 1e-8, 0.5, 100]])
    stats, linear, log, reach = weight_viz.summarize(vectors, weight_viz.CONFIG, 42)
    assert reach == weight_viz.CONFIG["norm_reach"]
    assert linear[0, 0] == linear[0, -1] == 1
    assert log[0, 0].sum() == 3 and log[0, 1].sum() == 4
    assert log[0, 0, 0] == 1 and log[0, 1, 0] == 2
    assert log[0, 0, -1] == log[0, 1, -1] == 1
    assert stats[0, 6] == pytest.approx(1 / 7)
    assert all(not torch.isfinite(torch.tensor(v)) for v in stats[0, 14:])
    with pytest.raises(ValueError, match="non-finite"):
        weight_viz.summarize(torch.tensor([[float("nan")]]), weight_viz.CONFIG, 42)


def test_weight_checkpoint_scan_covers_each_expert_and_resumes(tmp_path):
    from safetensors.torch import save_file

    names = [
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
        "model.norm.weight",
    ]
    weights = {
        names[0]: torch.eye(2),
        names[1]: torch.ones(2, 2),
        names[2]: torch.ones(2),
    }
    save_file(weights, tmp_path / "test.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "test.safetensors" for n in names}})
    )
    config = dict(weight_viz.CONFIG, snapshot=str(tmp_path))
    first = weight_viz.collect(config, "cpu", tmp_path)
    # Cache-only reuse must retain the CPU origin even with a CUDA request.
    second = weight_viz.collect(config, "cuda:7", tmp_path)
    assert {r[0] for r in first[0]} == set(names)
    assert {tuple(r[1:4]) for r in first[0]} == {
        (0, 0, "expert.down_proj"),
        (0, 1, "expert.down_proj"),
        (-1, -1, "norm"),
    }
    assert first[0] == second[0]
    for a, b in zip(first[1:4], second[1:4]):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), equal_nan=True)
    assert first[4] == second[4]
    origin = second[4]["test.safetensors"]
    assert origin["device"] == "cpu" and origin["torch"] == torch.__version__
    assert len(origin["source_sha256"]) == 64
    with pytest.raises(ValueError, match="Stale analysis cache"):
        weight_viz.collect(dict(config, seed=43), "cpu", tmp_path)


def test_weight_single_file_checkpoint_covers_dense_layers_and_global_weights(tmp_path):
    from safetensors.torch import save_file

    weights = {
        "model.layers.0.mlp.down_proj.weight": torch.eye(2),
        "model.embed_tokens.weight": torch.ones(3, 2),
        "model.norm.weight": torch.ones(2),
    }
    save_file(weights, tmp_path / "model.safetensors")
    config = dict(weight_viz.CONFIG, snapshot=str(tmp_path))
    records, stats, linear, log, analysis = weight_viz.collect(config, "cpu", tmp_path)
    assert {r[0] for r in records} == set(weights)
    assert all(r[2] == -1 for r in records)
    assert sum(math.prod(r[4]) for r in records) == 12
    assert linear.sum() == log.sum() == 12
    assert list(analysis) == ["model.safetensors"]


def test_weight_report_embeds_offline_data_and_enforces_actual_byte_limit():
    import gzip

    import pybase64 as base64

    stats, linear, log, reach = weight_viz.summarize(
        torch.eye(2)[None], weight_viz.CONFIG, 42
    )
    text = "</script><script>alert(1)</script>"
    records = [[text, 0, -1, "self_attn.q_proj", [2, 2], reach]]
    html = weight_viz.render(records, stats, linear, log, weight_viz.CONFIG, {})
    assert "%%DATA%%" not in html and "%%PLOTLY%%" not in html
    assert "<script src=" not in html and text not in html
    payload = html.split('<script id="payload" type="application/octet-stream">')[1]
    data = json.loads(gzip.decompress(base64.b64decode(payload.split("</script>")[0])))
    assert data["records"][0][0] == text
    with pytest.raises(ValueError, match="above"):
        weight_viz.render(
            records,
            stats,
            linear,
            log,
            dict(weight_viz.CONFIG, max_html_bytes=len(html.encode()) - 1),
            {},
        )


def test_weight_viewer_layer_parsing_and_expert_pooling_use_observable_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    lib = (
        weight_viz.TEMPLATE.read_text()
        .split('<script id="viz-lib">')[1]
        .split("</script>")[0]
    )
    script = (
        lib
        + """
const fails = text => {try {parseLayers(text, 48);return false;} catch {return true;}};
console.log(JSON.stringify({
  layers: parseLayers('0, 2-4, 2', 48),
  errors: ['', '48', '4-2', '-1', '0,,1'].map(fails),
  mean: meanMetric([0,1,2], [2,NaN,4], 1, 0),
  pooled: pooled([0,1], [1,2,3,4], 2),
  linear: linearPoints([1,2,3,4], 1),
  log: logPoints([1,2,3], [.1, 1]),
}));
"""
    )
    result = json.loads(
        subprocess.run(
            [node, "-e", script], capture_output=True, text=True, check=True
        ).stdout
    )
    assert result["layers"] == [0, 2, 3, 4] and all(result["errors"])
    assert result["mean"] == 3 and result["pooled"] == [4, 6]
    assert [p["x"] for p in result["linear"]] == [-1.5, -0.5, 0.5, 1.5]
    assert result["linear"][0]["lo"] is None
    assert result["linear"][-1]["hi"] is None
    assert result["log"][0]["lo"] == 0
    assert result["log"][-1]["hi"] is None


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


def test_server_session_checks_ignore_prefixed_sessions(tmp_path, monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path))
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG, session="srv"))

    def other_alive():
        command = ["tmux", "has-session", "-t", "=srv-other"]
        return subprocess.run(command, capture_output=True).returncode == 0

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "srv-other", "sleep 600"], check=True
    )
    try:
        assert not run.session_exists()
        run.stop()
        assert other_alive()
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", "srv", "sleep 600"], check=True
        )
        assert run.session_exists()
        run.stop()
        assert not run.session_exists()
        assert other_alive()
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


def test_registry_is_generative_only():
    assert run.CONFIG["task"] == run.BASE_CONFIG["task"] == chat_eval.TASK
    assert "fewshot" not in run.BASE_CONFIG
    assert (
        dict(
            run.BASE_CONFIG,
            launcher="mls",
            instance_type="a100.1gpu",
            python=run.JOB_PYTHON,
        )
        == run.CHAT_CONFIG
    )
    names = [c["name"] for c in run.experiment_configs]
    assert len(names) == len(set(names)) == len(run_jobs.job_configs())
    for config in run.experiment_configs:
        assert config["name"].startswith(("chat_", "prof_"))
        assert {k: config[k] for k in run.CHAT_CONFIG} == run.CHAT_CONFIG


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

    def evaluate(limit, output):
        assert not (run.job_root("chat_mlp_q0.5", 1) / "done.json").exists()
        if fail:
            raise RuntimeError("evaluation failed")
        return output

    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    monkeypatch.setattr(run.subprocess, "Popen", popen)
    monkeypatch.setattr(run, "wait_ready", lambda alive: alive())
    monkeypatch.setattr(run, "evaluate", evaluate)
    marker = run.job_root("chat_mlp_q0.5", 1) / "done.json"
    assert marker.parent != run.job_root("chat_mlp_q0.5", 2)
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            run.job("chat_mlp_q0.5", 1)
        assert not marker.exists()
    else:
        run.job("chat_mlp_q0.5", 1)
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
    queued = [run_jobs.job_desc(run_jobs.job_configs("chat_block_q0.5")[0], None)]
    submitted = fake_mls(monkeypatch, tmp_path, [queued[0] + " #other @someone"])
    done = run.job_root("chat_mlp_q0.25", None)
    done.mkdir(parents=True)
    (done / "done.json").write_text("{}")
    assert run_jobs.launch(launch_args(dry=True)) == [] and submitted == []
    launched = run_jobs.launch(launch_args())
    skipped = {"chat_mlp_q0.25", "chat_block_q0.5"}
    pending = [c["name"] for c in run_jobs.job_configs() if c["name"] not in skipped]
    assert [job["name"] for job in launched] == pending
    body = submitted[0]
    assert body["instance_type"] == "a100.1gpu" and body["region"] == "SR004"
    workdir = tmp_path / "staging" / "abc123" / run_jobs.SUBDIR
    assert body["script"] == (
        f"cd {workdir} && {run.JOB_PYTHON} run.py job --name chat_baseline"
    )
    smoke = run_jobs.launch(launch_args(filter="chat_mlp_q0.25", limit=1))
    assert submitted[-1]["script"].endswith("--name chat_mlp_q0.25 --limit 1")
    assert "smoke1" in smoke[0]["desc"]
    forced = run_jobs.launch(launch_args(filter="chat_mlp_q0.25", force=True))
    assert [job["name"] for job in forced] == ["chat_mlp_q0.25"]
    assert [c["name"] for c in run_jobs.job_configs("chat_*_q0.5")] == [
        "chat_mlp_q0.5",
        "chat_block_q0.5",
        "chat_moe_out_q0.5",
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


def test_chat_registry_rows():
    chat = [
        c
        for c in run.experiment_configs
        if c.get("task") == "mmlu_stem_chat"
        and c["name"].startswith("chat_")
        and not c["name"].startswith(("chat_adapt_", "chat_moe_out_"))
        and "_eps" not in c["name"]
    ]
    assert [c["name"] for c in chat] == [
        "chat_baseline",
        "chat_baseline_repeat",
        "chat_mlp_q0.25",
        "chat_mlp_q0.5",
        "chat_block_q0.25",
        "chat_block_q0.5",
        "chat_mlp_q0.6",
        "chat_mlp_q0.7",
        "chat_mlp_q0.8",
        "chat_block_q0.3",
        "chat_block_q0.4",
        "chat_mlp_q0.9",
        "chat_mlp_q0.95",
        "chat_mlp_q0.99",
    ]
    for config in chat:
        assert len(config["chat_subjects"]) * config["chat_per_subject"] == 304
        assert config["launcher"] == "mls" and config["max_num_seqs"] == 16
    assert chat[0]["round_activations_quantile"] == 0
    assert {k: v for k, v in chat[1].items() if k != "name"} == {
        k: v for k, v in chat[0].items() if k != "name"
    }


def test_generative_eps_and_moe_out_rows():
    rows = {
        c["name"]: c
        for c in run.experiment_configs
        if "_eps" in c.get("name", "") or c.get("name", "").startswith("chat_moe_out")
    }
    eps_names = [
        f"chat_{site}_eps{eps}"
        for site in ("mlp", "block", "moe_out")
        for eps in ("0.01", "0.1", "0.5")
    ]
    quantiles = ("0.4", "0.5", "0.7", "0.8", "0.9", "0.95", "0.99")
    q_names = [f"chat_moe_out_q{q}" for q in quantiles]
    assert list(rows) == eps_names + q_names
    baseline = next(
        c for c in run.experiment_configs if c.get("name") == "chat_baseline"
    )
    zeroing = ("name", "round_activations_site", "round_activations_eps")
    zeroing += ("round_activations_quantile",)
    for name, config in rows.items():
        # Same protocol and job settings as chat_baseline; only zeroing differs.
        assert {k: v for k, v in config.items() if k not in zeroing} == {
            k: v for k, v in baseline.items() if k not in zeroing
        }
        assert "round_activations_quantiles" not in config
        if "_eps" in name:
            assert config["round_activations_quantile"] == 0.0
            assert config["round_activations_eps"] == float(name.split("_eps")[1])
        else:
            assert config["round_activations_eps"] == 0.0
            assert config["round_activations_quantile"] == float(name.split("_q")[1])
    moe_out = [c for c in rows.values() if c["round_activations_site"] == "moe_out"]
    assert len(moe_out) == 10


def test_moe_out_zeroes_the_moe_output_and_keeps_the_residual_bits():
    hidden = torch.tensor([[0.001, -0.5, 0.004, 2.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[-0.001, 0.5, 3.0, -1.0]], dtype=torch.bfloat16)
    original = residual.view(torch.int16).clone()
    values, stream = layer_operands("moe_out", hidden, residual)
    assert values is hidden and stream is None
    # Selection uses |h| alone: position 1 cancels in h + r but is kept here.
    counts = threshold(values, 0.005, stream)
    assert hidden.tolist() == [[0, -0.5, 0, 2.0]]
    assert torch.equal(residual.view(torch.int16), original)
    assert counts.tolist() == [[4, 2, 0, 1, 1, 0]]
    hidden = torch.tensor([[0.25, -1.0, 0.5, 4.0]], dtype=torch.bfloat16)
    values, stream = layer_operands("moe_out", hidden, residual)
    threshold(values, 0, stream, quantile=0.5)
    assert hidden.tolist() == [[0, -1.0, 0, 4.0]]
    assert torch.equal(residual.view(torch.int16), original)
    before = hidden.view(torch.int16).clone()
    threshold(values, 0, stream, quantile=0.0)
    assert torch.equal(hidden.view(torch.int16), before)
    assert layer_operands("block", hidden, residual) == (hidden, residual)
    with pytest.raises(ValueError):
        layer_operands("mlp", hidden, residual)


def test_run_cli_site_choices_match_rounding_sites():
    # run.py cannot import rounding_stats (the launcher env lacks torch), so its
    # hard-coded choices must track SITES, which rounding_server.py uses directly.
    from rounding_stats import SITES

    help_text = subprocess.run(
        [sys.executable, str(run.SCRIPT), "--help"], capture_output=True, text=True
    ).stdout
    assert "--round-activations-site {" + ",".join(SITES) + "}" in help_text


def fake_eval_server(tmp_path, monkeypatch, served):
    calls = []
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    monkeypatch.setattr(run, "status", lambda: None)
    monkeypatch.setattr(
        run, "get_json", lambda path: served if "config" in path else {"layers": []}
    )
    monkeypatch.setattr(
        run.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    return calls


def test_evaluate_routes_chat_rows_to_chat_eval(tmp_path, monkeypatch):
    served = {"site": "block", "epsilon": 0.0, "quantile": 0.5}
    calls = fake_eval_server(tmp_path, monkeypatch, served)
    run.select("chat_block_q0.5")
    output = run.evaluate(2, tmp_path / "out")
    command, kwargs = calls[0]
    assert command[:3] == [
        run.JOB_PYTHON,
        str(run.SCRIPT.with_name("chat_eval.py")),
        "run",
    ]
    assert command[command.index("--limit") + 1] == "2"
    assert kwargs["cwd"] == output and kwargs["env"]["HF_DATASETS_OFFLINE"] == "1"
    stored = json.loads((output / "experiment.json").read_text())
    assert stored["name"] == "chat_block_q0.5"
    assert (output / "rounding-after.json").exists()


def test_default_eval_is_generative_and_other_tasks_are_rejected(tmp_path, monkeypatch):
    calls = fake_eval_server(tmp_path, monkeypatch, None)
    output = run.evaluate(1)
    assert output.parent == tmp_path / "chat-smoke"
    assert calls[0][0][1] == str(run.SCRIPT.with_name("chat_eval.py"))
    stored = json.loads((output / "experiment.json").read_text())
    assert stored["task"] == chat_eval.TASK
    run.CONFIG["task"] = "mmlu"
    with pytest.raises(RuntimeError, match="generative"):
        run.evaluate(1)
    assert len(calls) == 1


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
    write_chat_run(tmp_path, "chat_moe_out_eps0.1", ["A", "A"])
    rows = {row["experiment"]: row for row in chat_eval.summarize()}
    assert rows["chat_baseline"]["accuracy_percent"] == 100
    moe_out = rows["chat_moe_out_eps0.1"]
    assert (moe_out["site"], moe_out["epsilon"], moe_out["quantile"]) == (
        "moe_out",
        0.1,
        0,
    )
    assert moe_out["changed_vs_baseline"] == 0
    worse = rows["chat_block_q0.5"]
    assert worse["accuracy_percent"] == 50 and worse["unparsed"] == 1
    assert (worse["changed_vs_baseline"], worse["broken_vs_baseline"]) == (1, 1)
    assert worse["min_layer_total_zero_percent"] == 51
    assert (tmp_path / "rounding-chat-sweep-summary.csv").exists()
    chat_eval.summarize("chat_block*")
    assert (tmp_path / "rounding-chat-sweep-chat_block_-summary.csv").exists()
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


def test_schedule_maps_equal_consecutive_layer_groups():
    schedule = validate_schedule(["0", "0.5", 0.75, 0.9])
    assert [group_quantile(schedule, layer, 8) for layer in range(8)] == [
        0,
        0,
        0.5,
        0.5,
        0.75,
        0.75,
        0.9,
        0.9,
    ]
    assert group_quantile([0.1] * 8, 47, 48) == 0.1
    with pytest.raises(ValueError):
        group_quantile([0.1] * 5, 0, 48)
    for bad in ([], [0.5, 1.0], [float("nan")]):
        with pytest.raises(ValueError):
            validate_schedule(bad)


def test_schedule_rows_pass_and_check_the_schedule(monkeypatch):
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    run.select("prof_block_g2_q0.75")
    args = run.rounding_args()
    schedule = args[args.index("--round-activations-quantiles") + 1]
    assert schedule == "0.0,0.0,0.75,0.0,0.0,0.0,0.0,0.0"
    served = {"site": "block", "epsilon": 0.0, "quantile": 0.0}
    assert run.config_mismatch(dict(served, quantiles=[0.0] * 8))
    quantiles = [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert not run.config_mismatch(dict(served, quantiles=quantiles))
    run.select("chat_mlp_q0.5")
    assert not run.config_mismatch(
        {"site": "mlp", "epsilon": 0.0, "quantile": 0.5, "quantiles": None}
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--round-activations-quantiles", "0.5,x"],
        ["--round-activations-quantiles", "0.5,1.0"],
        ["--round-activations-eps", "0.1", "--round-activations-quantiles", "0.5"],
        ["--name", "prof_baseline", "--round-activations-quantiles", "0.5"],
    ],
)
def test_run_cli_rejects_bad_schedules(flags):
    result = subprocess.run(
        [sys.executable, str(run.SCRIPT), "eval", *flags],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr


def test_profiling_grid_is_one_group_at_a_time():
    prof = [c for c in run.experiment_configs if c.get("name", "").startswith("prof_")]
    assert len(prof) == 50 == len({c["name"] for c in prof})
    for config in prof:
        schedule = config["round_activations_quantiles"]
        assert len(schedule) == 8 and config["chat_reference"] == "chat_baseline"
        assert config["round_activations_quantile"] == 0 and config["launcher"] == "mls"
        nonzero = [value for value in schedule if value]
        if config["name"].startswith("prof_baseline"):
            assert not nonzero
        else:
            _, site, group, level = config["name"].split("_")
            assert config["round_activations_site"] == site
            assert schedule[int(group[1:])] == float(level[1:]) and len(nonzero) == 1
            assert float(level[1:]) in run.PROFILE_LEVELS[site]


class EncodingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<p:" + messages[0]["content"][-3:] + ">"

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 0


def test_reference_run_reuses_questions_and_scores_reference(tmp_path, monkeypatch):
    references = [
        dict(
            QUESTION,
            index=i,
            prompt=f"P{i}",
            reasoning="ab",
            prediction="B",
            correct=True,
            mode="natural",
        )
        for i in range(2)
    ]
    references[1].update(
        reasoning="x", forced_suffix="!", forced_text="B", mode="forced_cap"
    )
    monkeypatch.setattr(chat_eval, "reference_records", lambda name, limit: references)
    scored = []

    def post(body):
        if body.get("echo"):
            scored.append(body["prompt"])
            count = len(body["prompt"]) + 1
            return {
                "choices": [
                    {"logprobs": {"token_logprobs": [None] + [-0.5] * (count - 1)}}
                ]
            }
        return {
            "choices": [{"text": "Answer: B", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }

    config = dict(run.PROFILE_CONFIG, chat_concurrency=2)
    result = chat_eval.run_eval(
        config, tmp_path, "http://x", post=post, tokenizer=EncodingTokenizer()
    )
    records = [json.loads(line) for line in (tmp_path / "samples.jsonl").open()]
    assert [r["index"] for r in records] == [0, 1]
    # Natural stop scores its end token; the injected suffix "!" is context only.
    assert scored == [[ord(c) for c in "P0ab"] + [0], [ord(c) for c in "P1x!B"]]
    assert [r["reference_tokens"] for r in records] == [3, 2]
    assert records[1]["reference_logprob"] == -1.0
    assert result["reference"] == {
        "name": "chat_baseline",
        "tokens": 5,
        "logprob": -2.5,
        "nll_per_token": 0.5,
    }


def test_reference_scoring_rejects_missing_logprobs():
    def post(body):
        return {"choices": [{"logprobs": {"token_logprobs": [None, None, -1.0]}}]}

    with pytest.raises(RuntimeError, match="Missing prompt"):
        chat_eval.score_reference([1], [([2, 3], True)], post, CHAT)


def test_reference_records_limit_keeps_first_per_subject(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    output = tmp_path / "jobs" / "chat_baseline" / "a"
    output.mkdir(parents=True)
    rows = [{"subject": s, "index": i} for s in ("a", "b") for i in range(3)]
    (output / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    marker = tmp_path / "jobs" / "chat_baseline" / "done.json"
    marker.write_text(json.dumps({"output": str(output)}))
    kept = chat_eval.reference_records("chat_baseline", 2)
    assert [(r["subject"], r["index"]) for r in kept] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert len(chat_eval.reference_records("chat_baseline")) == 6


def test_allocation_meets_budget_at_minimum_cost():
    costs = {(g, q): (10 if g == 0 else 1) * q for g in range(4) for q in (0.5, 0.9)}
    plan = sensitivity.allocate(costs, (0.5, 0.9), 0.6, 4)
    assert plan["schedule"][0] == 0.0 and plan["mean"] >= 0.6
    brute = min(
        sum(costs[(g, q)] for g, q in enumerate(c) if q)
        for c in itertools.product((0.0, 0.5, 0.9), repeat=4)
        if sum(c) / 4 >= 0.6
    )
    assert plan["predicted_cost"] == pytest.approx(brute)
    with pytest.raises(ValueError, match="unreachable"):
        sensitivity.allocate(costs, (0.5,), 0.6, 4)


def write_profile_run(root, name, schedule, nll, predictions):
    output = root / name
    output.mkdir()
    records = {
        (s, 0): {"prediction": p, "correct": p == "A"}
        for s, p in zip(("s0", "s1"), predictions)
    }
    result = {
        "accuracy": sum(r["correct"] for r in records.values()) / 2,
        "modes": {"forced_cap": 1},
        "reasoning_tokens": {"mean": 9},
        "reference": {"nll_per_token": nll},
    }
    (output / "results.json").write_text(json.dumps(result))
    layers = [
        {"floats": 100, "newly_zeroed_floats": int(100 * q), "already_zero_floats": 0}
        for q in schedule
        for _ in range(2)
    ]
    (output / "rounding-before.json").write_text(json.dumps({"layers": []}))
    (output / "rounding-after.json").write_text(json.dumps({"layers": layers}))
    config = {
        "round_activations_site": "block",
        "round_activations_quantiles": schedule,
    }
    return config, output, records


def test_profile_rows_report_cost_changes_and_group_shares(tmp_path):
    runs = {
        "prof_baseline": write_profile_run(tmp_path, "b", [0, 0, 0], 0.4, "AA"),
        "prof_block_g1_q0.5": write_profile_run(tmp_path, "g", [0, 0.5, 0], 0.7, "AB"),
    }
    reference = {
        ("s0", 0): {"prediction": "A", "correct": True},
        ("s1", 0): {"prediction": "A", "correct": True},
    }
    rows = {r["experiment"]: r for r in sensitivity.profile(runs, reference)}
    row = rows["prof_block_g1_q0.5"]
    assert row["group"] == 1 and row["layers"] == "2-3" and row["quantile"] == 0.5
    assert row["delta_nll_per_token"] == pytest.approx(0.3)
    assert (row["changed_vs_baseline"], row["broken_vs_baseline"]) == (1, 1)
    assert row["group_zero_percent"] == 50 and row["other_groups_max_zero_percent"] == 0
    assert rows["prof_baseline"]["delta_nll_per_token"] == 0


def test_allocation_ties_are_deterministic_and_even():
    costs = {(g, q): 0.0 for g in range(8) for q in (0.5, 0.75, 0.9)}
    raw = dict(costs)
    first = sensitivity.allocate(costs, (0.5, 0.75, 0.9), 0.7, 8, raw)
    again = sensitivity.allocate(dict(costs), (0.5, 0.75, 0.9), 0.7, 8, dict(raw))
    assert first == again and first["mean"] == pytest.approx(0.7)
    # The exact budget wins over evenness; 4 x 0.5 + 4 x 0.9 is the only way.
    assert first["schedule"] == [0.5] * 4 + [0.9] * 4
    raw[(3, 0.9)] = -0.01  # cheapest by the unclamped loss
    assert sensitivity.allocate(costs, (0.9,), 0.1, 8, raw)["schedule"][3] == 0.9
    block = sensitivity.allocate(
        {(g, q): 0.0 for g in range(8) for q in (0.25, 0.5, 0.75)},
        (0.25, 0.5, 0.75),
        0.3,
        8,
    )
    assert block["mean"] == 0.3125


def test_adaptive_rows_are_chat_only(monkeypatch):
    schedule = [0.75, 0, 0.5, 0.75, 0.5, 0.75, 0.25, 0.5]
    rows = run.adaptive_rows({("block", 0.5): schedule})
    assert [r["name"] for r in rows] == ["chat_adapt_block_b0.5"]
    (chat,) = rows
    assert chat["task"] == "mmlu_stem_chat" and "chat_reference" not in chat
    assert chat["launcher"] == "mls"
    monkeypatch.setattr(run, "CONFIG", dict(chat))
    args = run.rounding_args()
    assert args[args.index("--round-activations-quantiles") + 1] == (
        "0.75,0,0.5,0.75,0.5,0.75,0.25,0.5"
    )


def test_live_check_counts_follow_the_schedule():
    import validate_live

    config = {
        "site": "block",
        "epsilon": 0.0,
        "quantile": 0.0,
        "quantiles": [0.5] + [0.0] * 7,
    }
    layers = [
        {
            "tokens": 2,
            "floats": 4096,
            "newly_zeroed_floats": 2048 if i < 6 else 0,
            "already_zero_floats": 0,
            "affected_tokens": 2 if i < 6 else 0,
        }
        for i in range(48)
    ]
    validate_live.check_counts({"rounding_statistics": {"layers": layers}}, config)
    layers[0]["newly_zeroed_floats"] = 100
    with pytest.raises(AssertionError):
        validate_live.check_counts({"rounding_statistics": {"layers": layers}}, config)
    assert validate_live.zeroing(config)
    assert not validate_live.zeroing(dict(config, quantiles=[0.0] * 8))


def test_registered_adaptive_schedules_meet_their_budgets():
    for (site, budget), schedule in run.ADAPTIVE_SCHEDULES.items():
        assert len(schedule) == run.GROUPS and budget in run.ADAPTIVE_BUDGETS[site]
        assert set(schedule) <= {0.0, *run.PROFILE_LEVELS[site]}
        assert sum(schedule) / run.GROUPS >= budget - 1e-9
    names = [c["name"] for c in run_jobs.job_configs("*adapt_*")]
    assert len(names) == len(run.ADAPTIVE_SCHEDULES) == len(set(names))


def test_token_stats_signed_moments_and_magnitude_quantiles():
    stats = hidden_viz.token_stats(torch.tensor([[-3.0, -1, 0, 2, 5, 1, -2, 4]]))
    # |x| sorted: 0 1 1 2 2 3 4 5; q takes the max(floor(q * 8), 1)-th smallest.
    assert stats.tolist() == [[-3, 5, 0.75, 0.5, 0, 1, 2, 3, 4]]
    odd = hidden_viz.token_stats(torch.tensor([[3.0, -1, 2]]))
    assert odd[0, :4].tolist() == pytest.approx([-1, 3, 4 / 3, 2])


def test_token_stats_quantiles_are_quantile_zeroing_cutoffs():
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(4, 2048, generator=generator).to(torch.bfloat16)
    stats = hidden_viz.token_stats(rows)
    for column, quantile in zip(range(4, 9), hidden_viz.QUANTILES):
        values = rows.clone()
        threshold(values, 0.0, None, quantile)
        expected = rows.float().abs() <= stats[:, column : column + 1]
        assert torch.equal((values == 0) & (rows != 0), expected & (rows != 0))


def test_viewer_state_ranges_and_colour_transforms():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    page = hidden_viz.TEMPLATE.read_text()
    lib = page.split('<script id="viz-lib">')[1].split("</script>")[0]
    script = (
        lib
        + """
const fails = (text) => { try { parseStates(text, 49); return null; }
  catch (error) { return error.message; } };
console.log(JSON.stringify({
  example: parseStates("1-8,15,25,30,35,40-48", 49),
  default: parseStates("0,8,16,24,32,40,48", 49),
  spaced: parseStates(" 3 - 5 , 3", 49),
  errors: ["", " ", "1-", "5-3", "49", "0-49", "a", "1,,2", "-1"].map(fails),
  signed: [transform("min", -99), untransform("min", transform("min", -99))],
  magnitude: [transform("q50", 100), transform("q50", 0), untransform("q50", -2)],
  dist_abs: distribution([0, 1, 2], [3, 4, 0], [1, 10, 100]),
  dist_linear: linearDistribution([5, 0, 7, 1], 2),
  sites: [siteStates([0, 1, 8, 48], 1), siteStates([0, 8], 0)],
}));
"""
    )
    result = json.loads(
        subprocess.run(
            [node, "-e", script], capture_output=True, text=True, check=True
        ).stdout
    )
    assert result["example"] == [*range(1, 9), 15, 25, 30, 35, *range(40, 49)]
    assert result["default"] == [0, 8, 16, 24, 32, 40, 48]
    assert result["spaced"] == [3, 4, 5]
    assert all(result["errors"])
    assert result["signed"] == pytest.approx([-2, -99])
    assert result["magnitude"] == pytest.approx([2, None, 0.01])
    absolute = result["dist_abs"]
    assert [p["x"] for p in absolute] == pytest.approx([10**0.5, 10**1.5])
    assert [p["count"] for p in absolute] == [5, 2]
    assert [(p["lo"], p["hi"]) for p in absolute] == [(1, 10), (10, None)]
    linear = result["dist_linear"]
    assert [p["x"] for p in linear] == pytest.approx([-0.75, -0.25, 0.25, 0.75])
    assert [(p["lo"], p["hi"]) for p in linear] == [(-2, -1), (-1, 0), (0, 1), (1, 2)]
    assert [p["count"] for p in linear] == [5, 0, 7, 1]
    assert result["sites"] == [
        {"shown": [1, 8, 48], "dropped": [0]},
        {"shown": [0, 8], "dropped": []},
    ]


def test_viewer_embeds_data_that_cannot_close_its_script():
    text = "</script><!-- <script> <b>"
    page = hidden_viz.render(
        [{"tokens": [text], "question": text}], {"stat_names": hidden_viz.STATS}
    )
    assert "%%" not in page and hidden_viz.PLOTLY in page
    data = page.split('<script type="application/json" id="data">')[1]
    data = data.split("</script>")[0]
    assert "<" not in data
    assert json.loads(data)["samples"][0] == {"tokens": [text], "question": text}


def test_viewer_commit_marks_uncommitted_experiment_changes(monkeypatch):
    outputs = {("rev-parse", "HEAD"): "abc", ("status", "--porcelain", "."): ""}

    def fake(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, outputs[tuple(args[1:])], "")

    monkeypatch.setattr(hidden_viz.subprocess, "run", fake)
    assert hidden_viz.commit() == "abc"
    outputs[("status", "--porcelain", ".")] = " M hidden_viz.py"
    assert hidden_viz.commit() == "abc-dirty"


def test_viewer_value_histogram_bins_by_sign_and_log_magnitude():
    values = torch.tensor([[0.0, 1e-6, -1.0, 1.0], [1.5, -3000.0, 2e4, -0.0]])
    counts = hidden_viz.value_histogram(values.to(torch.bfloat16).float())
    assert counts.shape == (2, len(hidden_viz.HIST_EDGES)) == (2, 91)
    assert counts.sum() == values.numel()
    # Bin i holds 10^((i - 51) / 10) <= |x| < 10^((i - 50) / 10); bin 0 below 1e-5.
    negative = {i: n for i, n in enumerate(counts[0].tolist()) if n}
    positive = {i: n for i, n in enumerate(counts[1].tolist()) if n}
    assert negative == {51: 1, 85: 1}
    assert positive == {0: 3, 51: 1, 52: 1, 90: 1}


def test_viewer_linear_histogram_spans_the_rows_quantile_of_magnitudes():
    values = torch.arange(1.0, 1001.0).view(10, 100)
    values[::2] *= -1
    counts, reach = hidden_viz.linear_histogram(values)
    # ceil(0.995 * 1000)-th smallest |x| is 995; |x| > 995 falls outside the range.
    assert reach == 995 and counts.shape == (hidden_viz.LINEAR_BINS,)
    assert counts.sum() == 995


class CharTokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 0}[token]


def test_viewer_sequence_rebuilds_reference_response_tokens():
    record = {"subject": "s", "index": 1, "prompt": "ab", "reasoning": "cd"}
    natural = dict(record, reasoning_tokens=3)
    prompt, response, scored = hidden_viz.sequence(natural, CharTokenizer())
    assert (prompt, response, scored) == ([97, 98], [99, 100, 0], [True] * 3)
    forced = dict(record, reasoning_tokens=2, forced_suffix="x", forced_text="y")
    _, response, scored = hidden_viz.sequence(forced, CharTokenizer())
    assert (response, scored) == ([99, 100, 120, 121], [True, True, False, True])
    with pytest.raises(ValueError, match="re-tokenizes"):
        hidden_viz.sequence(dict(record, reasoning_tokens=2), CharTokenizer())
    rows = [{"subject": s, "index": i} for s, i in reversed(hidden_viz.SAMPLES)]
    assert hidden_viz.pick(rows) == rows[::-1]
    with pytest.raises(ValueError, match="no records"):
        hidden_viz.pick(rows[1:])


def test_viewer_capture_records_every_site_per_layer():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    outputs, attention = [], []
    hooks = [
        layer.register_forward_hook(lambda module, args, out: outputs.append(out))
        for layer in model.model.layers
    ] + [
        layer.self_attn.register_forward_hook(
            lambda module, args, out: attention.append(out[0][0])
        )
        for layer in model.model.layers
    ]
    ids = [1, 5, 9, 3, 7]
    sites, argmax = hidden_viz.capture(model, ids)
    for hook in hooks:
        hook.remove()
    with torch.no_grad():
        embedding = model.model.embed_tokens(torch.tensor(ids))
        logits = model(torch.tensor([ids])).logits[0]
    shapes = {site: tuple(rows.shape) for site, rows in sites.items()}
    assert list(shapes) == list(hidden_viz.SITES)
    assert shapes == {
        "block": (4, 5, 16),
        "residual": (3, 5, 16),
        "moe_out": (3, 5, 16),
        "experts": (3, 5, 2 * 16),
        "mlp": (3, 5, 2 * 8),
    }
    block = sites["block"]
    assert torch.equal(block[0], embedding) and len(outputs) == 3
    for layer, output in enumerate(outputs):
        assert torch.equal(block[layer + 1], output[0])
        # r = layer input + attention output; the layer output is r + MoE output.
        residual = sites["residual"][layer]
        assert torch.equal(residual, block[layer] + attention[layer])
        assert torch.equal(block[layer + 1], residual + sites["moe_out"][layer])
        module = model.model.layers[layer]
        with torch.no_grad():
            x = module.post_attention_layernorm(residual)
            chosen = module.mlp.gate(x).softmax(dim=-1).topk(2, dim=-1).indices
            for token in range(len(ids)):
                for slot, index in enumerate(chosen[token].tolist()):
                    expert = module.mlp.experts[index]
                    row = x[token : token + 1]
                    act = expert.act_fn(expert.gate_proj(row)) * expert.up_proj(row)
                    mlp = sites["mlp"][layer, token, slot * 8 : (slot + 1) * 8]
                    out = sites["experts"][layer, token, slot * 16 : (slot + 1) * 16]
                    assert torch.allclose(mlp, act[0], atol=1e-6)
                    assert torch.allclose(out, expert(row)[0], atol=1e-6)
    assert torch.equal(argmax, logits.argmax(dim=-1))
