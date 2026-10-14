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

import benchmark_data
import benchmark_eval
import chat_eval
import compression
import embedding_viz
import gsm8k
import hidden_viz
import numpy as np
import pytest
import run_jobs
import sensitivity
import torch
import watch_jobs
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


@pytest.mark.parametrize(
    "text,count,expected",
    [
        ("Reasoning\nAnswer: J", 10, "J"),
        ("Answer: **I**", 10, "I"),
        ("Answer: E", 4, None),
        ("Answer: A or B", 4, None),
        ("Answer: A\nAnswer: D", 4, "D"),
    ],
)
def test_transfer_choice_parser_accepts_only_one_valid_final_option(
    text, count, expected
):
    assert benchmark_eval.choice_answer(text, count) == expected


def test_ifeval_preserves_instructions_without_answer_wrapper():
    question = {"question": "Respond with exactly one word: yes."}
    assert benchmark_eval.content(question, "ifeval") == question["question"]


def test_math_transfer_grades_equivalent_answers_and_rejects_truncation():
    pytest.importorskip("math_verify")
    question = dict(
        id="test",
        answer=r"\frac{1}{2}",
        response=r"The result is \boxed{0.5}",
        finish_reason="stop",
    )
    assert benchmark_eval.grade(question, "math500")["correct"]
    assert not benchmark_eval.grade(dict(question, finish_reason="length"), "math500")[
        "correct"
    ]
    assert not benchmark_eval.grade(dict(question, response=r"\boxed{2}"), "math500")[
        "correct"
    ]


def test_code_transfer_requires_base_and_extra_tests_to_pass():
    pytest.importorskip("evalplus")
    pytest.importorskip("tree_sitter_python")
    if not run.Path(run.CODE_GRADER["path"]).exists():
        pytest.skip("Pinned grader is prepared by benchmark_data.py")
    config = dict(
        benchmark={"dataset": "humaneval"},
        code_grader=run.CODE_GRADER,
        code_min_time_limit=0.1,
        code_time_limit_factor=4.0,
    )
    problem = dict(entry_point="identity", base_input=[[1]], plus_input=[[2]], atol=0)
    oracle = dict(base=([1], [0.001]), plus=([2], [0.001]))
    record = dict(response="def identity(x):\n    return x", finish_reason="stop")
    assert benchmark_eval.code_grade(record, config, problem, oracle)["correct"]
    broken = benchmark_eval.code_grade(
        dict(record, response="def identity(x):\n    return 1"), config, problem, oracle
    )
    assert broken["base_passed"] and not broken["plus_passed"]
    assert not broken["correct"]


def test_code_polynomial_oracle_accepts_a_correct_root_without_reference_equality():
    pytest.importorskip("evalplus")
    pytest.importorskip("tree_sitter_python")
    if not run.Path(run.CODE_GRADER["path"]).exists():
        pytest.skip("Pinned grader is prepared by benchmark_data.py")
    config = dict(
        benchmark={"dataset": "humaneval"},
        code_grader=run.CODE_GRADER,
        code_min_time_limit=0.1,
        code_time_limit_factor=4.0,
    )
    problem = dict(
        entry_point="find_zero",
        base_input=[[[-1.0, 0.0, 1.0]]],
        plus_input=[[[-1.0, 0.0, 1.0]]],
        atol=1e-6,
    )
    oracle = dict(base=([-1.0], [0.001]), plus=([-1.0], [0.001]))
    record = dict(response="def find_zero(xs):\n    return 1.0", finish_reason="stop")
    assert benchmark_eval.code_grade(record, config, problem, oracle)["correct"]
    assert not benchmark_eval.code_grade(
        dict(record, response="def find_zero(xs):\n    return 0.0"),
        config,
        problem,
        oracle,
    )["correct"]


def test_prepared_benchmark_rejects_changed_inputs(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id":"1"}\n')
    settings = {"expected_questions": 1}
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                benchmark=settings,
                packages=run.BENCHMARK_PACKAGES,
                files={"questions.jsonl": benchmark_data.digest(path)},
            )
        )
    )
    benchmark_data.validate(tmp_path, settings)
    path.write_text('{"id":"2"}\n')
    with pytest.raises(ValueError, match="checksum"):
        benchmark_data.validate(tmp_path, settings)


def test_code_oracle_roundtrip_preserves_tuple_set_and_arrays():
    import numpy as np

    original = {1: ({2, 3}, np.array([1.0, 2.0]), [complex(1, 2), 10**5000])}
    decoded = benchmark_data.decode_value(
        json.loads(json.dumps(benchmark_data.encode_value(original)))
    )
    assert isinstance(decoded[1], tuple) and decoded[1][0] == {2, 3}
    assert np.array_equal(decoded[1][1], original[1][1])
    assert decoded[1][2] == [complex(1, 2), 10**5000]


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
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight": torch.eye(2),
        "gpt_neox.layers.0.mlp.dense_4h_to_h.bias": torch.zeros(2),
        "gpt_neox.embed_in.weight": torch.ones(3, 2),
        "gpt_neox.final_layer_norm.weight": torch.ones(2),
        "gpt_neox.layers.0.attention.bias": torch.ones(1, 1, 2, 2, dtype=torch.uint8),
        "gpt_neox.layers.0.attention.masked_bias": torch.tensor(-10000.0),
        "gpt_neox.layers.0.attention.rotary_emb.inv_freq": torch.ones(2),
    }
    save_file(weights, tmp_path / "model.safetensors")
    pythia = next(
        c for c in weight_viz.experiment_configs if c["name"] == "pythia160m_weights"
    )
    config = dict(pythia, snapshot=str(tmp_path))
    records, stats, linear, log, analysis = weight_viz.collect(config, "cpu", tmp_path)
    assert {r[0] for r in records} == {
        n for n in weights if not n.endswith(tuple(config["buffer_suffixes"]))
    }
    assert all(r[2] == -1 for r in records)
    assert {tuple(r[1:4]) for r in records} == {
        (0, -1, "mlp.dense_4h_to_h"),
        (0, -1, "mlp.dense_4h_to_h.bias"),
        (-1, -1, "embed_in"),
        (-1, -1, "final_layer_norm"),
    }
    assert sum(math.prod(r[4]) for r in records) == 14
    assert linear.sum() == log.sum() == 14
    assert list(analysis) == ["model.safetensors"]


@pytest.mark.parametrize("indexed", [False, True])
def test_norm_heatmap_preserves_coordinates_and_excludes_other_widths(
    tmp_path, indexed
):
    import pybase64 as base64
    from safetensors.torch import save_file

    weights = {
        "model.layers.2.input_layernorm.weight": torch.tensor([3.0, -2.0, 0.0]),
        "model.layers.0.input_layernorm.weight": torch.tensor([1.0, 0.5, -1.0]),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(3),
        "model.layers.0.input_layernorm.bias": torch.zeros(3),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
        "model.norm.weight": torch.ones(3),
    }
    save_file(weights, tmp_path / "model.safetensors")
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {n: "model.safetensors" for n in weights}})
        )
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 3}))
    records = [
        [n, *weight_viz.identity(n), list(w.shape), 4.0] for n, w in weights.items()
    ]
    norms = weight_viz.norm_weights(
        records, dict(weight_viz.CONFIG, snapshot=str(tmp_path))
    )
    assert set(norms) == {"input_layernorm", "post_attention_layernorm"}
    norm = norms["input_layernorm"]
    assert norm["layers"] == [0, 2] and norm["width"] == 3
    values = torch.frombuffer(
        bytearray(base64.b64decode(norm["values"])), dtype=torch.float32
    )
    assert values.reshape(2, 3).tolist() == [[1.0, 0.5, -1.0], [3.0, -2.0, 0.0]]


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
  norm: normRow([3,-4,0],true),
  normScaled: normRow([30,-40,0],true),
  normRaw: normRow([3,-4,0],false),
  normZero: normRow([0,0],true),
  normMissing: normRow([null,null],true),
  similarity: normalizationSimilarity({layers:[0,1,2,3,4,5],width:4,
    values:Float32Array.from([1,2,2,4, 10,20,20,40, -1,-2,-2,-4,
      4,2,2,1, 1,1,1,1, 0,0,0,0])}),
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
    assert result["norm"] == pytest.approx(
        [3 * math.sqrt(3) / 5, -4 * math.sqrt(3) / 5, 0]
    )
    assert result["normScaled"] == pytest.approx(result["norm"])
    assert result["normRaw"] == [3, -4, 0]
    assert result["normZero"] == result["normMissing"] == [None, None]
    similarity = result["similarity"]
    assert similarity["cosine"][0][1] == pytest.approx(1)
    assert similarity["cosine"][0][2] == pytest.approx(-1)
    assert similarity["spearman"][0][1] == pytest.approx(1)
    assert similarity["spearman"][0][2] == pytest.approx(1)
    assert similarity["spearman"][0][3] == pytest.approx(-1)
    assert similarity["spearman"][4] == similarity["spearman"][5] == [None] * 6
    assert similarity["cosine"][5] == [None] * 6
    assert similarity["rms"][1] == pytest.approx(10 * similarity["rms"][0])


def test_threshold_counts_new_zeros_and_affected_tokens():
    values = torch.tensor([[0, 0.125, -0.125, 0.25], [0, 0, 1, -1]])
    counts = threshold(values, 0.125)
    assert values.tolist() == [[0, 0, 0, 0.25], [0, 0, 1, -1]]
    assert counts.tolist() == [[4, 2, 1, 1, 1, 0], [4, 0, 2, 1, 0, 0]]


def test_norm_layer_similarity_separates_scale_sign_and_coordinate_order():
    from norm_weight_analysis import layer_similarity

    weights = np.array(
        [
            [1.0, 2.0, 2.0, 4.0],
            [10.0, 20.0, 20.0, 40.0],
            [-1.0, -2.0, -2.0, -4.0],
            [4.0, 2.0, 2.0, 1.0],
        ]
    )
    cosine, spearman = layer_similarity(weights)
    assert cosine[0, 1] == pytest.approx(1)
    assert cosine[0, 2] == pytest.approx(-1)
    assert spearman[0, 1] == spearman[0, 2] == pytest.approx(1)
    assert spearman[0, 3] == pytest.approx(-1)
    relative = weights / np.sqrt(np.mean(weights**2, axis=1, keepdims=True))
    assert np.mean((relative[0] - relative[3]) ** 2) == pytest.approx(
        2 - 2 * cosine[0, 3]
    )
    _, constant = layer_similarity(np.ones((2, 4)))
    assert np.isnan(constant).all()


@pytest.mark.parametrize("offset", [0, 1])
def test_cached_norm_comparison_reads_decoder_scales_with_effective_offset(
    tmp_path, offset
):
    from qwen_norm_compare import FAMILIES, read_scales
    from safetensors.torch import save_file

    prefix = "model.language_model.layers"
    weights = {
        f"{prefix}.{layer}.{family}.weight": torch.tensor([-0.5, 0.0, 0.5])
        for layer in range(2)
        for family in FAMILIES
    }
    weights["model.visual.layers.0.input_layernorm.weight"] = torch.ones(7)
    weights[f"{prefix}.0.self_attn.q_norm.weight"] = torch.ones(2)
    save_file(weights, tmp_path / "norms.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "norms.safetensors" for n in weights}})
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 2, "hidden_size": 3}})
    )
    scales = read_scales(
        {
            "snapshot": str(tmp_path),
            "prefix": prefix,
            "text_config_path": ["text_config"],
            "weight_offset": offset,
        }
    )
    assert set(scales) == set(FAMILIES)
    for array in scales.values():
        np.testing.assert_array_equal(
            array, np.tile(np.array([-0.5, 0.0, 0.5]) + offset, (2, 1))
        )


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
        assert config["name"].startswith(("chat_", "prof_", "gsm8k_", "suite_"))
        if config["task"] == gsm8k.TASK:
            assert "chat_subjects" not in config and "chat_per_subject" not in config
            assert config["chat_expected_questions"] == 1319
        elif config["task"] in run.BENCHMARK_CONFIGS:
            assert config["benchmark"] == run.BENCHMARK_CONFIGS[config["task"]]
            original = next(
                c
                for c in run.experiment_configs
                if c["name"]
                == "gsm8k" + config["name"].removeprefix("suite_" + config["task"])
            )
            assert {k: v for k, v in config.items() if k.startswith("round_")} == {
                k: v for k, v in original.items() if k.startswith("round_")
            }
        else:
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
        "chat_weight_embedding_q0.5",
        "chat_weight_both_q0.5",
        "chat_mlp_signed_epsilon_q0.5",
        "chat_mlp_weights_zero_q0.5",
        "chat_mlp_weights_signed_epsilon_normal_q0.5",
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
        and not c.get("weight_zeroing")
        and not c.get("mlp_weight_rounding")
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
        if c.get("name", "").startswith(
            tuple(f"chat_{site}_eps" for site in ("mlp", "block", "moe_out"))
            + ("chat_moe_out",)
        )
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
    names = [c["name"] for c in run_jobs.job_configs("chat_adapt_*")]
    assert len(names) == len(run.ADAPTIVE_SCHEDULES) == len(set(names))
    gsm8k_adaptive = run_jobs.job_configs("gsm8k_adapt_*")
    assert len(gsm8k_adaptive) == 1
    schedule = gsm8k_adaptive[0]["round_activations_quantiles"]
    assert schedule == run.ADAPTIVE_SCHEDULES[("mlp", 0.8)]
    assert schedule is not run.ADAPTIVE_SCHEDULES[("mlp", 0.8)]


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: 1,200.00", "1200"),
        ("**Answer:** $0.00$", "0"),
        (r"Answer: \boxed{-12.50}", "-12.5"),
        ("Answer: +42", "42"),
        ("Reasoning contains 42 but no final answer", None),
        ("Answer: 42\nAnswer: uncertain", None),
        ("Answer: 2 + 3", None),
        ("Answer: 1/2", None),
        ("Answer: 12,34", None),
        ("Answer: 42 or 43", None),
        ("Answer: 42e3", None),
    ],
)
def test_gsm8k_final_answer_normalizes_only_unambiguous_numbers(text, expected):
    assert gsm8k.final_answer(text) == expected


def test_gsm8k_eval_uses_test_gold_without_leaking_solution(tmp_path):
    config = dict(run.GSM8K_CONFIG, chat_expected_questions=3)
    data = [
        {"question": f"Compute {n} + {n}", "answer": f"SECRET solution\n#### {2 * n}"}
        for n in range(3)
    ]
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert not kwargs["enable_thinking"]
            return messages[0]["content"]

    def post(body):
        calls.append(body)
        n = int(body["prompt"].split("Compute ")[1][0])
        return {
            "choices": [{"text": f"Answer: {2 * n}.0", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 4},
        }

    result = chat_eval.run_eval(
        config,
        tmp_path,
        "unused",
        load=lambda c: data,
        post=post,
        tokenizer=Tokenizer(),
    )
    assert result["task"] == gsm8k.TASK and result["questions"] == 3
    assert result["accuracy"] == 1 and result["unparsed"] == 0
    assert all("SECRET" not in call["prompt"] for call in calls)
    records = list(
        map(json.loads, (tmp_path / "samples.jsonl").read_text().splitlines())
    )
    assert [r["prediction"] for r in records] == ["0", "2", "4"]
    subset = gsm8k.select_questions(config, 2, lambda c: data)
    assert subset == gsm8k.select_questions(config, 2, lambda c: data)
    assert len(subset) == 2
    with pytest.raises(ValueError, match="gold answer"):
        gsm8k.select_questions(
            config, load=lambda c: [dict(r, answer="bad") for r in data]
        )


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_gsm8k_forced_number_rejects_truncated_continuations(finish):
    calls = []

    def post(body):
        calls.append(body)
        return {
            "choices": [
                {
                    "text": "reasoning" if len(calls) == 1 else " 1,200.00",
                    "finish_reason": "length" if len(calls) == 1 else finish,
                }
            ],
            "usage": {"completion_tokens": 8},
        }

    question = {"subject": "gsm8k", "index": 0, "question": "Q", "answer": "1200"}
    record = chat_eval.answer(question, "prompt", post, run.GSM8K_CONFIG)
    assert record["mode"] == "forced_cap"
    assert record["prediction"] == ("1200" if finish == "stop" else None)
    assert record["correct"] == (finish == "stop")
    assert calls[1]["max_tokens"] == run.GSM8K_CONFIG["chat_max_answer_tokens"]


@pytest.mark.parametrize(
    "name",
    [
        "gsm8k_baseline",
        "gsm8k_baseline_repeat",
        "gsm8k_mlp_q0.5",
        "gsm8k_mlp_q0.8",
        "gsm8k_moe_out_q0.4",
        "gsm8k_block_q0.25",
        "gsm8k_adapt_mlp_b0.8",
    ],
)
def test_gsm8k_local_eval_and_mls_payload_use_shared_launcher(
    tmp_path, monkeypatch, name
):
    config = run_jobs.job_configs(name)[0]
    calls = fake_eval_server(
        tmp_path,
        monkeypatch,
        {
            "site": config["round_activations_site"],
            "epsilon": config["round_activations_eps"],
            "quantile": config["round_activations_quantile"],
            "quantiles": config.get("round_activations_quantiles"),
        },
    )
    run.select(name)
    output = run.evaluate(2)
    assert output.parent == tmp_path / "gsm8k-smoke"
    stored = json.loads((output / "experiment.json").read_text())
    assert stored == config
    assert "HF_DATASETS_OFFLINE" not in calls[0][1]["env"]
    payload = run_jobs.payload(config, tmp_path, 2, "SR004")
    assert payload["script"].endswith(f"run.py job --name {name} --limit 2")


@pytest.mark.parametrize("task", list(run.BENCHMARK_CONFIGS))
def test_transfer_local_eval_dispatches_to_offline_benchmark_grader(
    tmp_path, monkeypatch, task
):
    name = "suite_" + task + "_baseline"
    config = run_jobs.job_configs(name)[0]
    calls = fake_eval_server(
        tmp_path,
        monkeypatch,
        {
            "site": config["round_activations_site"],
            "epsilon": config["round_activations_eps"],
            "quantile": config["round_activations_quantile"],
            "quantiles": None,
        },
    )
    run.select(name)
    output = run.evaluate(2)
    assert output.parent == tmp_path / (task + "-smoke")
    assert any(str(arg).endswith("benchmark_eval.py") for arg in calls[0][0])
    assert calls[0][1]["env"]["HF_DATASETS_OFFLINE"] == "1"
    assert json.loads((output / "experiment.json").read_text()) == config


def test_gsm8k_summary_keeps_baseline_separate_from_mmlu(tmp_path, monkeypatch):
    configs = [
        dict(c, chat_expected_questions=2)
        for c in run.experiment_configs
        if c["name"] in ("gsm8k_baseline", "gsm8k_baseline_repeat")
    ]
    monkeypatch.setattr(run, "experiment_configs", configs)
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    write_chat_run(tmp_path, "gsm8k_baseline", ["A", "A"])
    write_chat_run(tmp_path, "gsm8k_baseline_repeat", ["A", None])
    for config in configs:
        marker = run.job_root(config["name"], None) / "done.json"
        output = run.Path(json.loads(marker.read_text())["output"])
        records = list(
            map(json.loads, (output / "samples.jsonl").read_text().splitlines())
        )
        for index, record in enumerate(records):
            record.update(subject="gsm8k", index=index, answer="42")
            record["prediction"] = "42" if record["prediction"] else None
        (output / "samples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        (output / "results.json").write_text(
            json.dumps(chat_eval.summarize_records(records, config))
        )
    rows = {row["experiment"]: row for row in chat_eval.summarize(task=gsm8k.TASK)}
    assert rows["gsm8k_baseline"]["accuracy_percent"] == 100
    assert rows["gsm8k_baseline_repeat"]["broken_vs_baseline"] == 1
    assert (tmp_path / "rounding-gsm8k-sweep-summary.csv").exists()
    assert not (tmp_path / "rounding-chat-sweep-summary.csv").exists()


WATCH_JOBS = [
    {"name": f"arm{i}", "job_name": f"lm-mpi-job-00000000-0000-0000-0000-{i:012d}"}
    for i in range(2)
]


def test_watcher_rejects_incomplete_or_duplicate_job_ids(tmp_path):
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps({"jobs": WATCH_JOBS}))
    assert watch_jobs.load_jobs(manifest) == WATCH_JOBS
    for jobs in (
        [],
        [WATCH_JOBS[0]] * 2,
        [{"name": "arm", "job_name": "lm-mpi-job-00000000"}],
    ):
        manifest.write_text(json.dumps({"jobs": jobs}))
        with pytest.raises(ValueError, match="complete MLS"):
            watch_jobs.load_jobs(manifest)


def test_watcher_api_errors_cannot_be_mistaken_for_completion_or_failure():
    def query(job_id):
        if job_id == WATCH_JOBS[0]["job_name"]:
            raise TimeoutError("transient")
        return {"status": "Completed"}

    rows = watch_jobs.snapshot(WATCH_JOBS, query)
    assert [r["status"] for r in rows] == ["unknown", "completed"]
    assert rows[0]["query_error"] == "TimeoutError"
    assert (
        watch_jobs.terminal_event(rows, {"completed", "failed"}, {"completed"}) is None
    )


@pytest.mark.parametrize(
    "end", ["completed", "failed", "stopped", "terminated", "deleted"]
)
def test_watcher_resumes_once_on_failure_or_all_success(tmp_path, monkeypatch, end):
    phase = [0]
    resumes = []

    def query(job_id):
        return {"status": "running" if phase[0] == 0 else end}

    monkeypatch.setattr(
        watch_jobs.time, "sleep", lambda interval: phase.__setitem__(0, 1)
    )

    def resume(path):
        event = json.loads(path.read_text())
        assert "resume_requested_at_utc" in event
        resumes.append(event)

    terminal = {"completed", "failed", "stopped", "terminated", "deleted"}
    result = watch_jobs.watch(
        WATCH_JOBS, query, terminal, {"completed"}, tmp_path, 1, resume=resume
    )
    assert len(resumes) == 1
    assert result["condition"] == (
        "all_succeeded" if end == "completed" else "job_failed"
    )
    assert (
        json.loads((tmp_path / "status.json").read_text())["jobs"][0]["status"] == end
    )
    assert "resume_worker_returned_at_utc" in json.loads(
        (tmp_path / "event.json").read_text()
    )
    with pytest.raises(RuntimeError, match="already triggered"):
        watch_jobs.watch(
            WATCH_JOBS, query, terminal, {"completed"}, tmp_path, 1, resume=resume
        )
    assert len(resumes) == 1


def test_watcher_one_failed_job_triggers_while_others_are_running():
    rows = [
        dict(job, status=status)
        for job, status in zip(WATCH_JOBS, ["failed", "running"])
    ]
    event = watch_jobs.terminal_event(rows, {"completed", "failed"}, {"completed"})
    assert event["condition"] == "job_failed" and event["failed_jobs"] == rows[:1]
    rows[0].update(status="completed", error_code=1)
    assert (
        watch_jobs.terminal_event(rows, {"completed"}, {"completed"})["condition"]
        == "job_failed"
    )


@pytest.mark.parametrize("label", ["GSM8K", "transfer benchmark suite"])
def test_watcher_resume_targets_original_session_and_topic(
    tmp_path, monkeypatch, label
):
    calls = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    config = {
        "workdir": str(tmp_path),
        "python_executable": "python",
        "message_id": 123,
        "telegram_chat_id": "chat",
        "model": "sol",
        "user_id": 1,
        "session_id": "original-session",
        "telegram_thread_id": 7,
        "evaluation_label": label,
    }
    event_path = tmp_path / "event.json"
    watch_jobs.resume_conversation(
        config, event_path, lambda **kwargs: calls.append(kwargs)
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["resume_session_id"] == config["session_id"]
    assert call["telegram_chat_id"] == "chat" and call["telegram_thread_id"] == 7
    assert call["message_id"] == 123 and call["user_id"] == 1
    assert str(event_path) in call["question"]
    assert label + " results" in call["question"]
    assert call["telegram_token"] == "test-token"


@pytest.mark.parametrize("restart", [False, True])
def test_watcher_retains_and_retries_failed_resume_delivery(
    tmp_path, monkeypatch, restart
):
    attempts = []
    monkeypatch.setattr(watch_jobs.time, "sleep", lambda interval: None)

    def resume(path):
        event = json.loads(path.read_text())
        attempts.append(event)
        if len(attempts) == 1:
            raise ConnectionError("temporary delivery failure")

    def query(job_id):
        return {"status": "failed"}

    if restart:
        with pytest.raises(ConnectionError):
            watch_jobs.watch(
                WATCH_JOBS,
                query,
                {"failed"},
                {"completed"},
                tmp_path,
                1,
                resume=resume,
                once=True,
            )
        pending = json.loads((tmp_path / "event.json").read_text())
        assert pending["resume_error"] == "ConnectionError"
        assert "resume_worker_returned_at_utc" not in pending

        def query(job_id):
            raise AssertionError(
                "Restart should deliver the saved event without polling"
            )

    result = watch_jobs.watch(
        WATCH_JOBS, query, {"failed"}, {"completed"}, tmp_path, 1, resume=resume
    )
    assert len(attempts) == result["resume_attempts"] == 2
    assert result["condition"] == "job_failed"
    assert "resume_error" not in result
    assert "resume_worker_returned_at_utc" in result
    assert attempts[0]["checked_at_utc"] == attempts[1]["checked_at_utc"]


def test_ruler_multivalue_partial_credit_and_qa_aliases():
    import ruler

    record = dict(
        id="niah_multivalue/0",
        subtask="niah_multivalue",
        answers=["123", "456"],
        response="The value is 123.",
        finish_reason="length",
        completion_tokens=128,
        input_tokens=8000,
    )
    scored = ruler.grade(record)
    assert scored["score"] == 0.5
    assert not scored["correct"]
    alias = ruler.grade(dict(record, subtask="qa_1", answers=["value", "other"]))
    assert alias["score"] == 1
    assert alias["correct"]
    full = ruler.grade(dict(record, response="123 and 456", finish_reason="stop"))
    # Official RULER counts matched answers even on a generation cap.
    result = benchmark_eval.metrics([scored, full, alias], "ruler_8k")
    assert result["subtask_scores"] == {"niah_multivalue": 0.75, "qa_1": 1}
    assert result["accuracy"] == 0.875
    assert result["truncated"] == 2


def test_ruler_rejects_overflow_without_truncating_needle():
    import ruler

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return list(text)

    question = dict(id="niah_single_1/0", tokens_to_generate=3, input_tokens=6)
    prompt = "needle"
    assert ruler.check_budget(prompt, question, {"max_model_len": 9}, Tokenizer()) == 6
    with pytest.raises(ValueError, match="exceeds context budget"):
        ruler.check_budget(prompt, question, {"max_model_len": 8}, Tokenizer())
    with pytest.raises(ValueError, match="tokenizer changed"):
        ruler.check_budget("changed", question, {"max_model_len": 12}, Tokenizer())


def test_ruler_long_context_arms_preserve_zeroing_and_share_server_settings():
    rows = run_jobs.job_configs("suite_ruler_*")
    assert len(rows) == 48
    for row in rows:
        assert row["max_model_len"] == row["benchmark"]["context_length"]
        assert row["chat_concurrency"] == row["max_num_seqs"] == 2
        assert row["memory_fraction"] == 0.9
        reference = run_jobs.job_configs(
            "gsm8k" + row["name"].removeprefix("suite_" + row["task"])
        )[0]
        assert {k: v for k, v in row.items() if k.startswith("round_")} == {
            k: v for k, v in reference.items() if k.startswith("round_")
        }


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_ruler_job_requires_preflight_before_full_completion(
    tmp_path, monkeypatch, fail_preflight
):
    name = "suite_ruler_32k_baseline"
    calls, servers = [], []

    def popen(command, **kwargs):
        server = FakeServer(command, **kwargs)
        servers.append(server)
        return server

    def evaluate(limit, output):
        calls.append((limit, output))
        assert not (run.job_root(name, None) / "done.json").exists()
        if fail_preflight:
            raise RuntimeError("preflight failed")
        return output

    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(run, "CONFIG", dict(run.CONFIG))
    monkeypatch.setattr(run.subprocess, "Popen", popen)
    monkeypatch.setattr(run, "wait_ready", lambda alive: alive())
    monkeypatch.setattr(run, "evaluate", evaluate)
    if fail_preflight:
        with pytest.raises(RuntimeError, match="preflight failed"):
            run.job(name, None)
        assert [limit for limit, _ in calls] == [13]
        assert not (run.job_root(name, None) / "done.json").exists()
    else:
        run.job(name, None)
        assert [limit for limit, _ in calls] == [13, None]
        marker = json.loads((run.job_root(name, None) / "done.json").read_text())
        assert marker["smoke_output"] == str(calls[0][1])
        assert marker["output"] == str(calls[1][1])
        assert calls[0][1] != calls[1][1]
    assert servers[0].signals == [run.signal.SIGINT]


def test_ruler_source_hydrates_lfs_and_reuses_cache_after_python_import(
    tmp_path, monkeypatch
):
    import hashlib
    import io
    import tarfile

    import ruler

    words = b'{"0":"apple"}'
    essay = b'{"text":"example"}'
    buffer = io.BytesIO()
    files = {
        "scripts/data/synthetic/json/english_words.json": b"version https://git-lfs.github.com/spec/v1\n",
        "scripts/data/synthetic/json/PaulGrahamEssays.json": essay,
        "scripts/data/synthetic/__pycache__/constants.pyc": b"old cache",
    }
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in files.items():
            member = tarfile.TarInfo("RULER-test/" + name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    raw = buffer.getvalue()
    settings = dict(
        revision="test",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        corpus_urls={"english_words.json": "https://example.test/lfs"},
    )
    calls = []

    def download(url):
        calls.append(url)
        return words if url.endswith("/lfs") else raw

    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        run,
        "RULER_CORPUS_SHA256",
        {
            "english_words.json": hashlib.sha256(words).hexdigest(),
            "PaulGrahamEssays.json": hashlib.sha256(essay).hexdigest(),
        },
    )
    monkeypatch.setattr(benchmark_data, "download", download)
    root, provenance = ruler.source(settings)
    assert json.loads(
        (root / "scripts/data/synthetic/json/english_words.json").read_text()
    ) == {"0": "apple"}
    (root / "scripts/data/synthetic/__pycache__/constants.pyc").write_bytes(
        b"new cache"
    )
    assert ruler.source(settings) == (root, provenance)
    assert len(calls) == 2


def test_embedding_channel_stats_distinguish_bias_from_variation():
    """Constant channels retain RMS/energy but have no population variance."""
    result = embedding_viz.summarize(np.array([[1.0, 2.0], [3.0, 2.0]]), axis=0)
    assert result["mean"] == [2.0, 2.0]
    assert result["std"] == [1.0, 0.0]
    assert result["rms"] == pytest.approx([math.sqrt(5), 2.0])
    assert result["energy_share"] == pytest.approx([10 / 18, 8 / 18])
    assert result["excess_kurtosis"][0] == -2.0
    assert math.isnan(result["excess_kurtosis"][1])


def test_embedding_pair_cosines_preserve_sign_and_undefined_zero_norm():
    """Opposed vectors must be negative; zero-norm pairs are undefined."""
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    result = embedding_viz.cosines(a, b, axis=1)
    assert result[:2] == pytest.approx([1.0, -1.0])
    assert math.isnan(result[2])


def test_embedding_viewer_program_parses_and_decodes_exact_bf16():
    """Offline browser code must parse and retain tiny and signed BF16 weights."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    template = (
        embedding_viz.Path(embedding_viz.__file__).with_suffix(".html").read_text()
    )
    program = template.split("'use strict';", 1)[1].split("</script>", 1)[0]
    subprocess.run(
        [node, "--check"], input=program, text=True, check=True, capture_output=True
    )
    decoder = next(
        line for line in program.splitlines() if line.startswith("function bf16(")
    )
    rows = torch.tensor([1.0, -2.0, 0.0, 1e-9], dtype=torch.bfloat16)
    encoded = embedding_viz.base64.b64encode(
        rows.view(torch.uint16).numpy().astype("<u2").tobytes()
    ).decode()
    script = (
        decoder
        + "\nconsole.log(JSON.stringify(Array.from(bf16("
        + json.dumps(encoded)
        + "))));"
    )
    output = subprocess.run(
        [node, "-e", script], text=True, capture_output=True, check=True
    )
    assert json.loads(output.stdout) == rows.float().tolist()


def test_embedding_zeroing_uses_each_token_vector_and_stable_channel_ties():
    """Different row scales share the fraction, not a matrix-wide cutoff."""
    from embedding_zeroing import zero_rows

    weights = torch.tensor(
        [[4.0, -1.0, 1.0, -2.0], [40.0, -10.0, 10.0, -20.0]], dtype=torch.bfloat16
    )
    report = zero_rows(weights, 0.5, 1)
    assert weights.tolist() == [[4.0, 0.0, 0.0, -2.0], [40.0, 0.0, 0.0, -20.0]]
    assert report["zeros_after"] == 4
    assert report["cutoff_min_median_max"] == [1.0, 5.5, 10.0]
    ties = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.bfloat16)
    zero_rows(ties, 0.5, 2)
    assert ties.tolist() == [[0.0, 0.0, 1.0, -1.0]]


@pytest.mark.parametrize(
    "scope,tied",
    [("embedding", False), ("both", False), ("embedding", True), ("both", True)],
)
def test_embedding_zeroing_handles_scope_tying_and_vocab_padding(scope, tied):
    """Real tied storage is mutated once and unrelated output/padding is preserved."""
    from types import SimpleNamespace

    from embedding_zeroing import apply

    embedding = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.bfloat16
    )
    output = embedding if tied else embedding.clone()
    original = output.clone()
    model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embedding)),
        lm_head=SimpleNamespace(weight=output),
        config=SimpleNamespace(vocab_size=1, hidden_size=4, tie_word_embeddings=tied),
    )
    report = apply(model, dict(scope=scope, quantile=0.5, batch_rows=1))
    assert embedding.tolist() == [[0.0, 0.0, 3.0, -4.0], [5.0, 6.0, 7.0, 8.0]]
    assert report["tied"] is tied
    assert report["matrices"]["embedding"]["newly_zeroed"] == 2
    if scope == "both" or tied:
        assert output.tolist() == embedding.tolist()
    else:
        assert torch.equal(output, original)
    if tied:
        assert report["matrices"]["unembedding"] == {"shared_with": "embedding"}


def test_embedding_weight_registry_covers_every_baseline_except_mmlu_pro(monkeypatch):
    """Weight arms inherit protocols; extreme sparsity affects input weights only."""
    rows = [c for c in run.experiment_configs if c.get("weight_zeroing")]
    assert len(rows) == 90
    assert len({c["name"] for c in rows}) == 90
    for task in {c["task"] for c in rows}:
        assert {
            (c["weight_zeroing"]["scope"], c["weight_zeroing"]["quantile"])
            for c in rows
            if c["task"] == task
        } == {
            *(("embedding", q) for q in (0.5, 0.7, 0.9, 0.95, 0.97, 0.99)),
            *(("both", q) for q in (0.5, 0.7, 0.9)),
        }
    assert {c["task"] for c in rows} == {
        "mmlu_stem_chat",
        "gsm8k_chat",
        *set(run.BENCHMARK_CONFIGS) - {"mmlu_pro"},
    }
    for row in rows:
        assert row["round_activations_quantile"] == 0
        assert row["round_activations_eps"] == 0
        assert not row.get("round_activations_quantiles")
        assert row["weight_zeroing"]["quantile"] in (0.5, 0.7, 0.9, 0.95, 0.97, 0.99)
        baseline_name = row["name"].split("weight_", 1)[0] + "baseline"
        baseline = next(
            c for c in run.experiment_configs if c.get("name") == baseline_name
        )
        assert {
            k: v for k, v in row.items() if k not in ("name", "weight_zeroing")
        } == {k: v for k, v in baseline.items() if k != "name"}
    monkeypatch.setattr(run, "CONFIG", dict(rows[0]))
    cmd = run.server_command(embedding_viz.Path("/tmp/example.sqlite"))
    assert (
        json.loads(cmd[cmd.index("--weight-zeroing") + 1]) == rows[0]["weight_zeroing"]
    )
    actual = dict(site="mlp", epsilon=0, quantile=0)
    assert run.config_mismatch(actual)
    actual["weight_zeroing"] = rows[0]["weight_zeroing"]
    assert not run.config_mismatch(actual)


def test_transfer_summary_labels_each_weight_arm_and_preserves_baseline(
    tmp_path, monkeypatch
):
    """Mixed completed runs must retain their own weight scope and fraction."""
    task = "math500"
    names = [
        "suite_math500_baseline",
        "suite_math500_weight_embedding_q0.5",
        "suite_math500_weight_both_q0.9",
    ]
    data = tmp_path / "data"
    data.mkdir()
    question = dict(id="q0", answer="42")
    (data / "questions.jsonl").write_text(json.dumps(question) + "\n")
    configs = [
        dict(
            c,
            benchmark=dict(c["benchmark"], expected_questions=1),
            benchmark_data=str(data),
        )
        for c in run.experiment_configs
        if c.get("name") in names
    ]
    monkeypatch.setattr(run, "experiment_configs", configs)
    monkeypatch.setattr(run, "BENCHMARK_CONFIGS", {task: {}})
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(benchmark_data, "validate", lambda root, settings: {})
    monkeypatch.setattr(chat_eval, "layer_zero_shares", lambda output: [0.0])
    for config in configs:
        root = run.job_root(config["name"], None)
        output = root / "attempt"
        output.mkdir(parents=True)
        record = dict(
            question,
            prompt="same prompt",
            prediction="42",
            correct=True,
            finish_reason="stop",
            completion_tokens=1,
        )
        (root / "done.json").write_text(json.dumps(dict(output=str(output))))
        (output / "experiment.json").write_text(json.dumps(config))
        (output / "data-manifest.json").write_text("{}")
        (output / "samples.jsonl").write_text(json.dumps(record) + "\n")
        (output / "results.json").write_text(
            json.dumps(benchmark_eval.metrics([record], task))
        )
    benchmark_eval.summarize()
    rows = {
        r["experiment"]: r
        for r in json.loads((tmp_path / "benchmarks-summary.json").read_text())
    }
    assert "weight_scope" not in rows[names[0]]
    assert rows[names[1]]["weight_scope"] == "embedding"
    assert rows[names[1]]["weight_quantile"] == 0.5
    assert rows[names[2]]["weight_scope"] == "both"
    assert rows[names[2]]["weight_quantile"] == 0.9
    assert all(r["delta_baseline_pp"] == 0 for r in rows.values())
    assert (tmp_path / "benchmarks-summary.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "benchmarks-heatmap.png").read_bytes().startswith(b"\x89PNG")


def test_heatmap_scores_align_tasks_and_keep_pending_distinct_from_zero():
    """Mixed summary units and missing results cannot silently become zero scores."""
    import metrics_heatmap

    rows = [
        dict(experiment="chat_baseline", task="mmlu_stem_chat", accuracy_percent=80),
        dict(
            experiment="chat_weight_embedding_q0.99",
            task="mmlu_stem_chat",
            accuracy_percent=0,
        ),
        dict(experiment="suite_math500_baseline", task="math500", accuracy=0.5),
    ]
    expected = [dict(name="suite_math500_weight_embedding_q0.99", task="math500")]
    arms, tasks, scores, delta = metrics_heatmap.score_matrix(rows, expected)
    i = arms.index("weight_embedding_q0.99")
    assert scores[i, tasks.index("mmlu_stem_chat")] == 0
    assert delta[i, tasks.index("mmlu_stem_chat")] == -80
    assert scores[arms.index("baseline"), tasks.index("math500")] == 50
    assert np.isnan(scores[i, tasks.index("math500")])
    assert np.isnan(delta[i, tasks.index("math500")])


def test_empty_benchmark_summary_saves_pending_heatmap(tmp_path, monkeypatch):
    """A registry with no finished jobs still produces a summary and pending plot."""
    config = next(
        c for c in run.experiment_configs if c["name"] == "suite_math500_baseline"
    )
    monkeypatch.setattr(run, "experiment_configs", [config])
    monkeypatch.setattr(run, "BENCHMARK_CONFIGS", {"math500": {}})
    monkeypatch.setattr(run, "ARTIFACTS", tmp_path)
    benchmark_eval.summarize()
    assert json.loads((tmp_path / "benchmarks-summary.json").read_text()) == []
    assert (tmp_path / "benchmarks-summary.png").read_bytes().startswith(b"\x89PNG")


def paper_module(name):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = (
        (root / "scripts/paper/token_lengths.py")
        if name == "token_lengths"
        else (root / "paper/lint_paper.py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_lengths_include_forced_answer_but_not_injected_suffix():
    module = paper_module("token_lengths")
    config = dict(name="gsm8k_baseline", task="gsm8k_chat", chat_expected_questions=2)
    records = [
        dict(subject="gsm8k", index=0, prompt="aa", reasoning="bbb", mode="natural"),
        dict(
            subject="gsm8k",
            index=1,
            prompt="aaaa",
            reasoning="bb",
            mode="forced_cap",
            forced_text="cc",
            forced_suffix="this injected suffix is not output",
        ),
    ]
    result = module.summarize(records, config, len)
    assert result["mean_input_tokens"] == 3
    assert result["mean_output_text_tokens"] == 3.5
    with pytest.raises(ValueError, match="Incomplete samples"):
        module.summarize(records[:1], config, len)
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        module.summarize([records[0], records[0]], config, len)
    del records[1]["forced_text"]
    with pytest.raises(ValueError, match="Missing forced continuation"):
        module.summarize(records, config, len)


def test_paper_transfer_lengths_count_original_response_and_validate_input():
    module = paper_module("token_lengths")
    config = dict(
        name="suite_ruler_8k_baseline",
        task="ruler_8k",
        benchmark=dict(expected_questions=1),
    )
    record = dict(
        id="task/0",
        prompt="abcd",
        input_tokens=4,
        response="abc",
        completion_tokens=4,
        solution="sanitized code",
    )
    result = module.summarize([record], config, len)
    assert result["mean_output_text_tokens"] == 3
    assert result["mean_input_tokens"] == 4
    record["input_tokens"] = 5
    with pytest.raises(ValueError, match="Input token count mismatch"):
        module.summarize([record], config, len)


def test_paper_matrix_uses_benchmark_columns_and_marks_missing_cells():
    module = paper_module("token_lengths")
    rows = [
        dict(experiment="chat_baseline", task="mmlu_stem_chat", mean_input_tokens=12),
        dict(
            experiment="gsm8k_baseline_repeat", task="gsm8k_chat", mean_input_tokens=34
        ),
    ]
    text = module.render(rows, "mean_input_tokens")
    assert "| Experiment | MMLU STEM | GSM8K |" in text
    assert "| baseline | 12.0 | -- |" in text
    assert "| baseline_repeat | -- | 34.0 |" in text
    tex = module.render(rows, "mean_input_tokens", latex=True)
    assert r"baseline\_repeat" in tex
    assert "34.0" in tex


def test_paper_lint_rejects_missing_assets_raw_tables_and_nan(tmp_path):
    module = paper_module("lint_paper")
    (tmp_path / "article.tex").write_text(
        r"\input{tables/good}"
        "\n"
        r"\includegraphics{missing}"
        "\n"
        r"\begin{tabular}{lr}"
        "\n"
    )
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables/good.tex").write_text("value & nan \\\\\n")
    errors = module.checks(tmp_path)
    assert any("missing includegraphics" in error for error in errors)
    assert any("raw tables" in error for error in errors)
    assert any("NaN" in error for error in errors)
    (tmp_path / "article.tex").write_text(r"\input{tables/good}")
    (tmp_path / "tables/good.tex").write_text("value & 1.0 \\\\\n")
    assert module.checks(tmp_path) == []

    nested = tmp_path / "figures/nested"
    nested.mkdir(parents=True)
    image = nested / "plot.png"
    image.write_bytes(b"fixture")
    assert any("unused attachment" in error for error in module.checks(tmp_path))
    (tmp_path / "article.tex").write_text(r"\input{tables/good}\includegraphics{plot}")
    assert module.checks(tmp_path) == []
    (tmp_path / "figures/plot.png").write_bytes(b"duplicate")
    assert any("duplicate attachment" in error for error in module.checks(tmp_path))


def test_paper_mmlu_rejects_wrong_subject_coverage_despite_complete_count():
    module = paper_module("token_lengths")
    config = dict(
        name="chat_baseline",
        task="mmlu_stem_chat",
        chat_subjects=["anatomy", "astronomy"],
        chat_per_subject=2,
    )
    records = [
        dict(subject=subject, index=index, prompt="aa", reasoning="b", mode="natural")
        for subject in config["chat_subjects"]
        for index in range(2)
    ]
    assert module.summarize(records, config, len)["questions"] == 4
    records[-1]["subject"] = "unexpected_subject"
    with pytest.raises(ValueError, match="MMLU subject coverage differs"):
        module.summarize(records, config, len)
    records[-1]["subject"] = "anatomy"
    records[-1]["index"] = 2
    with pytest.raises(ValueError, match="MMLU subject coverage differs"):
        module.summarize(records, config, len)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.8])
def test_signed_epsilon_preserves_sign_and_reduces_normal_mse(q):
    from rounding_stats import normal_coefficient

    generator = torch.Generator().manual_seed(42)
    original = torch.randn(8, 4096, generator=generator)
    zeroed, replaced = original.clone(), original.clone()
    threshold(zeroed, 0, quantile=q)
    counts = threshold(replaced, 0, quantile=q, replacement="signed_epsilon_normal")
    mask = zeroed == 0
    cutoff = original.abs().kthvalue(math.floor(q * 4096), dim=1).values
    expected = original.sign() * (cutoff * normal_coefficient(q))[:, None]
    assert torch.allclose(replaced[mask], expected[mask])
    assert torch.equal(replaced[~mask], original[~mask])
    assert torch.equal(replaced.sign(), original.sign())
    assert ((replaced - original) ** 2).mean() < ((zeroed - original) ** 2).mean()
    assert torch.all(counts[:, 1] == 0)
    assert torch.equal(counts[:, 6], mask.sum(dim=1))


def test_signed_epsilon_keeps_original_zeros_and_reports_actual_changes():
    from rounding_stats import normal_coefficient

    values = torch.tensor([[0.0, -0.0, -1.0, 1.0, 4.0, -5.0, 6.0, 7.0]])
    counts = threshold(values, 0, quantile=0.5, replacement="signed_epsilon_normal")
    eps = normal_coefficient(0.5)
    assert torch.allclose(values[0, 2:4], torch.tensor([-eps, eps]))
    assert values[0, 0] == 0 and torch.signbit(values[0, 1])
    assert counts.tolist() == [[8, 0, 2, 1, 1, 0, 4, 2]]
    original = values.clone()
    threshold(values, 0, quantile=0, replacement="signed_epsilon_normal")
    assert torch.equal(values, original)
    assert threshold(
        torch.empty(0, 8), 0, quantile=0.5, replacement="signed_epsilon_normal"
    ).shape == (0, 8)


@pytest.mark.parametrize("mode", ["zero", "signed_epsilon_normal"])
def test_mlp_weights_use_independent_output_rows_and_preserve_router(mode):
    from types import SimpleNamespace

    from mlp_weight_rounding import apply
    from rounding_stats import normal_coefficient

    w13 = torch.tensor(
        [
            [
                [0.0, 1.0, -2.0, 3.0],
                [0.0, 10.0, -20.0, 30.0],
                [0.0, 100.0, -200.0, 300.0],
                [0.0, 1000.0, -2000.0, 3000.0],
            ]
        ],
        dtype=torch.bfloat16,
    )
    w2 = torch.tensor(
        [[[1.0, -2.0], [10.0, -20.0], [100.0, -200.0], [1000.0, -2000.0]]],
        dtype=torch.bfloat16,
    )
    router = torch.ones(1, 4)
    experts = SimpleNamespace(w13_weight=w13, w2_weight=w2)
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts=1, moe_intermediate_size=2, hidden_size=4),
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts, gate=router))]
        ),
    )
    original = w13.clone()
    config = dict(quantile=0.5, replacement=mode, batch_rows=1)
    report = apply(model, config)
    eps = normal_coefficient(0.5) if mode != "zero" else 0
    assert torch.equal(w13[0, :, 0], original[0, :, 0])
    assert torch.equal(w13[0, :, 1], (original[0, :, 1].float() * eps).bfloat16())
    assert torch.equal(w13[0, :, 2:], original[0, :, 2:])
    assert router.eq(1).all()
    assert [m["family"] for m in report["matrices"]] == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert all(m["modified"] > 0 and m["mse"] > 0 for m in report["matrices"])


def test_signed_epsilon_registry_inherits_benchmark_protocol_and_reaches_server():
    rows = [
        c
        for c in run.experiment_configs
        if c.get("round_activations_replacement") or c.get("mlp_weight_rounding")
    ]
    assert len(rows) == 42
    assert {c["task"] for c in rows} == set(run.EPSILON_TASKS)
    for row in rows:
        prefix = row["name"].split("mlp_", 1)[0]
        baseline = next(
            c for c in run.experiment_configs if c["name"] == prefix + "baseline"
        )
        for key in ("model", "snapshot", "task", "chat_seed", "chat_concurrency"):
            assert row[key] == baseline[key]
        saved = run.CONFIG.copy()
        try:
            run.select(row["name"])
            args = run.rounding_args()
            actual = dict(
                site="mlp",
                epsilon=0,
                quantile=row["round_activations_quantile"],
                quantiles=None,
                replacement=row.get("round_activations_replacement", "zero"),
                mlp_weight_rounding=row.get("mlp_weight_rounding"),
            )
            assert not run.config_mismatch(actual)
            if row.get("mlp_weight_rounding"):
                assert (
                    json.loads(args[args.index("--mlp-weight-rounding") + 1])
                    == row["mlp_weight_rounding"]
                )
            else:
                assert (
                    args[args.index("--round-activations-replacement") + 1]
                    == "signed_epsilon_normal"
                )
        finally:
            run.CONFIG.clear()
            run.CONFIG.update(saved)


def test_signed_epsilon_sweep_reuses_controls_without_baselines():
    rows = run_jobs.job_configs(signed_epsilon=True)
    assert len(rows) == 56
    assert len({row["name"] for row in rows}) == 56
    assert all(
        not row["name"].endswith(("baseline", "baseline_repeat")) for row in rows
    )
    assert {row["task"] for row in rows} == set(run.EPSILON_TASKS)
    assert len(run_jobs.job_configs("chat_*", signed_epsilon=True)) == 8


def test_signed_epsilon_stats_keep_selection_separate_from_zero_share(tmp_path):
    path = tmp_path / "signed.sqlite"
    with connect(path) as db:
        db.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?)",
            ("r", "r", "[[8, 0, 2, 1, 1, 0, 4, 2]]", 0),
        )
    layer = read_stats(path, "mlp", 0, quantile=0.5)["layers"][0]
    assert layer["newly_zeroed_floats"] == 0
    assert layer["already_zero_floats"] == 2
    assert layer["selected_floats"] == 4
    assert layer["modified_floats"] == 2


def test_signed_epsilon_preserves_nonfinite_values_without_spreading_them():
    values = torch.tensor([[1.0, -2.0, float("inf"), float("nan")]])
    threshold(values, 0, quantile=0.8, replacement="signed_epsilon_normal")
    assert torch.isfinite(values[0, :2]).all()
    assert values[0, 0] > 0 and values[0, 1] < 0
    assert torch.isinf(values[0, 2]) and torch.isnan(values[0, 3])


def test_weight_only_live_validation_skips_original_parity_and_keeps_unique_paths():
    import validate_live

    baseline = dict(site="mlp", epsilon=0.0, quantile=0.0)
    paths = {validate_live.validation_path(baseline)}
    assert not validate_live.model_modified(baseline)
    for replacement, q in itertools.product(
        ("zero", "signed_epsilon_normal"), (0.25, 0.5)
    ):
        config = dict(
            baseline,
            mlp_weight_rounding=dict(
                replacement=replacement, quantile=q, batch_rows=256
            ),
        )
        assert not validate_live.zeroing(config)
        assert validate_live.model_modified(config)
        paths.add(validate_live.validation_path(config))
    assert len(paths) == 5
