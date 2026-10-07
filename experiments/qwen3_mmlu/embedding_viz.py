# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare sampled English-token embedding and unembedding checkpoint weights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pybase64 as base64
import regex as re
import torch
from plotly.offline import get_plotlyjs
from safetensors import safe_open
from transformers import AutoTokenizer

import run

experiment_configs = [
    {
        "name": "qwen3_english_embeddings",
        "model": run.CONFIG["model"],
        "snapshot": run.CONFIG["snapshot"],
        "output": str(run.ARTIFACTS / "embedding-viz"),
        "dictionary": str(
            run.ARTIFACTS
            / "runtime/ruler-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
            / "scripts/data/synthetic/json/english_words.json"
        ),
        "dictionary_sha256": run.RULER_CORPUS_SHA256["english_words.json"],
        "tokens": 1000,
        "seed": 42,
        "min_letters": 3,
        "bins": 160,
        "heatmap_channel_groups": 64,
        "cpu_threads": 4,
    }
]


def summarize(x, axis=None):
    """Population statistics of all values or of each hidden channel."""
    x = x.astype(np.float64)
    mean = x.mean(axis=axis)
    std = x.std(axis=axis)
    centered = x - (mean if axis is None else mean[None, :])
    kurt = (
        np.divide(
            (centered**4).mean(axis=axis),
            std**4,
            out=np.zeros_like(std),
            where=std > 0,
        )
        - 3
    )
    kurt = np.where(std > 0, kurt, np.nan)
    result = {
        "mean": mean,
        "std": std,
        "rms": np.sqrt((x * x).mean(axis=axis)),
        "min": x.min(axis=axis),
        "max": x.max(axis=axis),
        "zero_share": (x == 0).mean(axis=axis),
        "negative_share": (x < 0).mean(axis=axis),
        "excess_kurtosis": kurt,
        "outside_3std_share": (np.abs(centered) > 3 * std).mean(axis=axis),
    }
    for q in [0.5, 0.9, 0.99, 0.999]:
        result[f"abs_q{q:g}"] = np.quantile(np.abs(x), q, axis=axis)
    if axis == 0:
        energy = (x * x).sum(axis=0)
        result["energy_share"] = energy / energy.sum()
    return {k: v.tolist() for k, v in result.items()}


def cosines(a, b, axis):
    """Cosines, with undefined zero-norm comparisons represented as NaN."""
    numerator = (a * b).sum(axis=axis)
    denom = np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    return np.divide(
        numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 0
    )


def select_tokens(config, snapshot):
    """Uniformly sample distinct token IDs passing a pinned lexical filter."""
    raw = Path(config["dictionary"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config["dictionary_sha256"]
    words = {w.lower() for w in json.loads(raw).values()}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    special = set(tokenizer.all_special_ids)
    candidates = []
    pattern = rf" ?[A-Za-z]{{{config['min_letters']},}}"
    for token_id in sorted(set(tokenizer.get_vocab().values()) - special):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if re.fullmatch(pattern, text) and text.strip().lower() in words:
            candidates.append((token_id, text))
    if len(candidates) < config["tokens"]:
        raise ValueError("Not enough English-token candidates")
    rng = np.random.default_rng(config["seed"])
    selected = rng.choice(len(candidates), config["tokens"], replace=False)
    return [candidates[i] for i in selected], len(candidates), tokenizer


def analyze(config):
    """Read only selected vocabulary rows and write reproducible offline artifacts."""
    torch.set_num_threads(config["cpu_threads"])
    snapshot = Path(config["snapshot"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    tokens, pool_size, tokenizer = select_tokens(config, snapshot)
    ids = [t[0] for t in tokens]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    model_config = json.loads((snapshot / "config.json").read_text())
    assert model_config["tie_word_embeddings"] is False
    matrices, encoded, shapes, source = {}, {}, {}, {}
    for label, name in [
        ("embedding", "model.embed_tokens.weight"),
        ("unembedding", "lm_head.weight"),
    ]:
        shard = snapshot / index["weight_map"][name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            shapes[label] = view.get_shape()
            assert shapes[label][1] == model_config["hidden_size"]
            assert max(ids) < shapes[label][0]
            rows = torch.stack([view[i, :] for i in ids])
        assert rows.dtype == torch.bfloat16
        bits = rows.view(torch.uint16).numpy().astype("<u2").tobytes()
        encoded[label] = base64.b64encode(bits).decode("ascii")
        matrices[label] = rows.float().numpy()
        assert np.isfinite(matrices[label]).all()
        source[label] = {
            "tensor": name,
            "shard": shard.name,
            "resolved_shard": str(shard.resolve()),
            "selected_bf16_sha256": hashlib.sha256(bits).hexdigest(),
        }
    e, u = [matrices[k].astype(np.float64) for k in matrices]
    token_cos = cosines(e, u, 1)
    channel_cos = cosines(e, u, 0)
    channel_pearson = cosines(e - e.mean(0), u - u.mean(0), 0)
    stats = {k: summarize(x) for k, x in matrices.items()}
    channels = {k: summarize(x, axis=0) for k, x in matrices.items()}
    histograms = {}
    reach = max(abs(x).max() for x in matrices.values())
    edges = np.linspace(-reach, reach, config["bins"] + 1)
    positive_min = min(np.abs(x[x != 0]).min() for x in matrices.values())
    log_edges = np.linspace(
        np.floor(np.log10(float(positive_min))),
        np.ceil(np.log10(float(reach))),
        config["bins"] + 1,
    )
    for label, x in matrices.items():
        positive = np.abs(x[x != 0])
        assert not len(positive) or np.log10(positive.min()) >= log_edges[0]
        histograms[label] = {
            "signed": np.histogram(x, edges)[0].tolist(),
            "log_abs": np.histogram(np.log10(positive), log_edges)[0].tolist(),
        }
        assert sum(histograms[label]["signed"]) == x.size
        assert sum(histograms[label]["log_abs"]) + (x == 0).sum() == x.size
    groups = config["heatmap_channel_groups"]
    width = e.shape[1]
    assert width % groups == 0
    heatmaps = {
        k: np.sqrt((x.reshape(len(ids), groups, -1) ** 2).mean(2)).tolist()
        for k, x in matrices.items()
    }
    summary = {
        "global": stats,
        "token_cosine_quantiles": np.quantile(
            token_cos, [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "channel_pearson_quantiles": np.quantile(
            channel_pearson, [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
    }
    for label in matrices:
        energy = np.array(channels[label]["energy_share"])
        order = np.argsort(energy)[::-1]
        summary[label + "_energy"] = {
            "top_1_share": float(energy[order[:1]].sum()),
            "top_10_share": float(energy[order[:10]].sum()),
            "top_1pct_share": float(energy[order[: int(np.ceil(width * 0.01))]].sum()),
            "top_channels": order[:10].tolist(),
            "rms_max_to_median": float(
                max(channels[label]["rms"]) / np.median(channels[label]["rms"])
            ),
            "mean_energy_fraction": float(
                np.square(channels[label]["mean"]).sum()
                / (matrices[label] ** 2).mean(0).sum()
            ),
        }
    data = {
        "model": config["model"],
        "revision": snapshot.name,
        "config": config,
        "pool_size": pool_size,
        "tokens": tokens,
        "width": width,
        "shapes": shapes,
        "source": source,
        "summary": summary,
        "channels": channels,
        "token_cosines": token_cos.tolist(),
        "channel_cosines": channel_cos.tolist(),
        "channel_pearson": channel_pearson.tolist(),
        "token_norms": {
            k: np.linalg.norm(x, axis=1).tolist() for k, x in matrices.items()
        },
        "edges": edges.tolist(),
        "log_edges": log_edges.tolist(),
        "histograms": histograms,
        "heatmaps": heatmaps,
        "bf16": encoded,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "provenance.json").write_text(
        json.dumps(
            {
                k: data[k]
                for k in [
                    "model",
                    "revision",
                    "config",
                    "pool_size",
                    "shapes",
                    "source",
                ]
            },
            indent=2,
        )
    )
    with (output / "tokens.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_row",
                "token_id",
                "decoded",
                "cosine",
                "embedding_norm",
                "unembedding_norm",
            ]
        )
        for i, (token_id, text) in enumerate(tokens):
            assert (
                tokenizer.decode([token_id], clean_up_tokenization_spaces=False) == text
            )
            writer.writerow(
                [
                    i,
                    token_id,
                    text,
                    token_cos[i],
                    data["token_norms"]["embedding"][i],
                    data["token_norms"]["unembedding"][i],
                ]
            )
    with (output / "channels.csv").open("w") as f:
        writer = csv.writer(f)
        keys = list(channels["embedding"])
        writer.writerow(["matrix", "channel", *keys, "paired_cosine", "paired_pearson"])
        for label in matrices:
            for i in range(width):
                writer.writerow(
                    [
                        label,
                        i,
                        *[channels[label][k][i] for k in keys],
                        channel_cos[i],
                        channel_pearson[i],
                    ]
                )
    np.savez_compressed(output / "selected_weights.npz", token_ids=ids, **matrices)
    template = Path(__file__).with_suffix(".html").read_text()
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("</", "<\\/")
    html = template.replace(
        "%%PLOTLY%%", get_plotlyjs().replace("</script", "<\\/script")
    )
    html = html.replace("%%DATA%%", payload)
    (output / "embeddings.html").write_text(html)
    print(
        json.dumps(
            {"output": str(output), "pool_size": pool_size, "summary": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    analyze(experiment_configs[0])
