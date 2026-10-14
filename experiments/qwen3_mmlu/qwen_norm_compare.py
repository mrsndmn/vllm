# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare only decoder RMSNorm scales of locally cached Qwen checkpoints."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from norm_weight_analysis import layer_similarity, plt
from safetensors import safe_open
from scipy.stats import rankdata

import run

CACHE = Path("/workspace-SR004.nfs2/.cache/huggingface/hub")
LOCAL_CACHE = Path("/home/jovyan/.cache/huggingface/hub")
OUTPUT = run.ARTIFACTS / "weight-viz/qwen-norm-comparison"
FAMILIES = ("input_layernorm", "post_attention_layernorm")
experiment_configs = [
    {
        "model": "Qwen/" + model,
        "snapshot": str(root / ("models--Qwen--" + model) / "snapshots" / revision),
        "prefix": prefix,
        "text_config_path": config_path,
        "weight_offset": offset,
        "tail_fraction": 0.1,
        "persistent_fraction": 0.9,
        "dpi": 160,
    }
    for model, revision, root, prefix, config_path, offset in [
        (
            "Qwen2.5-7B-Instruct",
            "a09a35458c702b33eeacc393d103063234e8bc28",
            CACHE,
            "model.layers",
            [],
            0,
        ),
        (
            "Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
            LOCAL_CACHE,
            "model.layers",
            [],
            0,
        ),
        (
            "Qwen3-8B",
            "b968826d9c46dd6066d109eabc6255188de91218",
            CACHE,
            "model.layers",
            [],
            0,
        ),
        (
            "Qwen3-30B-A3B-Instruct-2507",
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            CACHE,
            "model.layers",
            [],
            0,
        ),
        (
            "Qwen3-30B-A3B-Thinking-2507",
            "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
            CACHE,
            "model.layers",
            [],
            0,
        ),
        (
            "Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            CACHE,
            "model.language_model.layers",
            ["text_config"],
            1,
        ),
        (
            "Qwen2.5-VL-7B-Instruct",
            "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            CACHE,
            "model.layers",
            [],
            0,
        ),
        (
            "Qwen2.5-Omni-7B",
            "ae9e1690543ffd5c0221dc27f79834d0294cba00",
            CACHE,
            "thinker.model.layers",
            ["thinker_config", "text_config"],
            0,
        ),
        (
            "Qwen3-VL-8B-Instruct",
            "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            CACHE,
            "model.language_model.layers",
            ["text_config"],
            0,
        ),
        (
            "Qwen3-VL-30B-A3B-Instruct",
            "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
            CACHE,
            "model.language_model.layers",
            ["text_config"],
            0,
        ),
        (
            "Qwen3-Omni-30B-A3B-Instruct",
            "26291f793822fb6be9555850f06dfe95f2d7e695",
            CACHE,
            "thinker.model.layers",
            ["thinker_config", "text_config"],
            0,
        ),
    ]
]
MOE_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
experiment_configs.insert(
    6,
    dict(
        experiment_configs[5],
        model="Qwen/Qwen3.5-35B-A3B",
        snapshot=str(OUTPUT / "norm-only" / "Qwen3.5-35B-A3B" / MOE_REVISION),
        revision=MOE_REVISION,
        fetch_norms=True,
        max_remote_header_bytes=2_000_000,
        max_remote_metadata_bytes=2_000_000,
    ),
)


def read_scales(config):
    """Read just the two decoder scale vectors per layer, without model loading."""
    if config.get("fetch_norms"):
        from norm_checkpoint_fetch import fetch_norm_checkpoint

        fetch_norm_checkpoint(config, FAMILIES)
    snapshot = Path(config["snapshot"])
    metadata = json.loads((snapshot / "config.json").read_text())
    for key in config["text_config_path"]:
        metadata = metadata[key]
    depth, width = metadata["num_hidden_layers"], metadata["hidden_size"]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    arrays = {family: np.empty((depth, width)) for family in FAMILIES}
    shards = defaultdict(list)
    for family in FAMILIES:
        for layer in range(depth):
            name = f"{config['prefix']}.{layer}.{family}.weight"
            shards[index["weight_map"][name]].append((family, layer, name))
    for shard, names in shards.items():
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            for family, layer, name in names:
                vector = handle.get_tensor(name).float().numpy()
                if vector.shape != (width,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid normalization vector: {name}")
                arrays[family][layer] = vector + config["weight_offset"]
    return arrays


def summarize(weights, config):
    rms = np.sqrt(np.mean(weights**2, axis=1))
    if np.any(rms == 0):
        raise ValueError("Zero-RMS normalization scale")
    cosine, spearman = layer_similarity(weights)
    ranks = (rankdata(np.abs(weights), axis=1) - 1) / (weights.shape[1] - 1)
    high = np.mean(ranks >= 1 - config["tail_fraction"], axis=0)
    low = np.mean(ranks <= config["tail_fraction"], axis=0)
    pairs = spearman[np.triu_indices(len(weights), k=1)]
    relative = weights / rms[:, None]
    rmse = np.sqrt(np.mean(np.diff(relative, axis=0) ** 2, axis=1))
    result = {
        "layers": len(weights),
        "hidden_size": weights.shape[1],
        "persistent_top_count": int(np.sum(high >= config["persistent_fraction"])),
        "persistent_bottom_count": int(np.sum(low >= config["persistent_fraction"])),
        "adjacent_rmse_median": float(np.median(rmse)),
        "rms_max_min_ratio": float(rms.max() / rms.min()),
    }
    for name, values in [
        ("adjacent_cosine", np.diag(cosine, 1)),
        ("adjacent_spearman", np.diag(spearman, 1)),
        ("all_pairs_spearman", pairs),
    ]:
        valid = values[np.isfinite(values)]
        result[name] = (
            {
                "p10": float(np.quantile(valid, 0.1)),
                "median": float(np.median(valid)),
                "p90": float(np.quantile(valid, 0.9)),
            }
            if len(valid)
            else None
        )
        result[name + "_valid_count"] = int(len(valid))
    return result, spearman


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results, matrices = [], []
    for config in experiment_configs:
        weights = read_scales(config)
        row = {"config": config, "families": {}}
        maps = {}
        for family, array in weights.items():
            row["families"][family], maps[family] = summarize(array, config)
        results.append(row)
        matrices.append(maps)
        print(config["model"], row["families"], flush=True)
        name = config["model"].split("/")[-1]
        np.savez_compressed(OUTPUT / f"{name}-effective-scales.npz", **weights)
    (OUTPUT / "comparison.json").write_text(json.dumps(results, indent=2))
    labels = [row["config"]["model"].removeprefix("Qwen/") for row in results]
    y = np.arange(len(labels))
    colours = ["#225588", "#cc7722"]
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 8), sharey=True, constrained_layout=True
    )
    for ax, metric, title in zip(
        axes,
        ["adjacent_cosine", "adjacent_spearman", "all_pairs_spearman"],
        [
            "Neighboring layers: signed cosine",
            "Neighboring layers: magnitude Spearman",
            "Every layer pair: magnitude Spearman",
        ],
    ):
        for family, offset, colour in zip(FAMILIES, [-0.13, 0.13], colours):
            stats = [r["families"][family][metric] for r in results]
            med = np.array([s["median"] if s else np.nan for s in stats])
            lower = np.array([s["p10"] if s else np.nan for s in stats])
            upper = np.array([s["p90"] if s else np.nan for s in stats])
            ax.errorbar(
                med,
                y + offset,
                xerr=[med - lower, upper - med],
                fmt="o",
                color=colour,
                capsize=3,
                label=family,
            )
        limits = (-1.02, 1.02)
        if metric == "adjacent_cosine":
            lower_bound = min(
                r["families"][f][metric]["p10"] for r in results for f in FAMILIES
            )
            limits = (lower_bound - 0.02, 1.005)
        ax.set(title=title, xlabel="Median and 10–90% of layer pairs", xlim=limits)
        ax.grid(alpha=0.15)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Qwen decoder norms: cached checkpoints + norm-only Qwen3.5 MoE subset\n"
        "Qwen3.5 uses 1 + checkpoint weight; others use checkpoint weight"
    )
    fig.savefig(OUTPUT / "01-similarity-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True, constrained_layout=True
    )
    for ax, family in zip(axes, FAMILIES):
        for key, offset, colour, label in [
            ("persistent_top_count", -0.16, "#228866", "Persistently in top 10%"),
            ("persistent_bottom_count", 0.16, "#994477", "Persistently in bottom 10%"),
        ]:
            percent = [
                100 * r["families"][family][key] / r["families"][family]["hidden_size"]
                for r in results
            ]
            ax.barh(y + offset, percent, height=0.3, color=colour, label=label)
        ax.set(title=family, xlabel="Share of hidden dimensions (%)")
        ax.grid(axis="x", alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Persistent high/low decoder norm coordinates across Qwens\n"
        "Top/bottom decile within each layer in at least 90% of layers; "
        "average tie ranks"
    )
    fig.savefig(OUTPUT / "02-persistence-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for family in FAMILIES:
        fig, axes = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True)
        for ax, label, maps in zip(axes.flat, labels, matrices):
            im = ax.imshow(maps[family], vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set(title=label, xlabel="Layer", ylabel="Layer")
        for ax in list(axes.flat)[len(labels) :]:
            ax.set_visible(False)
        fig.colorbar(
            im, ax=list(axes.flat), shrink=0.8, label="Spearman of effective |scales|"
        )
        fig.suptitle(
            f"{family}: all-layer coordinate-order similarity\n"
            "Language decoder only; no vision, audio, talker, Q/K or MTP norms"
        )
        fig.savefig(OUTPUT / f"03-{family}-rank-maps.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
