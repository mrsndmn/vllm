# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare decoder norm scales from cached and norm-only Qwen/OLMo checkpoints."""

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
ROLES = ("Attention branch norm", "FFN branch norm")
experiment_configs = [
    {
        "model": "Qwen/" + model,
        "snapshot": str(root / ("models--Qwen--" + model) / "snapshots" / revision),
        "prefix": prefix,
        "text_config_path": config_path,
        "weight_offset": offset,
        "families": FAMILIES,
        "norm_layout": "pre",
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
experiment_configs.extend(
    dict(
        experiment_configs[0],
        model=model,
        snapshot=str(OUTPUT / "norm-only" / model.split("/")[-1] / revision),
        revision=revision,
        families=families,
        norm_layout=layout,
        fetch_norms=True,
        max_remote_header_bytes=2_000_000,
        max_remote_metadata_bytes=2_000_000,
    )
    for model, revision, families, layout in [
        (
            "allenai/OLMoE-1B-7B-0924",
            "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
            FAMILIES,
            "pre",
        ),
        (
            "allenai/OLMo-2-1124-7B",
            "7df9a82518afdecae4e8c026b27adccc8c1f0032",
            ("post_attention_layernorm", "post_feedforward_layernorm"),
            "post",
        ),
        (
            "allenai/OLMo-2-0325-32B",
            "cc9d3cf9c7230b86ee6b84607b37db1c01e3f1ed",
            ("post_attention_layernorm", "post_feedforward_layernorm"),
            "post",
        ),
        (
            "allenai/Olmo-3-7B-Instruct",
            "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
            ("post_attention_layernorm", "post_feedforward_layernorm"),
            "post",
        ),
    ]
)


def norm_families(config):
    """Return attention/FFN norm names in role order, retaining their actual names."""
    return tuple(config.get("families", FAMILIES))


def role_stats(row, role):
    return row["families"][norm_families(row["config"])[role]]


def read_scales(config):
    """Read just the two decoder scale vectors per layer, without model loading."""
    families = norm_families(config)
    if config.get("fetch_norms"):
        from norm_checkpoint_fetch import fetch_norm_checkpoint

        fetch_norm_checkpoint(config, families)
    snapshot = Path(config["snapshot"])
    metadata = json.loads((snapshot / "config.json").read_text())
    for key in config["text_config_path"]:
        metadata = metadata[key]
    depth, width = metadata["num_hidden_layers"], metadata["hidden_size"]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    arrays = {family: np.empty((depth, width)) for family in families}
    shards = defaultdict(list)
    for family in families:
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
    labels = [
        r["config"]["model"].split("/")[-1] + f" [{r['config']['norm_layout']}]"
        for r in results
    ]
    y = np.arange(len(labels))
    colours = ["#225588", "#cc7722"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, max(8, len(labels) * 0.6)),
        sharey=True,
        constrained_layout=True,
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
        for role, offset, colour in zip(range(2), [-0.13, 0.13], colours):
            stats = [role_stats(r, role)[metric] for r in results]
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
                label=ROLES[role],
            )
        limits = (-1.02, 1.02)
        if metric == "adjacent_cosine":
            lower_bound = min(
                role_stats(r, role)[metric]["p10"] for r in results for role in range(2)
            )
            limits = (lower_bound - 0.02, 1.005)
        ax.set(title=title, xlabel="Median and 10–90% of layer pairs", xlim=limits)
        ax.grid(alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Qwen + OLMo decoder norm scales: local weights and norm-only subsets\n"
        "[pre]: before branch; [post]: branch output before residual addition\n"
        "Qwen3.5 uses 1 + checkpoint weight; other models use checkpoint weight"
    )
    fig.savefig(OUTPUT / "01-similarity-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(8, len(labels) * 0.6)),
        sharey=True,
        constrained_layout=True,
    )
    for role, ax in enumerate(axes):
        for key, offset, colour, label in [
            ("persistent_top_count", -0.16, "#228866", "Persistently in top 10%"),
            ("persistent_bottom_count", 0.16, "#994477", "Persistently in bottom 10%"),
        ]:
            percent = [
                100 * role_stats(r, role)[key] / role_stats(r, role)["hidden_size"]
                for r in results
            ]
            ax.barh(y + offset, percent, height=0.3, color=colour, label=label)
        ax.set(title=ROLES[role], xlabel="Share of hidden dimensions (%)")
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
        "Persistent high/low decoder norm coordinates: Qwen + OLMo\n"
        "Top/bottom decile within each layer in at least 90% of layers; "
        "average tie ranks\n"
        "[pre]: before branch; [post]: branch output before residual addition"
    )
    fig.savefig(OUTPUT / "02-persistence-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rows = (len(labels) + 2) // 3
    for role in range(2):
        fig, axes = plt.subplots(
            rows, 3, figsize=(14, rows * 4), constrained_layout=True, squeeze=False
        )
        for ax, label, maps, result in zip(axes.flat, labels, matrices, results):
            family = norm_families(result["config"])[role]
            im = ax.imshow(maps[family], vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set(title=label + "\n" + family, xlabel="Layer", ylabel="Layer")
        for ax in list(axes.flat)[len(labels) :]:
            ax.set_visible(False)
        fig.colorbar(
            im, ax=list(axes.flat), shrink=0.8, label="Spearman of effective |scales|"
        )
        fig.suptitle(
            f"{ROLES[role]}: all-layer coordinate-order similarity\n"
            "[pre]: before branch; [post]: branch output before residual addition\n"
            "Language decoder only; no vision, audio, talker, Q/K or MTP norms"
        )
        name = "attention_norm" if role == 0 else "ffn_norm"
        fig.savefig(OUTPUT / f"03-{name}-rank-maps.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
