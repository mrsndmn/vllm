# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static diagnostics of decoder norm scales from an offline weight report."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np
import pybase64 as base64
from matplotlib.colors import LogNorm
from scipy.stats import rankdata

import run

matplotlib.use("Agg")
import matplotlib.pyplot as plt

experiment_configs = [
    {
        "name": "qwen3_norm_weights",
        "report": str(
            Path(__file__).resolve().parents[2] / "run/weight-viz/weights.html"
        ),
        "output": str(run.ARTIFACTS / "weight-viz/norm-analysis"),
        "tail_fraction": 0.1,
        "persistent_fraction": 0.9,
        "heatmap_quantile": 0.99,
        "band_quantiles": [0.1, 0.9],
        "dpi": 160,
    }
]
experiment_configs.append(
    {
        "name": "qwen3_post_attention_distributions",
        "analysis": "distributions",
        "report": experiment_configs[0]["report"],
        "output": str(run.ARTIFACTS / "weight-viz/post-attention-distributions"),
        "family": "post_attention_layernorm",
        "layer_range": [4, 44],
        "histogram_bins": 128,
        "quantiles": [0.01, 0.1, 0.5, 0.9, 0.99],
        "example_layers": [4, 14, 24, 34, 44],
        "dpi": 160,
    }
)
experiment_configs.append(
    {
        "name": "qwen3_post_attention_large_channels",
        "analysis": "channels",
        "report": experiment_configs[0]["report"],
        "output": str(run.ARTIFACTS / "weight-viz/post-attention-large-channels"),
        "family": "post_attention_layernorm",
        "layer_range": [4, 44],
        "relative_threshold": 1.0,
        "dpi": 160,
    }
)


def load_norms(report):
    html = Path(report).read_text()
    encoded = html.split('<script id="payload" type="application/octet-stream">')[1]
    data = json.loads(gzip.decompress(base64.b64decode(encoded.split("</script>")[0])))
    norms = {}
    for family, norm in data["norm_weights"].items():
        weights = np.frombuffer(base64.b64decode(norm["values"]), dtype="<f4")
        weights = weights.reshape(len(norm["layers"]), norm["width"]).astype(float)
        rms = np.sqrt(np.mean(weights**2, axis=1))
        if not np.isfinite(weights).all() or np.any(rms == 0):
            raise ValueError(f"Non-finite or zero-RMS weights: {family}")
        norms[family] = (np.array(norm["layers"]), weights, rms, weights / rms[:, None])
    if not norms:
        raise ValueError("Report has no decoder norm weights")
    return data["config"]["model"], norms


def layer_similarity(weights):
    """Return scale-free signed cosine and magnitude-rank correlation matrices."""
    unit = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    ranks = rankdata(np.abs(weights), axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(ranks, axis=1, keepdims=True)
    ranks = np.divide(
        ranks, lengths, out=np.full_like(ranks, np.nan), where=lengths > 0
    )
    return np.clip(unit @ unit.T, -1, 1), np.clip(ranks @ ranks.T, -1, 1)


def plot(config):
    """Write six figures and exact, unclipped summary statistics."""
    model, norms = load_norms(config["report"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    count = len(norms)

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=config["dpi"], bbox_inches="tight")
        plt.close(fig)

    limit = float(
        np.quantile(
            np.concatenate([np.abs(v[3]).ravel() for v in norms.values()]),
            config["heatmap_quantile"],
        )
    )
    fig, axes = plt.subplots(
        count, 1, figsize=(14, 4 * count), squeeze=False, constrained_layout=True
    )
    for ax, (family, (layers, weights, rms, relative)) in zip(
        axes[:, 0], norms.items()
    ):
        image = ax.imshow(
            relative,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ticks = np.arange(0, len(layers), max(1, len(layers) // 12))
        ax.set_yticks(ticks, layers[ticks])
        ax.set(
            title=family, xlabel="Hidden dimension (0-based)", ylabel="Decoder layer"
        )
        fig.colorbar(image, ax=ax, label="Signed weight / row weight RMS")
    fig.suptitle(
        f"{model}\nRelative norm weights; colour clipped at the shared "
        f"{config['heatmap_quantile']:.0%} |weight/RMS| quantile ({limit:.2f})"
    )
    save(fig, "01-relative-heatmaps")

    fig, axes = plt.subplots(
        count, 1, figsize=(14, 3.6 * count), squeeze=False, constrained_layout=True
    )
    for ax, (family, (_, _, _, relative)) in zip(axes[:, 0], norms.items()):
        magnitude = np.abs(relative)
        lo, hi = np.quantile(magnitude, config["band_quantiles"], axis=0)
        median = np.median(magnitude, axis=0)
        dims = np.arange(len(median))
        ax.fill_between(
            dims, lo, hi, color="#4477aa", alpha=0.25, label="10–90% across layers"
        )
        ax.plot(
            dims, median, color="#225588", linewidth=0.8, label="Median across layers"
        )
        ax.axhline(
            1, color="#777777", linestyle="--", linewidth=0.8, label="Row weight RMS"
        )
        ax.set(
            title=family,
            xlabel="Hidden dimension (0-based)",
            ylabel="|weight| / row RMS",
            yscale="log",
        )
        ax.legend(loc="upper right")
        ax.grid(alpha=0.15)
    fig.suptitle(
        f"{model}\nCoordinate profiles: persistent amplification and suppression"
    )
    save(fig, "02-coordinate-profiles")

    fig, axes = plt.subplots(
        1, count, figsize=(7 * count, 5.8), squeeze=False, constrained_layout=True
    )
    summary = {"model": model, "report": config["report"], "families": {}}
    for ax, (family, (layers, weights, rms, relative)) in zip(axes[0], norms.items()):
        ranks = (rankdata(np.abs(weights), axis=1) - 1) / (weights.shape[1] - 1)
        high = np.mean(ranks >= 1 - config["tail_fraction"], axis=0)
        low = np.mean(ranks <= config["tail_fraction"], axis=0)
        median = np.median(np.abs(relative), axis=0)
        ax.scatter(high * 100, low * 100, s=12, alpha=0.2, color="#4477aa")
        annotations = {}
        for dim in np.unique(
            [np.argmax(high), np.argmax(low), np.argmax(median), np.argmin(median)]
        ):
            ax.scatter(high[dim] * 100, low[dim] * 100, s=35, color="#bb3322")
            position = (high[dim] * 100, low[dim] * 100)
            annotations.setdefault(position, []).append(str(dim))
        for position, dimensions in annotations.items():
            ax.annotate(
                ", ".join(dimensions),
                position,
                xytext=(5, 5),
                textcoords="offset points",
            )
        threshold = config["persistent_fraction"] * 100
        ax.axvline(threshold, color="#777777", linestyle="--")
        ax.axhline(threshold, color="#777777", linestyle="--")
        ax.set(
            title=family,
            xlabel="Layers in top 10% of |weights| (%)",
            ylabel="Layers in bottom 10% of |weights| (%)",
            xlim=(-3, 110),
            ylim=(-3, 110),
        )
        ax.grid(alpha=0.15)
        summary["families"][family] = {
            "layers": len(layers),
            "hidden_size": weights.shape[1],
            "persistent_top_decile": int(np.sum(high >= config["persistent_fraction"])),
            "persistent_bottom_decile": int(
                np.sum(low >= config["persistent_fraction"])
            ),
            "negative_share": float(np.mean(weights < 0)),
            "zero_share": float(np.mean(weights == 0)),
            "rms_min": float(rms.min()),
            "rms_max": float(rms.max()),
        }
    fig.suptitle(
        f"{model}\nPersistence by coordinate: one point per hidden dimension; "
        "annotated numbers are dimension indices\n"
        "Ranks use |weights| within each layer; ties receive average ranks"
    )
    save(fig, "03-coordinate-persistence")

    fig, axes = plt.subplots(
        count, 2, figsize=(14, 3.5 * count), squeeze=False, constrained_layout=True
    )
    for row, (family, (layers, weights, rms, relative)) in enumerate(norms.items()):
        ax = axes[row, 0]
        ax.plot(layers, rms, label="RMS", color="#225588")
        ax.plot(
            layers,
            np.mean(np.abs(weights), axis=1),
            label="Mean |weight|",
            color="#cc7722",
        )
        ax.set(title=family, xlabel="Decoder layer", ylabel="Original weight scale")
        ax.legend()
        ax = axes[row, 1]
        k = int(np.ceil(weights.shape[1] * config["tail_fraction"]))
        energy = np.sort(weights**2, axis=1)[:, -k:].sum(axis=1) / np.sum(
            weights**2, axis=1
        )
        ax.plot(layers, energy * 100, color="#994477")
        ax.axhline(
            config["tail_fraction"] * 100,
            color="#777777",
            linestyle="--",
            label="Equal-magnitude weights",
        )
        ax.set(
            title=family,
            xlabel="Decoder layer",
            ylabel="Squared-weight share of top 10% (%)",
        )
        ax.legend()
        summary["families"][family]["top_decile_squared_weight_share_median"] = float(
            np.median(energy)
        )
        for ax in axes[row]:
            ax.grid(alpha=0.15)
    fig.suptitle(
        f"{model}\nLayer scale and concentration of norm weights\n"
        "Squared-weight share describes parameters, not activation or output energy"
    )
    save(fig, "04-layer-scale-concentration")

    fig, axes = plt.subplots(
        count, 3, figsize=(16, 3.6 * count), squeeze=False, constrained_layout=True
    )
    matrices = {}
    for row, (family, (layers, weights, rms, relative)) in enumerate(norms.items()):
        cosine, spearman = layer_similarity(weights)
        matrices[family] = (layers, spearman)
        adjacent = np.diff(layers) == 1
        x = layers[1:][adjacent]
        cos = np.diag(cosine, k=1)[adjacent]
        rho = np.diag(spearman, k=1)[adjacent]
        difference = np.sqrt(np.mean(np.diff(relative, axis=0) ** 2, axis=1))[adjacent]
        ratio = (rms[1:] / rms[:-1])[adjacent]
        ax = axes[row, 0]
        ax.plot(x, cos, label="Signed cosine", color="#225588")
        ax.plot(x, rho, label="Spearman of |weights|", color="#cc7722")
        ax.set(title=family, ylabel="Similarity (1 = identical profile)")
        ax.legend()
        ax = axes[row, 1]
        ax.plot(x, difference, color="#994477")
        ax.set(title=family, ylabel="RMS difference of weight / row RMS")
        ax = axes[row, 2]
        ax.plot(x, ratio, color="#228866")
        ax.axhline(1, color="#777777", linestyle="--")
        ax.set(title=family, ylabel="RMS(layer l) / RMS(layer l−1)", yscale="log")
        for ax in axes[row]:
            ax.set_xlabel("Layer l: comparison of l−1 with l")
            ax.grid(alpha=0.15)
        summary["families"][family]["adjacent_pairs"] = [
            {
                "layers": [int(layer - 1), int(layer)],
                "cosine": float(c),
                "magnitude_spearman": float(s) if np.isfinite(s) else None,
                "relative_rmse": float(d),
                "rms_ratio": float(r),
            }
            for layer, c, s, d, r in zip(x, cos, rho, difference, ratio)
        ]
    fig.suptitle(
        f"{model}\nNeighboring layer norm weights: profile vs overall scale\n"
        "Relative RMS difference = √(2−2·cosine); Spearman compares coordinate ordering"
    )
    save(fig, "05-adjacent-layer-similarity")

    fig, axes = plt.subplots(
        1, count, figsize=(7 * count, 6), squeeze=False, constrained_layout=True
    )
    for ax, (family, (layers, matrix)) in zip(axes[0], matrices.items()):
        image = ax.imshow(
            matrix, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest"
        )
        ticks = np.arange(0, len(layers), max(1, len(layers) // 12))
        ax.set_xticks(ticks, layers[ticks])
        ax.set_yticks(ticks, layers[ticks])
        ax.set(title=family, xlabel="Decoder layer", ylabel="Decoder layer")
        fig.colorbar(image, ax=ax, label="Spearman correlation of |weights|")
    fig.suptitle(
        f"{model}\nCoordinate-order similarity across every layer pair\n"
        "Cells next to the diagonal compare neighbors; "
        "constant-magnitude rows are undefined"
    )
    save(fig, "06-layer-rank-similarity")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def plot_distributions(config):
    """Plot full-range per-layer distributions, quantiles and pooled counts."""
    model, norms = load_norms(config["report"])
    layers, weights, rms, relative = norms[config["family"]]
    first, last = config["layer_range"]
    selected = (layers >= first) & (layers <= last)
    layers, weights, rms, relative = (
        a[selected] for a in (layers, weights, rms, relative)
    )
    if not np.array_equal(layers, np.arange(first, last + 1)):
        raise ValueError("Requested layer range has missing normalization tensors")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    heading = (
        f"{model}\n{config['family']} · layers {first}–{last} inclusive "
        f"· D = {weights.shape[1]}"
    )
    series = [
        (weights, "Original signed weight"),
        (relative, "Signed weight / layer weight RMS"),
    ]
    histograms = []
    for values, label in series:
        edges = np.linspace(values.min(), values.max(), config["histogram_bins"] + 1)
        counts = np.array([np.histogram(row, bins=edges)[0] for row in values])
        if not np.all(counts.sum(axis=1) == weights.shape[1]):
            raise RuntimeError("Histogram does not include every weight")
        histograms.append((edges, counts / weights.shape[1], label))

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=config["dpi"], bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, (edges, probabilities, label) in zip(axes, histograms):
        image = ax.imshow(
            np.ma.masked_equal(probabilities, 0),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            norm=LogNorm(vmin=1 / weights.shape[1], vmax=probabilities.max()),
            extent=[edges[0], edges[-1], last + 0.5, first - 0.5],
        )
        ax.set(
            xlabel=label,
            ylabel="Decoder layer",
            title="Per-layer histograms; shared bins",
        )
        ax.set_yticks(np.arange(first, last + 1, 4))
        fig.colorbar(
            image, ax=ax, label="Fraction of layer weights per bin (log colour)"
        )
    fig.suptitle(
        heading
        + "\nEvery row sums to 100%; white = empty bin; full weight range included"
    )
    save(fig, "01-layer-distributions")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    quantiles = []
    for ax, (values, label) in zip(axes, series):
        q = np.quantile(values, config["quantiles"], axis=1)
        quantiles.append(q)
        ax.fill_between(layers, q[0], q[4], alpha=0.15, color="#4477aa", label="1–99%")
        ax.fill_between(layers, q[1], q[3], alpha=0.3, color="#4477aa", label="10–90%")
        ax.plot(layers, q[2], color="#225588", label="Median")
        ax.set(
            xlabel="Decoder layer",
            ylabel=label,
            title="Exact quantiles across all hidden coordinates",
        )
        ax.legend()
        ax.grid(alpha=0.15)
    axes[1].axhline(1, color="#777", linestyle="--", linewidth=0.8)
    fig.suptitle(heading)
    save(fig, "02-layer-quantiles")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, (edges, probabilities, label) in zip(axes, histograms):
        ax.stairs(
            probabilities.mean(axis=0) * 100,
            edges,
            color="black",
            linewidth=2,
            label=f"All {len(layers)} layers pooled",
        )
        for layer in config["example_layers"]:
            position = int(np.flatnonzero(layers == layer)[0])
            ax.stairs(
                probabilities[position] * 100, edges, alpha=0.8, label=f"Layer {layer}"
            )
        ax.set(
            xlabel=label,
            ylabel="Weight share per bin (%)",
            title="Selected layers vs pooled distribution",
        )
        ax.legend()
        ax.grid(alpha=0.15)
    fig.suptitle(
        heading + "\nEqual-width bins; all weights counted; each curve sums to 100%"
    )
    save(fig, "03-distribution-comparison")
    summary = {
        "model": model,
        "family": config["family"],
        "layer_range": [first, last],
        "layers": len(layers),
        "weights_per_layer": weights.shape[1],
        "total_weights": weights.size,
        "negative_count": int(np.sum(weights < 0)),
        "zero_count": int(np.sum(weights == 0)),
        "raw_min": float(weights.min()),
        "raw_max": float(weights.max()),
        "raw_pooled_quantiles": np.quantile(weights, config["quantiles"]).tolist(),
        "quantiles": config["quantiles"],
        "per_layer_raw_quantiles": quantiles[0].T.tolist(),
        "per_layer_relative_quantiles": quantiles[1].T.tolist(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def channel_occupancy(relative, threshold):
    """Count layers per channel using the inclusive signed weight threshold."""
    return np.sum(relative >= threshold, axis=0)


def plot_channels(config):
    """Plot threshold occupancy by channel and its discrete distribution."""
    model, norms = load_norms(config["report"])
    layers, _, _, relative = norms[config["family"]]
    first, last = config["layer_range"]
    selected = (layers >= first) & (layers <= last)
    layers, relative = layers[selected], relative[selected]
    if not np.array_equal(layers, np.arange(first, last + 1)):
        raise ValueError("Requested layer range has missing normalization tensors")
    threshold = config["relative_threshold"]
    counts = channel_occupancy(relative, threshold)
    common = counts == len(layers)
    union = counts > 0
    per_layer = np.sum(relative >= threshold, axis=1)
    histogram = np.bincount(counts, minlength=len(layers) + 1)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model,
        "family": config["family"],
        "layer_range": [first, last],
        "layer_count": len(layers),
        "total_channels": len(counts),
        "signed_relative_threshold": threshold,
        "channels_in_any_layer": int(union.sum()),
        "channels_in_every_layer": int(common.sum()),
        "common_share_of_all_channels": float(common.mean()),
        "common_share_of_union": float(common.sum() / union.sum())
        if union.any()
        else None,
        "channel_layer_hits": int(counts.sum()),
        "per_layer_counts": per_layer.tolist(),
        "occupancy_histogram": histogram.tolist(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(
        output / "channel-occupancy.csv",
        np.column_stack([np.arange(len(counts)), counts, common]),
        delimiter=",",
        fmt="%d",
        header="channel,layers_at_or_above_threshold,in_every_layer",
        comments="",
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    channels = np.arange(len(counts))
    axes[0].scatter(
        channels[~common], counts[~common], s=7, alpha=0.45, color="#4477aa"
    )
    axes[0].scatter(
        channels[common],
        counts[common],
        s=10,
        color="#bb3322",
        label=f"All {len(layers)} layers: {common.sum()} channels",
    )
    axes[0].axhline(len(layers), color="#bb3322", linestyle="--", linewidth=0.7)
    axes[0].set(
        xlabel="Hidden channel (0-based)",
        ylabel="Number of layers with signed weight / RMS ≥ threshold",
        title="Occupancy of each hidden channel",
        ylim=(-1, len(layers) + 2),
    )
    axes[0].legend(loc="lower right")
    colours = ["#4477aa"] * len(histogram)
    colours[-1] = "#bb3322"
    axes[1].bar(np.arange(len(histogram)), histogram, color=colours)
    axes[1].set(
        xlabel="Number of qualifying layers",
        ylabel="Number of channels",
        title=f"Discrete distribution across all {len(counts)} channels",
    )
    axes[1].annotate(
        str(common.sum()),
        (len(layers), histogram[-1]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        f"{model}\n{config['family']} · layers {first}–{last} inclusive "
        f"· signed weight / layer RMS ≥ {threshold:g}\n"
        f"Union: {union.sum()} channels; intersection: {common.sum()} "
        f"({common.mean():.2%} of all channels)"
    )
    fig.savefig(
        output / "channel-occupancy.png", dpi=config["dpi"], bbox_inches="tight"
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        choices=[c["name"] for c in experiment_configs],
        default=experiment_configs[0]["name"],
    )
    args = parser.parse_args()
    config = next(c for c in experiment_configs if c["name"] == args.name)
    if config.get("analysis") == "distributions":
        plot_distributions(config)
    elif config.get("analysis") == "channels":
        plot_channels(config)
    else:
        plot(config)
