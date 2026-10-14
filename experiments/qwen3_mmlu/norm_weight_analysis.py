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


def plot(config):
    """Write four figures and exact, unclipped summary statistics."""
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
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        choices=[c["name"] for c in experiment_configs],
        default=experiment_configs[0]["name"],
    )
    args = parser.parse_args()
    plot(next(c for c in experiment_configs if c["name"] == args.name))
