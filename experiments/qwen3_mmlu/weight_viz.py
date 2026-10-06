# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline checkpoint weight report, without loading a model or running inference."""

import argparse
import datetime
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pybase64 as base64
import regex as re
import torch
from safetensors import safe_open

import run

experiment_configs = [
    {
        "name": "qwen3_weights",
        "model": run.CONFIG["model"],
        "snapshot": run.CONFIG["snapshot"],
        "output": str(run.ARTIFACTS / "weight-viz"),
        "batch_size": 16,
        "cpu_threads": 4,
        "sample_size": 8192,
        "power_steps": 20,
        "power_starts": 2,
        "seed": 42,
        "linear_bins": 128,
        "linear_reach": 0.5,
        "norm_reach": 4.0,
        "log_edges": np.logspace(-7, 1, 65).tolist(),
        "max_html_bytes": 20_000_000,
    }
]
CONFIG = experiment_configs[0]
TEMPLATE = Path(__file__).with_suffix(".html")
STATS = (
    "min",
    "max",
    "mean",
    "std",
    "rms",
    "mean_abs",
    "zero_share",
    "negative_share",
    "abs_q10_est",
    "abs_q50_est",
    "abs_q90_est",
    "abs_q99_est",
    "excess_kurtosis",
    "frobenius",
    "sigma_max_est",
    "stable_rank_est",
    "row_norm_cv",
    "col_norm_cv",
)


def identity(name):
    """Checkpoint tensor's (layer, expert, family), with -1 for global/non-expert."""
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)\.weight", name)
    if match:
        layer, tail = int(match[1]), match[2]
        expert = re.fullmatch(r"mlp\.experts\.(\d+)\.(\w+)", tail)
        if expert:
            return layer, int(expert[1]), "expert." + expert[2]
        return layer, -1, tail
    return -1, -1, name.removeprefix("model.").removesuffix(".weight")


def generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def norm_cv(norms):
    mean = norms.mean(dim=1)
    return norms.std(dim=1, correction=0) / mean.clamp_min(1e-30)


@torch.inference_mode()
def summarize(batch, config, seed):
    """Reduce equally shaped tensors to moments, histograms, and spectral estimates.

    Args:
        batch: [batch, rows, columns] matrices or [batch, width] vectors.
        config: Analysis configuration from experiment_configs.
        seed: Reproducible seed for quantile sampling and power iteration.

    Returns:
        FP64 scalar statistics, integer linear/sign-split log histograms, and
        each tensor's linear histogram reach. NaN denotes unavailable metrics.

    """
    x = batch.float()
    flat = x.flatten(1)
    if not torch.isfinite(flat).all():
        raise ValueError("Checkpoint contains non-finite weights")
    count, width = flat.shape
    mean = flat.mean(dim=1)
    centered = flat - mean[:, None]
    variance = centered.square().mean(dim=1)
    square_sum = flat.square().sum(dim=1)
    stats = torch.full((count, len(STATS)), torch.nan, device=x.device)
    stats[:, 0] = flat.amin(dim=1)
    stats[:, 1] = flat.amax(dim=1)
    stats[:, 2] = mean
    stats[:, 3] = variance.sqrt()
    stats[:, 4] = (square_sum / width).sqrt()
    stats[:, 5] = flat.abs().mean(dim=1)
    stats[:, 6] = (flat == 0).float().mean(dim=1)
    stats[:, 7] = (flat < 0).float().mean(dim=1)
    if width <= config["sample_size"]:
        sample = flat.abs()
    else:
        indices = torch.randint(
            width,
            (count, config["sample_size"]),
            device=x.device,
            generator=generator(x.device, seed),
        )
        sample = flat.gather(1, indices).abs()
    q = torch.tensor([0.1, 0.5, 0.9, 0.99], device=x.device)
    stats[:, 8:12] = torch.quantile(sample, q, dim=1).T
    fourth = centered.square().square().mean(dim=1)
    stats[:, 12] = torch.where(
        variance > 0, fourth / variance.square().clamp_min(1e-30) - 3, torch.nan
    )
    stats[:, 13] = square_sum.sqrt()
    if x.ndim == 3:
        # Work on the smaller Gram space, without ever materializing the Gram.
        matrix = x.transpose(1, 2) if x.shape[1] < x.shape[2] else x
        v = torch.randn(
            count,
            matrix.shape[2],
            config["power_starts"],
            device=x.device,
            generator=generator(x.device, seed + 1),
        )
        for _ in range(config["power_steps"]):
            v = torch.bmm(matrix.transpose(1, 2), torch.bmm(matrix, v))
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
        sigma = torch.bmm(matrix, v).norm(dim=1).amax(dim=1)
        stats[:, 14] = sigma
        stats[:, 15] = torch.where(
            sigma > 0, square_sum / sigma.square(), torch.zeros_like(sigma)
        )
        stats[:, 16] = norm_cv(x.norm(dim=2))
        stats[:, 17] = norm_cv(x.norm(dim=1))
    reach = config["norm_reach"] if x.ndim == 2 else config["linear_reach"]
    linear_edges = torch.linspace(
        -reach, reach, config["linear_bins"] + 1, device=x.device
    )
    edges = torch.tensor(config["log_edges"], device=x.device)
    # Two tail bins around the finite linear interval; every value is counted.
    bins = torch.bucketize(flat.contiguous(), linear_edges, right=True)
    offsets = torch.arange(count, device=x.device)[:, None]
    linear_width = len(linear_edges) + 1
    linear = torch.bincount(
        (bins + offsets * linear_width).flatten(), minlength=count * linear_width
    ).reshape(count, linear_width)
    log_width = len(edges) + 1
    bins = torch.bucketize(flat.abs().contiguous(), edges, right=True)
    bins += (flat >= 0).long() * log_width
    log = torch.bincount(
        (bins + offsets * log_width * 2).flatten(), minlength=count * log_width * 2
    ).reshape(count, 2, log_width)
    if not torch.all(linear.sum(dim=1) == width) or not torch.all(
        log.sum(dim=(1, 2)) == width
    ):
        raise RuntimeError("Histogram does not cover every weight")
    return (
        stats.double().cpu().numpy(),
        linear.cpu().numpy(),
        log.cpu().numpy(),
        reach,
    )


def tensor_batches(handle, names, config):
    """Bound CPU/GPU memory, including large embedding and output matrices."""
    pending = []
    previous = None
    for name in names:
        shape = tuple(handle.get_slice(name).get_shape())
        if len(shape) not in (1, 2):
            raise ValueError(f"Unsupported tensor shape: {name} {shape}")
        # Keep tensors of one family together, with deterministic batching.
        key = (shape, identity(name)[2])
        capacity = max(1, min(config["batch_size"], 16_000_000 // math.prod(shape)))
        if pending and (key != previous or len(pending) >= capacity):
            yield pending
            pending = []
        pending.append(name)
        previous = key
    if pending:
        yield pending


def collect(config, device, output):
    """Scan each safetensors shard once and checkpoint completed shards to disk."""
    snapshot = Path(config["snapshot"])
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = defaultdict(list)
    for name, shard in index["weight_map"].items():
        shards[shard].append(name)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode() + source_hash.encode()
    ).hexdigest()
    records, stats, linear, log = [], [], [], []
    analysis = {}
    start = time.monotonic()
    for shard, names in sorted(shards.items()):
        cache = output / (shard + ".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if "analysis" not in saved or str(saved["fingerprint"]) != fingerprint:
                    raise ValueError(f"Stale analysis cache: {cache}")
                analysis[shard] = json.loads(str(saved["analysis"]))
                records.extend(json.loads(str(saved["records"])))
                stats.extend(saved["stats"])
                linear.extend(saved["linear"])
                log.extend(saved["log"])
            print(f"Cached {shard}: {len(records)} tensors", flush=True)
            continue
        shard_records, shard_stats, shard_linear, shard_log = [], [], [], []
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            ordered = sorted(names, key=lambda n: (identity(n)[2], identity(n)[:2], n))
            for group in tensor_batches(handle, ordered, config):
                batch = torch.stack([handle.get_tensor(name) for name in group]).to(
                    device
                )
                seed = config["seed"] + int.from_bytes(
                    hashlib.sha256(group[0].encode()).digest()[:4], "little"
                )
                moments, lh, gh, reach = summarize(batch, config, seed)
                for i, name in enumerate(group):
                    layer, expert, family = identity(name)
                    shape = list(batch.shape[1:])
                    shard_records.append([name, layer, expert, family, shape, reach])
                    shard_stats.append(moments[i])
                    shard_linear.append(lh[i])
                    shard_log.append(gh[i])
                del batch
        temporary = cache.with_suffix(".tmp.npz")
        analysis[shard] = {
            "device": str(torch.device(device)),
            "torch": torch.__version__,
            "source_sha256": source_hash,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        np.savez_compressed(
            temporary,
            fingerprint=fingerprint,
            records=json.dumps(shard_records),
            stats=shard_stats,
            linear=shard_linear,
            log=shard_log,
            analysis=json.dumps(analysis[shard]),
        )
        temporary.replace(cache)
        records.extend(shard_records)
        stats.extend(shard_stats)
        linear.extend(shard_linear)
        log.extend(shard_log)
        print(
            f"Scanned {shard}: {len(records)}/{len(index['weight_map'])} tensors, "
            f"{time.monotonic() - start:.0f}s",
            flush=True,
        )
    if len(records) != len(index["weight_map"]):
        raise RuntimeError("Not every checkpoint tensor was analysed")
    return records, np.array(stats), np.array(linear), np.array(log), analysis


def encode(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def render(records, stats, linear, log, config, provenance):
    """Embed Plotly and gzipped data; enforce the limit on actual UTF-8 bytes."""
    from plotly.offline import get_plotlyjs

    payload = {
        "config": config,
        "provenance": provenance,
        "records": records,
        "stat_names": STATS,
        "stats": encode(stats, "<f4"),
        "linear": encode(linear, "<u4"),
        "log": encode(log, "<u4"),
    }
    if max(linear.max(), log.max()) > np.iinfo(np.uint32).max:
        raise ValueError("Histogram count exceeds uint32")
    compressed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode())
    data = base64.b64encode(compressed).decode()
    # Escape literal closing tags in bundled library without changing JS strings.
    plotly = get_plotlyjs().replace("</script", "<\\/script")
    html = TEMPLATE.read_text().replace("%%PLOTLY%%", plotly).replace("%%DATA%%", data)
    size = len(html.encode())
    if size > config["max_html_bytes"]:
        raise ValueError(f"HTML is {size:,} bytes, above {config['max_html_bytes']:,}")
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--render-only", action="store_true", help="require shard caches"
    )
    args = parser.parse_args()
    config = CONFIG.copy()
    torch.set_num_threads(config["cpu_threads"])
    output = args.output or Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        index = json.loads(
            (Path(config["snapshot"]) / "model.safetensors.index.json").read_text()
        )
        for shard in set(index["weight_map"].values()):
            if not (output / (shard + ".npz")).exists():
                parser.error(f"Missing cache for {shard}")
    torch.backends.cuda.matmul.allow_tf32 = False
    records, stats, linear, log, analysis = collect(config, args.device, output)
    provenance = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "snapshot": Path(config["snapshot"]).name,
        "analysis_shards": analysis,
        "render": {"torch": torch.__version__},
        "tensor_count": len(records),
        "parameter_count": sum(math.prod(r[4]) for r in records),
    }
    page = render(records, stats, linear, log, config, provenance)
    path = output / "weights.html"
    temporary = path.with_suffix(".tmp.html")
    temporary.write_text(page)
    temporary.replace(path)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"{path}: {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
