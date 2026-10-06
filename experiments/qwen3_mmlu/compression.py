# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert weight compression screening on held-out routed activations.

FP32 replay of BF16 checkpoint weights isolates compression error. This is local
layer analysis, not model accuracy evaluation or a sparse serving implementation.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import run

experiment_configs = [
    {
        "name": "expert_compression_screen",
        "launcher": "mls",
        "entrypoint": "compression.py",
        "python": run.JOB_PYTHON,
        "instance_type": "a100.1gpu",
        "model": run.CONFIG["model"],
        "snapshot": run.CONFIG["snapshot"],
        "output": str(run.ARTIFACTS / "compression" / "expert_compression_screen_v2"),
        "weight_stats": str(run.ARTIFACTS / "weight-viz" / "qwen3-30b-v2"),
        "dataset": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "splits": {"calibration": "train", "heldout": "validation"},
        "tokens": {"calibration": 32768, "heldout": 16384},
        "max_rows": {"calibration": 2048, "heldout": 512},
        "min_rows": {"calibration": 256, "heldout": 64},
        "sequence_length": 512,
        "layers": [0, 24, 47],
        "rank_quantiles": [0.1, 0.5, 0.9],
        "routing_extremes": True,
        "families": ["gate_proj", "up_proj", "down_proj"],
        "methods": ["magnitude", "wanda", "svd", "whitened_svd"],
        "storage_fractions": [0.25, 0.5, 0.75, 0.9],
        "covariance_damping": 1e-4,
        "seed": 42,
        "cpu_threads": 4,
        "value_bytes": 2,
        "csr_index_bytes": 4,
        "error_threshold": 0.1,
    }
]


def fingerprint(config):
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(
        (json.dumps(config, sort_keys=True) + source).encode()
    ).hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


class Reservoir:
    """Uniform priority sample of routed token rows with bounded storage."""

    def __init__(self, capacity, seed):
        import torch

        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)
        self.rows = None
        self.priorities = torch.empty(0, dtype=torch.float64)
        self.seen = 0

    def add(self, rows):
        import torch

        rows = rows.detach().to("cpu").clone()
        self.seen += len(rows)
        priorities = torch.rand(
            len(rows), generator=self.generator, dtype=torch.float64
        )
        self.rows = rows if self.rows is None else torch.cat([self.rows, rows])
        self.priorities = torch.cat([self.priorities, priorities])
        if len(self.rows) > self.capacity:
            keep = self.priorities.topk(self.capacity).indices
            self.rows = self.rows[keep]
            self.priorities = self.priorities[keep]


def corpus_tokens(config, split, tokenizer):
    """Seeded document sampling with separate, pinned dataset splits."""
    from datasets import load_dataset

    dataset = load_dataset(
        config["dataset"],
        config["dataset_config"],
        split=config["splits"][split],
        revision=config["dataset_revision"],
    ).shuffle(seed=config["seed"])
    target = config["tokens"][split]
    texts, ids = [], []
    for row in dataset:
        if not row["text"].strip():
            continue
        texts.append(row["text"])
        # Documents retain their own tokenization; every chunk resets context.
        ids.extend(tokenizer.encode(row["text"] + "\n", add_special_tokens=False))
        if len(ids) >= target:
            break
    if len(ids) < target:
        raise ValueError(f"Not enough tokens in {split}: {len(ids)} < {target}")
    return ids[:target], hashlib.sha256("\n".join(texts).encode()).hexdigest()


def capture_hook(samples):
    """Record only tokens routed to each expert, before its gate/up projections."""

    def record(module, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        selected = module.gate(x).float().softmax(-1).topk(module.top_k, dim=-1).indices
        for expert in selected.unique().tolist():
            token = (selected == expert).any(dim=1)
            samples[expert].add(x[token])

    return record


def capture(config, device, output):
    """Capture all experts' actual post-norm inputs at the selected layers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    marker = output / "capture.json"
    digest = fingerprint(config)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["fingerprint"] != digest:
            raise ValueError("Stale activation capture; use a new output directory")
        if not all(
            (output / f"inputs-{s}-{layer}.pt").exists()
            for s in config["splits"]
            for layer in config["layers"]
        ):
            raise ValueError("Activation capture is incomplete")
        return saved
    tokenizer = AutoTokenizer.from_pretrained(config["snapshot"], local_files_only=True)
    corpora = {
        split: corpus_tokens(config, split, tokenizer) for split in config["splits"]
    }
    model = AutoModelForCausalLM.from_pretrained(
        config["snapshot"],
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    metadata = {
        "fingerprint": digest,
        "config": config,
        "torch": torch.__version__,
        "device": device,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "splits": {},
        "num_experts": model.config.num_experts,
    }
    for split, (ids, text_hash) in corpora.items():
        reservoirs = {
            layer: [
                Reservoir(
                    config["max_rows"][split],
                    config["seed"]
                    + layer * 1000
                    + expert
                    + (100000 if split == "heldout" else 0),
                )
                for expert in range(model.config.num_experts)
            ]
            for layer in config["layers"]
        }

        handles = [
            model.model.layers[layer].mlp.register_forward_pre_hook(
                capture_hook(reservoirs[layer])
            )
            for layer in config["layers"]
        ]
        try:
            with torch.inference_mode():
                for start in range(0, len(ids), config["sequence_length"]):
                    chunk = torch.tensor(
                        [ids[start : start + config["sequence_length"]]], device=device
                    )
                    model.model(chunk, use_cache=False)
                    if start % (config["sequence_length"] * 8) == 0:
                        print(
                            f"Capture {split}: {start + len(chunk[0])}/{len(ids)} "
                            "tokens",
                            flush=True,
                        )
        finally:
            for handle in handles:
                handle.remove()
        counts = {}
        for layer, samples in reservoirs.items():
            data = {
                e: {"rows": r.rows, "seen": r.seen}
                for e, r in enumerate(samples)
                if r.rows is not None
            }
            path = output / f"inputs-{split}-{layer}.pt"
            temp = path.with_suffix(".tmp.pt")
            torch.save(data, temp)
            temp.replace(path)
            counts[str(layer)] = {
                str(e): {"seen": r["seen"], "stored": len(r["rows"])}
                for e, r in data.items()
            }
        metadata["splits"][split] = {
            "text_sha256": text_hash,
            "token_sha256": hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest(),
            "tokens": len(ids),
            "counts": counts,
        }
        del reservoirs
    del model
    atomic_json(marker, metadata)
    return metadata


def weight_ranks(config):
    """Prior stable-rank estimates for selecting a stratified expert sample."""
    import numpy as np
    from weight_viz import STATS

    records = {}
    paths = sorted(Path(config["weight_stats"]).glob("*.npz"))
    if not paths:
        raise ValueError("Missing checkpoint weight statistics")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for record, stats in zip(json.loads(str(data["records"])), data["stats"]):
                if record[1] in config["layers"] and record[2] >= 0:
                    records.setdefault((record[1], record[2]), []).append(
                        float(stats[STATS.index("stable_rank_est")])
                    )
    return {key: sum(values) / len(values) for key, values in records.items()}


def select_experts(config, layer, calibration, heldout, ranks):
    """Rank strata plus frequent/rare routing, only where both splits suffice."""
    import numpy as np

    eligible = [
        e
        for e in calibration
        if e in heldout
        and (layer, e) in ranks
        and len(calibration[e]["rows"]) >= config["min_rows"]["calibration"]
        and len(heldout[e]["rows"]) >= config["min_rows"]["heldout"]
    ]
    if not eligible:
        return {}, []
    picked = {}
    values = [ranks[layer, e] for e in eligible]
    for q in config["rank_quantiles"]:
        target = float(np.quantile(values, q))
        e = min(eligible, key=lambda e: (abs(ranks[layer, e] - target), e))
        picked.setdefault(e, []).append(f"rank_q{q}")
    if config["routing_extremes"]:
        ordered = sorted(eligible, key=lambda e: (calibration[e]["seen"], e))
        picked.setdefault(ordered[0], []).append("rare_eligible")
        picked.setdefault(ordered[-1], []).append("frequent")
    return picked, eligible


def storage_budget(shape, fraction, config):
    """Equal byte budgets: BF16 factors versus BF16/int32 CSR, including rowptr."""
    m, n = shape
    dense = config["value_bytes"] * m * n
    budget = math.floor(dense * fraction)
    rank = min(m, n, budget // (config["value_bytes"] * (m + n)))
    nnz = max(
        0,
        min(
            m * n,
            (budget - config["csr_index_bytes"] * (m + 1))
            // (config["value_bytes"] + config["csr_index_bytes"]),
        ),
    )
    return dense, budget, rank, nnz


def prune(weight, score, nnz, per_output=False):
    """Keep an exact global count, deterministically resolving tied scores."""
    import torch

    if per_output:
        # Wanda selects within each output; distribute remainder positions evenly.
        ordered = torch.argsort(score, dim=1, descending=True, stable=True)
        counts = torch.full((len(weight),), nnz // len(weight), device=weight.device)
        counts[: nnz % len(weight)] += 1
        mask = (
            torch.arange(weight.shape[1], device=weight.device)[None] < counts[:, None]
        )
        keep = (
            ordered
            + torch.arange(len(weight), device=weight.device)[:, None] * weight.shape[1]
        )[mask]
    else:
        keep = torch.argsort(score.flatten(), descending=True, stable=True)[:nnz]
    result = torch.zeros_like(weight).flatten()
    result[keep] = weight.flatten()[keep]
    return result.reshape_as(weight)


def svd_basis(weight, calibration, damping):
    """Ordinary and damped covariance-whitened SVD; no sequential weight update."""
    import torch

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    calibration64 = calibration.double()
    covariance = calibration64.T @ calibration64 / len(calibration)
    ridge = damping * covariance.diagonal().mean().clamp_min(1e-12)
    covariance.diagonal().add_(ridge)
    chol = torch.linalg.cholesky(covariance)
    wu, ws, wvh = torch.linalg.svd(weight.double() @ chol, full_matrices=False)
    inverse_basis = torch.linalg.solve_triangular(chol.T, wvh.T, upper=True).T
    return (u, s, vh), (wu, ws, inverse_basis)


def factors(basis, rank):
    """BF16 stored factors, replayed in FP32 to include storage rounding error."""
    u, s, vh = basis
    scale = s[:rank].sqrt()
    return (u[:, :rank] * scale).bfloat16().float(), (
        scale[:, None] * vh[:rank]
    ).bfloat16().float()


def apply(compressed, x):
    return (
        x @ compressed.T
        if not isinstance(compressed, tuple)
        else (x @ compressed[1].T) @ compressed[0].T
    )


def errors(reference, estimate):
    import torch
    import torch.nn.functional as F

    difference = estimate - reference
    token = difference.norm(dim=1) / reference.norm(dim=1).clamp_min(1e-12)
    cosine = F.cosine_similarity(reference, estimate, dim=1)
    return {
        "relative_frobenius": float(
            difference.norm() / reference.norm().clamp_min(1e-12)
        ),
        "token_median": float(token.median()),
        "token_p95": float(torch.quantile(token, 0.95)),
        "token_max": float(token.max()),
        "mean_cosine": float(cosine.mean()),
    }


def expert_output(weights, x, family=None, compressed=None):
    import torch.nn.functional as F

    gate = apply(compressed, x) if family == "gate_proj" else x @ weights["gate_proj"].T
    up = apply(compressed, x) if family == "up_proj" else x @ weights["up_proj"].T
    hidden = F.silu(gate) * up
    return (
        apply(compressed, hidden)
        if family == "down_proj"
        else hidden @ weights["down_proj"].T
    )


def screen(config, device, output):
    """Evaluate selected matrices and their expert branches on held-out inputs."""
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    manifest = json.loads((output / "capture.json").read_text())
    if manifest["fingerprint"] != fingerprint(config):
        raise ValueError("Stale activation capture")
    index = json.loads(
        (Path(config["snapshot"]) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    ranks = weight_ranks(config)
    results, coverage = [], []
    for layer in config["layers"]:
        cal = torch.load(output / f"inputs-calibration-{layer}.pt", weights_only=True)
        held = torch.load(output / f"inputs-heldout-{layer}.pt", weights_only=True)
        picked, eligible = select_experts(config, layer, cal, held, ranks)
        coverage.append(
            {
                "layer": layer,
                "eligible": eligible,
                "selected": picked,
                "unobserved_or_insufficient": sorted(
                    set(range(manifest["num_experts"])) - set(eligible)
                ),
            }
        )
        for expert, labels in sorted(picked.items()):
            weights = {}
            for family in config["families"]:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{family}.weight"
                with safe_open(
                    Path(config["snapshot"]) / index[name], framework="pt", device="cpu"
                ) as handle:
                    weights[family] = handle.get_tensor(name).to(device).float()
            xcal, xtest = (
                cal[expert]["rows"].to(device).float(),
                held[expert]["rows"].to(device).float(),
            )
            original = expert_output(weights, xtest)
            inner_cal = F.silu(xcal @ weights["gate_proj"].T) * (
                xcal @ weights["up_proj"].T
            )
            inner_test = F.silu(xtest @ weights["gate_proj"].T) * (
                xtest @ weights["up_proj"].T
            )
            for family in config["families"]:
                weight = weights[family]
                cx, tx = (
                    (inner_cal, inner_test) if family == "down_proj" else (xcal, xtest)
                )
                ordinary, whitened = svd_basis(weight, cx, config["covariance_damping"])
                singular = ordinary[1]
                stable = float(singular.square().sum() / singular[0].square())
                energy = singular.square().cumsum(0) / singular.square().sum()
                r99 = int(torch.searchsorted(energy, 0.99)) + 1
                for fraction in config["storage_fractions"]:
                    dense, budget, rank, nnz = storage_budget(
                        weight.shape, fraction, config
                    )
                    for method in config["methods"]:
                        if method in ("magnitude", "wanda"):
                            score = weight.abs()
                            if method == "wanda":
                                score = score * cx.norm(dim=0)[None]
                            compressed = prune(
                                weight, score, nnz, per_output=method == "wanda"
                            )
                            stored = nnz * (
                                config["value_bytes"] + config["csr_index_bytes"]
                            ) + config["csr_index_bytes"] * (weight.shape[0] + 1)
                        else:
                            compressed = factors(
                                ordinary if method == "svd" else whitened, rank
                            )
                            stored = rank * sum(weight.shape) * config["value_bytes"]
                        linear_error = errors(tx @ weight.T, apply(compressed, tx))
                        branch_error = errors(
                            original, expert_output(weights, xtest, family, compressed)
                        )
                        results.append(
                            {
                                "layer": layer,
                                "expert": expert,
                                "family": family,
                                "selection": ";".join(labels),
                                "method": method,
                                "budget_fraction": fraction,
                                "stored_bytes": stored,
                                "dense_bytes": dense,
                                "storage_fraction_actual": stored / dense,
                                "rank": rank if isinstance(compressed, tuple) else "",
                                "retained_weights": nnz
                                if not isinstance(compressed, tuple)
                                else "",
                                "stable_rank": stable,
                                "rank99_weight_energy": r99,
                                "calibration_rows": len(cx),
                                "heldout_rows": len(tx),
                                "calibration_seen": cal[expert]["seen"],
                                **{f"linear_{k}": v for k, v in linear_error.items()},
                                **{f"expert_{k}": v for k, v in branch_error.items()},
                            }
                        )
                print(
                    f"Screen layer {layer} expert {expert} {family}: "
                    f"stable rank {stable:.1f}, "
                    f"r99 {r99}",
                    flush=True,
                )
            del weights, xcal, xtest, original, inner_cal, inner_test
        del cal, held
    if not results:
        raise ValueError("No experts have sufficient calibration and held-out coverage")
    path = output / "results.csv"
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(path)
    atomic_json(output / "coverage.json", coverage)
    atomic_json(
        output / "screen.json",
        {
            "fingerprint": fingerprint(config),
            "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "capture_sha256": hashlib.sha256(
                (output / "capture.json").read_bytes()
            ).hexdigest(),
            "device": device,
            "torch": torch.__version__,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    return results


def plot(config, output):
    """Standalone figures separating matrix and whole-expert errors."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = json.loads((output / "screen.json").read_text())
    if (
        saved["fingerprint"] != fingerprint(config)
        or saved["results_sha256"]
        != hashlib.sha256((output / "results.csv").read_bytes()).hexdigest()
        or saved["capture_sha256"]
        != hashlib.sha256((output / "capture.json").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Stale screening results; use matching source/config and artifacts"
        )
    with (output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for column, family in enumerate(config["families"]):
        for i, metric in enumerate(
            ["linear_relative_frobenius", "expert_relative_frobenius"]
        ):
            ax = axes[i, column]
            for method in config["methods"]:
                medians, low, high = [], [], []
                for fraction in config["storage_fractions"]:
                    values = [
                        float(r[metric])
                        for r in rows
                        if r["family"] == family
                        and r["method"] == method
                        and float(r["budget_fraction"]) == fraction
                    ]
                    q = np.quantile(values, [0.25, 0.5, 0.75])
                    low.append(q[0])
                    medians.append(q[1])
                    high.append(q[2])
                (line,) = ax.plot(
                    config["storage_fractions"], medians, marker="o", label=method
                )
                ax.fill_between(
                    config["storage_fractions"],
                    low,
                    high,
                    color=line.get_color(),
                    alpha=0.12,
                )
            ax.set_title(f"{family}: {'matrix' if i == 0 else 'whole expert'}")
            ax.set_ylabel("Relative output error (median / IQR)")
            ax.grid(alpha=0.2)
            if i == 1:
                ax.set_xlabel("Stored bytes / dense BF16 bytes")
    axes[0, 0].legend()
    fig.suptitle("Held-out routed tokens; selected expert sample; FP32 replay")
    fig.tight_layout()
    fig.savefig(output / "compression-error.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, family in zip(axes, config["families"]):
        for method in config["methods"]:
            selected = [
                r for r in rows if r["family"] == family and r["method"] == method
            ]
            keys = {(r["layer"], r["expert"]) for r in selected}
            xs, ys = [], []
            for key in sorted(keys):
                trials = [r for r in selected if (r["layer"], r["expert"]) == key]
                accepted = [
                    float(r["storage_fraction_actual"])
                    for r in trials
                    if float(r["expert_relative_frobenius"])
                    <= config["error_threshold"]
                ]
                xs.append(float(trials[0]["stable_rank"]))
                ys.append(min(accepted) if accepted else 1.0)
            ax.scatter(xs, ys, label=method, alpha=0.7)
        ax.set_title(family)
        ax.set_xlabel("Stable rank (full SVD)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(
        "Smallest tested storage fraction within error threshold\n"
        "1 = no tested compression passed"
    )
    axes[0].legend()
    fig.suptitle("Exploratory stratified sample; not a population correlation estimate")
    fig.tight_layout()
    fig.savefig(output / "rank-vs-compression.png", dpi=160)
    plt.close(fig)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "capture", "screen", "plot"])
    parser.add_argument("--name", default=experiment_configs[0]["name"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = next(c for c in experiment_configs if c["name"] == args.name)
    output = args.output or Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        if args.stage in ("all", "capture"):
            capture(config, args.device, output)
        if args.stage in ("all", "screen"):
            screen(config, args.device, output)
        if args.stage in ("all", "screen", "plot"):
            plot(config, output)
    if args.stage == "all":
        atomic_json(
            output / "done.json",
            {
                "fingerprint": fingerprint(config),
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    main()
