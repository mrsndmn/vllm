# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""In-place per-token magnitude pruning of loaded input/output weights."""

import math

import torch


def validate_config(config):
    """Validate the complete worker protocol before modifying any weights."""
    if set(config) != {"scope", "quantile", "batch_rows"}:
        raise ValueError("Weight zeroing requires scope, quantile and batch_rows")
    if config["scope"] not in ("embedding", "both"):
        raise ValueError("Weight zeroing scope must be embedding or both")
    q = config["quantile"]
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= q < 1:
        raise ValueError("Weight zeroing quantile must be in [0, 1)")
    batch = config["batch_rows"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("Weight zeroing batch_rows must be a positive integer")
    return config


@torch.no_grad()
def zero_rows(weight, quantile, batch_rows):
    """Zero floor(q * hidden_size) smallest magnitudes independently per row.

    Equal magnitudes are resolved by channel index. Existing zeros count towards
    the target; rows already beyond it retain their existing sparsity.
    """
    validate_config(dict(scope="embedding", quantile=quantile, batch_rows=batch_rows))
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("Weight zeroing requires a contiguous BF16 matrix")
    rows, width = weight.shape
    if not rows or not width:
        raise ValueError("Weight zeroing requires nonempty token vectors")
    count = math.floor(quantile * width)
    zeros_before = zeros_after = 0
    energy_before = energy_after = 0.0
    cutoffs = []
    for start in range(0, rows, batch_rows):
        batch = weight[start : start + batch_rows]
        if not torch.isfinite(batch).all():
            raise ValueError("Weight zeroing encountered nonfinite checkpoint values")
        zeros_before += int((batch == 0).sum().item())
        energy_before += float(batch.double().square().sum().item())
        if count:
            order = batch.abs().argsort(dim=1, stable=True)
            selected = order[:, :count]
            cutoff = batch.gather(1, selected[:, -1:]).abs().float().cpu()
            cutoffs.append(cutoff)
            batch.scatter_(1, selected, 0)
        zeros_after += int((batch == 0).sum().item())
        energy_after += float(batch.double().square().sum().item())
    thresholds = torch.cat(cutoffs).flatten() if cutoffs else torch.zeros(rows)
    return {
        "rows": rows,
        "hidden_size": width,
        "target_zeros_per_row": count,
        "target_zero_share": count / width,
        "zeros_before": zeros_before,
        "zeros_after": zeros_after,
        "newly_zeroed": zeros_after - zeros_before,
        "zero_share": zeros_after / weight.numel(),
        "retained_energy_share": energy_after / energy_before
        if energy_before
        else None,
        "cutoff_min_median_max": torch.quantile(
            thresholds, torch.tensor([0.0, 0.5, 1.0])
        ).tolist(),
    }


def apply(model, config):
    """Prune vocabulary rows once per storage, preserving any genuine tying."""
    validate_config(config)
    embedding = model.model.embed_tokens.weight
    output = model.lm_head.weight
    vocab = model.config.vocab_size
    hidden = model.config.hidden_size
    for weight in (embedding, output):
        if weight.ndim != 2 or weight.shape[0] < vocab or weight.shape[1] != hidden:
            raise ValueError("Unsupported input/output vocabulary layout")
    shared = (
        embedding.device == output.device
        and embedding.untyped_storage().data_ptr()
        == output.untyped_storage().data_ptr()
    )
    tied = (
        shared
        and embedding.data_ptr() == output.data_ptr()
        and embedding.shape == output.shape
    )
    if shared and not tied:
        raise ValueError("Partially aliased input/output weights are unsupported")
    if bool(model.config.tie_word_embeddings) != tied:
        raise ValueError("Checkpoint tying declaration differs from loaded storage")
    selected = [("embedding", embedding)]
    if config["scope"] == "both" and not tied:
        selected.append(("unembedding", output))
    reports = {}
    for name, weight in selected:
        reports[name] = zero_rows(
            weight[:vocab], config["quantile"], config["batch_rows"]
        )
    if tied:
        reports["unembedding"] = {"shared_with": "embedding"}
    return {
        "config": config,
        "tied": tied,
        "rule": "per-token floor(q * hidden_size), ties by channel index",
        "matrices": reports,
    }
