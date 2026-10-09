# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-output-row quantile replacement of loaded Qwen3 expert projections."""

import math

import torch
from rounding_stats import REPLACEMENTS, normal_coefficient, validate_quantile


def validate_config(config):
    if set(config) != {"quantile", "replacement", "batch_rows"}:
        raise ValueError("MLP weights require quantile, replacement and batch_rows")
    validate_quantile(config["quantile"])
    if isinstance(config["quantile"], bool) or not isinstance(
        config["quantile"], (int, float)
    ):
        raise ValueError("Quantile must be numeric")
    if config["replacement"] not in REPLACEMENTS:
        raise ValueError("Unknown MLP weight replacement")
    batch = config["batch_rows"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("batch_rows must be a positive integer")
    return config


@torch.no_grad()
def replace_rows(weight, config):
    """Modify logical output rows in place, including all cutoff ties."""
    validate_config(config)
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("Expected a BF16 output-by-input matrix")
    width = weight.shape[1]
    rank = math.floor(config["quantile"] * width)
    coefficient = (
        normal_coefficient(config["quantile"])
        if config["replacement"] != "zero"
        else 0.0
    )
    report = dict(
        floats=weight.numel(),
        selected=0,
        modified=0,
        zeros_before=0,
        zeros_after=0,
        squared_error=0.0,
        original_energy=0.0,
        cutoff_min=None,
        cutoff_max=None,
        epsilon_min=None,
        epsilon_max=None,
    )
    for start in range(0, len(weight), config["batch_rows"]):
        batch = weight[start : start + config["batch_rows"]]
        original = batch.float()
        if not torch.isfinite(original).all():
            raise ValueError("Nonfinite MLP checkpoint weights")
        report["zeros_before"] += (original == 0).sum()
        report["original_energy"] += original.square().sum()
        if rank:
            cutoff = original.abs().kthvalue(rank, dim=1, keepdim=True).values
            epsilon = cutoff * coefficient
            selected = original.abs() <= cutoff
            mask = selected & (original != 0)
            target = original.sign() * epsilon if config["replacement"] != "zero" else 0
            batch.copy_(torch.where(mask, target, original).to(batch.dtype))
            report["selected"] += selected.sum()
            for name, values in (("cutoff", cutoff), ("epsilon", epsilon)):
                low, high = values.min(), values.max()
                old_low, old_high = report[name + "_min"], report[name + "_max"]
                report[name + "_min"] = (
                    low if old_low is None else torch.minimum(old_low, low)
                )
                report[name + "_max"] = (
                    high if old_high is None else torch.maximum(old_high, high)
                )
        report["modified"] += (batch.float() != original).sum()
        report["zeros_after"] += (batch == 0).sum()
        report["squared_error"] += (batch.float() - original).square().sum()
    keys = [key for key, value in report.items() if isinstance(value, torch.Tensor)]
    if keys:
        numbers = torch.stack([report[key].double() for key in keys]).cpu().tolist()
        for key, value in zip(keys, numbers):
            report[key] = (
                int(value)
                if key in ("selected", "modified", "zeros_before", "zeros_after")
                else value
            )
    report["mse"] = (
        report["squared_error"] / report["floats"] if report["floats"] else 0
    )
    return report


def apply(model, config):
    """Replace gate/up/down expert weights before the first forward pass."""
    validate_config(config)
    layout = model.config
    expected = (
        (layout.num_experts, 2 * layout.moe_intermediate_size, layout.hidden_size),
        (layout.num_experts, layout.hidden_size, layout.moe_intermediate_size),
    )
    layers = model.model.layers
    for layer in layers:
        experts = layer.mlp.experts
        for weight, shape in zip((experts.w13_weight, experts.w2_weight), expected):
            if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16:
                raise ValueError("Unsupported Qwen3 fused expert weight layout")
    reports = []
    for index, layer in enumerate(layers):
        experts = layer.mlp.experts
        for expert in range(layout.num_experts):
            gate, up = experts.w13_weight[expert].split(layout.moe_intermediate_size)
            for family, weight in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", experts.w2_weight[expert]),
            ):
                reports.append(
                    dict(
                        layer=index,
                        expert=expert,
                        family=family,
                        **replace_rows(weight, config),
                    )
                )
    return {
        "config": config,
        "rule": "per-output row; cutoff ties included",
        "matrices": reports,
    }
