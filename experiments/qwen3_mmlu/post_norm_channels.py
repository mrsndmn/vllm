# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep a pinned intersection of large RMSNorm channels at selected decoder sites."""

import hashlib
import math

import torch


def validate_config(config):
    keys = {"first_layer", "last_layer", "threshold", "expected_kept", "mask_sha256"}
    optional = {"rescale_rms", "preserve_routing"}
    if not keys <= set(config) or set(config) - keys - optional:
        raise ValueError(
            "Norm channel mask requires layer range, threshold and mask pin"
        )
    for key in optional:
        if key in config and type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("first_layer", "last_layer", "expected_kept"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
    if (
        not 0 <= config["first_layer"] <= config["last_layer"]
        or config["expected_kept"] <= 0
    ):
        raise ValueError("Invalid layer range or kept channel count")
    value = config["threshold"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Threshold must be finite and positive")
    digest = config["mask_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Mask SHA256 must be lowercase hexadecimal")
    return config


def keep_mask(weights, config):
    """Compute the signed relative-weight intersection on CPU, checking its pin."""
    validate_config(config)
    if weights.ndim != 2 or not torch.isfinite(weights).all():
        raise ValueError("Expected finite layer-by-hidden normalization weights")
    values = weights.detach().cpu().double()
    rms = values.square().mean(dim=1, keepdim=True).sqrt()
    if (rms == 0).any():
        raise ValueError("Zero-RMS normalization weights")
    keep = (values / rms >= config["threshold"]).all(dim=0)
    digest = hashlib.sha256(keep.to(torch.uint8).numpy().tobytes()).hexdigest()
    if int(keep.sum()) != config["expected_kept"] or digest != config["mask_sha256"]:
        raise ValueError(
            "Loaded normalization weights do not match the pinned channel mask"
        )
    return keep


def mask_values(hidden, discard, rescale_rms=False):
    """Mask a copy and optionally preserve each token's original weighted RMS."""
    masked = hidden.masked_fill(discard, 0)
    if not rescale_rms:
        return masked
    before = hidden.float().square().sum(dim=-1, keepdim=True)
    after = masked.float().square().sum(dim=-1, keepdim=True)
    denominator = torch.where(after > 0, after, torch.ones_like(after))
    scale = (before / denominator).sqrt()
    scale = torch.where(after > 0, scale, torch.ones_like(scale))
    return (masked.float() * scale).to(hidden.dtype)


def apply(model, config):
    """Mask normalized outputs or expert inputs while preserving residuals."""
    validate_config(config)
    layers = model.model.layers
    first, last = config["first_layer"], config["last_layer"]
    if last >= len(layers):
        raise ValueError("Norm channel layer range exceeds model depth")
    norms = [layers[i].post_attention_layernorm for i in range(first, last + 1)]
    weights = torch.stack([norm.weight.detach().cpu() for norm in norms])
    if weights.shape[1] != model.config.hidden_size:
        raise ValueError("Unexpected normalization hidden width")
    keep = keep_mask(weights, config)
    rescale = config.get("rescale_rms", False)
    preserve = config.get("preserve_routing", False)
    if preserve:
        experts = [layers[i].mlp.experts for i in range(first, last + 1)]
        if any(getattr(module, "gate", None) is not None for module in experts):
            raise ValueError("Preserved routing requires externally computed logits")
    for i, norm in enumerate(norms):
        discard = (~keep).to(norm.weight.device)

        if preserve:

            def mask_expert_input(module, args, kwargs, *, discard=discard):
                if "hidden_states" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = mask_values(
                        kwargs["hidden_states"], discard, rescale
                    )
                    return args, kwargs
                if not args:
                    raise ValueError("Missing expert hidden_states input")
                return (mask_values(args[0], discard, rescale), *args[1:]), kwargs

            experts[i].register_forward_pre_hook(mask_expert_input, with_kwargs=True)
        else:

            def mask_output(module, inputs, output, *, discard=discard):
                if isinstance(output, tuple):
                    hidden, residual = output
                    return mask_values(hidden, discard, rescale), residual
                return mask_values(output, discard, rescale)

            norm.register_forward_hook(mask_output)
    return {
        "config": config,
        "site": "expert input after external routing"
        if preserve
        else "post_attention_layernorm output before MoE router and experts",
        "rescale_rms": rescale,
        "preserve_routing": preserve,
        "routing_reference": "unmasked current-layer norm output"
        if preserve
        else "masked norm output",
        "zero_energy_policy": "leave zero vector unchanged",
        "layers": list(range(first, last + 1)),
        "hidden_size": len(keep),
        "kept_channels": keep.nonzero().flatten().tolist(),
        "zeroed_channels_per_layer": int((~keep).sum()),
        "residual": "preserved",
    }
