# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep a pinned intersection of large RMSNorm channels at selected decoder sites."""

import hashlib
import math

import torch


def validate_config(config):
    keys = {"first_layer", "last_layer", "threshold", "expected_kept", "mask_sha256"}
    if set(config) != keys:
        raise ValueError(
            "Norm channel mask requires layer range, threshold and mask pin"
        )
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


def apply(model, config):
    """Mask normalized hidden states before MoE routing; preserve residual tensors."""
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
    for norm in norms:
        discard = (~keep).to(norm.weight.device)

        def mask_output(module, inputs, output, *, discard=discard):
            if isinstance(output, tuple):
                hidden, residual = output
                return hidden.masked_fill(discard, 0), residual
            return output.masked_fill(discard, 0)

        norm.register_forward_hook(mask_output)
    return {
        "config": config,
        "site": "post_attention_layernorm output before MoE router and experts",
        "layers": list(range(first, last + 1)),
        "hidden_size": len(keep),
        "kept_channels": keep.nonzero().flatten().tolist(),
        "zeroed_channels_per_layer": int((~keep).sum()),
        "residual": "preserved",
    }
