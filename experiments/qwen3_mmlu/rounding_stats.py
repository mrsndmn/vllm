"""Thresholding and request-scoped counters for the local rounding experiment."""

import json
import math
import sqlite3

import torch

FIELDS = (
    "floats",
    "newly_zeroed_floats",
    "already_zero_floats",
    "tokens",
    "affected_tokens",
    "all_zero_tokens_after",
)
SITES = ("mlp", "block", "moe_out")


def layer_operands(site, hidden, residual):
    """(values, residual) a decoder-layer output site passes to ``threshold``.

    ``hidden`` is the layer's MoE output and ``residual`` the stream it is added to.
    ``block`` tests their sum and zeroes both; ``moe_out`` tests and zeroes the MoE
    output alone, leaving the residual stream untouched.
    """
    if site == "block":
        return hidden, residual
    if site == "moe_out":
        return hidden, None
    raise ValueError(f"Site {site!r} is not a decoder-layer output site")


def validate_eps(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("round_activations_eps must be finite and nonnegative")
    return value


def validate_quantile(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("round_activations_quantile must be in [0, 1)")
    return value


def validate_schedule(values):
    """Per-group quantiles; groups split the layers into equal consecutive runs."""
    values = [validate_quantile(value) for value in values]
    if not values:
        raise ValueError("round_activations_quantiles must not be empty")
    return values


def group_quantile(schedule, layer, layers):
    if layers % len(schedule):
        raise ValueError("Quantile groups must divide the layer count evenly")
    return schedule[layer * len(schedule) // layers]


def threshold(values, eps, residual=None, quantile=0.0):
    """Modify selected values in place; return per-token int64 counters.

    ``values`` has one row per token, including all selected experts for MLPs.
    Residual sums are tested in FP32, as in vLLM's fused add/RMSNorm. Keeping
    both original operands outside the mask preserves the fused arithmetic.
    A nonzero ``quantile`` q replaces ``eps`` with a per-row cutoff: the row's
    floor(q * width)-th smallest magnitude; ties at the cutoff are zeroed too.
    Epsilon and quantile zero are an exact no-op, including signed zeros and NaNs.
    """
    observed = values.float()
    if residual is not None:
        observed = observed + residual.float()
    zero = observed == 0
    width = values.shape[1]
    rank = math.floor(quantile * width)
    if rank and values.shape[0]:
        magnitude = observed.abs()
        cutoff = magnitude.kthvalue(rank, dim=1, keepdim=True).values
        # kthvalue ranks NaN last, so a NaN cutoff means every finite value is below it.
        limit = torch.where(cutoff.isnan(), torch.inf, cutoff)
    elif eps:
        magnitude = observed.abs()
        limit = min(eps, torch.finfo(torch.float32).max)
    else:
        limit = None
    if limit is None:
        mask = torch.zeros_like(zero)
    else:
        mask = (magnitude <= limit) & ~zero & torch.isfinite(observed)
        values.masked_fill_(mask, 0)
        if residual is not None:
            residual.masked_fill_(mask, 0)
    changed = mask.sum(dim=1, dtype=torch.int64)
    existing = zero.sum(dim=1, dtype=torch.int64)
    return torch.stack(
        (
            torch.full_like(changed, width),
            changed,
            existing,
            torch.ones_like(changed),
            (changed > 0).to(torch.int64),
            (changed + existing == width).to(torch.int64),
        ),
        dim=1,
    )


def connect(path):
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS requests "
        "(id TEXT PRIMARY KEY, response_id TEXT, counts TEXT, skipped INTEGER)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS response_idx ON requests(response_id)")
    return db


def format_layers(counts):
    layers = []
    for index, row in enumerate(counts):
        layer = dict(zip(FIELDS, row))
        layer["layer"] = index
        layer["newly_zeroed_float_percent"] = 100 * row[1] / row[0] if row[0] else 0.0
        layer["affected_token_percent"] = 100 * row[4] / row[3] if row[3] else 0.0
        layers.append(layer)
    return layers


def read_stats(path, site, eps, response_id=None, quantile=0.0, quantiles=None):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        sql = "SELECT counts, skipped FROM requests"
        params = ()
        if response_id is not None:
            sql += " WHERE response_id = ?"
            params = (response_id,)
        totals = None
        skipped = requests = 0
        for raw, cached in db.execute(sql, params):
            counts = json.loads(raw)
            if totals is None:
                totals = [[0] * len(FIELDS) for _ in counts]
            for total, row in zip(totals, counts):
                for i, value in enumerate(row):
                    total[i] += value
            skipped += cached
            requests += 1
    return {
        "site": site,
        "epsilon": eps,
        "quantile": quantile,
        "quantiles": quantiles,
        "coverage": "executed token positions; cached prefixes excluded; "
        "recomputed positions counted again; last sampled token is not forwarded",
        "engine_sequences": requests,
        "initial_cached_tokens_excluded": skipped,
        "layers": format_layers(totals or []),
    }
