# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prediction queries over real paged KV and ephemeral group-local slots."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prediction_attention(
    Q,
    PK,
    PV,
    CACHE,
    TABLE,
    POS,
    OUT,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    CS3: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    token = row // N
    slot = row % N
    kvhead = head // (H // HKV)
    d = tl.arange(0, BD)
    q = tl.load(Q + (row * H + head) * D + d, d < D, other=0)
    p = tl.arange(0, BP)
    offsets = ((token * N + p[:, None]) * HKV + kvhead) * D + d[None, :]
    k = tl.load(PK + offsets, (p[:, None] <= slot) & (d[None, :] < D), other=0)
    v = tl.load(PV + offsets, (p[:, None] <= slot) & (d[None, :] < D), other=0)
    score = tl.sum(k.to(tl.float32) * q[None, :].to(tl.float32), 1) * SCALE
    score = tl.where(p <= slot, score, -float("inf"))
    maximum = tl.max(score, 0)
    prob = tl.exp(score - maximum)
    norm = tl.sum(prob, 0)
    accum = tl.sum(prob[:, None] * v.to(tl.float32), 0)
    prefix = tl.load(POS + token) + 1
    j = tl.arange(0, BK)
    for start in range(0, prefix, BK):
        index = start + j
        page = tl.load(TABLE + index // PAGE, index < prefix, other=0)
        cache_offsets = (
            page[:, None] * CS0
            + kvhead * CS1
            + (index[:, None] % PAGE) * CS2
            + d[None, :] * CS3
        )
        valid = (index[:, None] < prefix) & (d[None, :] < D)
        cache_k = tl.load(CACHE + cache_offsets, valid, other=0)
        cache_v = tl.load(CACHE + cache_offsets + D * CS3, valid, other=0)
        cache_score = (
            tl.sum(cache_k.to(tl.float32) * q[None, :].to(tl.float32), 1) * SCALE
        )
        cache_score = tl.where(index < prefix, cache_score, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(cache_score, 0))
        alpha = tl.exp(maximum - new_maximum)
        cache_prob = tl.exp(cache_score - new_maximum)
        accum = accum * alpha + tl.sum(cache_prob[:, None] * cache_v.to(tl.float32), 0)
        norm = norm * alpha + tl.sum(cache_prob, 0)
        maximum = new_maximum
    tl.store(OUT + (row * H + head) * D + d, accum / norm, d < D)


def sps_prediction_attention(
    q, prediction_k, prediction_v, cache, block_table, positions
):
    """Attend to real positions <= each prefix plus causal prediction slots.

    Supports one unpadded request and the unquantized Triton KV cache layout
    [pages, kv_heads, page_size, 2*head_dim]. Prediction KV is never persisted.
    """
    tokens, horizons, heads, dim = q.shape
    kvheads = prediction_k.shape[2]
    if (
        block_table.shape[0] != 1
        or cache.ndim != 4
        or cache.shape[1] != kvheads
        or cache.shape[-1] != 2 * dim
        or heads % kvheads
        or positions.numel() != tokens
    ):
        raise ValueError("SPS requires one request and unquantized Triton KV")
    if prediction_k.shape != prediction_v.shape or prediction_k.shape[:2] != (
        tokens,
        horizons,
    ):
        raise ValueError("Prediction KV does not match the query groups")
    q = q.contiguous()
    prediction_k = prediction_k.contiguous()
    prediction_v = prediction_v.contiguous()
    output = torch.empty_like(q)
    _prediction_attention[(tokens * horizons, heads)](
        q,
        prediction_k,
        prediction_v,
        cache,
        block_table[0],
        positions,
        output,
        *cache.stride(),
        horizons,
        heads,
        kvheads,
        dim,
        cache.shape[2],
        dim**-0.5,
        triton.next_power_of_2(dim),
        triton.next_power_of_2(horizons),
        64,
    )
    return output
