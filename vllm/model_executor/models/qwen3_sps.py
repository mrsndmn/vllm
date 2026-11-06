# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request Qwen3 SPS with shared-backbone MTP and real-only paged KV."""

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.ops.sps_prediction import sps_prediction_attention

from .qwen3 import Qwen3ForCausalLM


class Qwen3SPSForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        scheduler = vllm_config.scheduler_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or scheduler.max_num_seqs != 1
        ):
            raise ValueError("SPS pilot requires TP=PP=DP=1 and max_num_seqs=1")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("SPS pilot requires enforce_eager=True")
        if (
            scheduler.enable_chunked_prefill
            or scheduler.async_scheduling
            or vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError(
                "SPS pilot requires chunked prefill, async scheduling "
                "and prefix caching disabled"
            )
        if vllm_config.use_v2_model_runner:
            raise ValueError("SPS pilot requires VLLM_USE_V2_MODEL_RUNNER=0")
        if vllm_config.quant_config is not None:
            raise ValueError("SPS pilot does not support quantized weights")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.sps_prediction_embeddings = nn.Parameter(
            torch.empty(self.config.sps_horizons, self.config.hidden_size)
        )
        self.sps_horizons = (
            self.config.sps_horizons if vllm_config.speculative_config else 1
        )
        self.sps_hidden = None
        self.sps_positions = None
        self.sps_forward_count = 0
        for layer in self.model.layers:
            attention = layer.self_attn.attn
            if type(attention.impl).__name__ != "TritonAttentionImpl":
                raise ValueError(
                    "SPS pilot requires attention_config.backend=TRITON_ATTN"
                )
            if attention.kv_cache_dtype not in {"auto", "bfloat16", "float16"}:
                raise ValueError("SPS pilot requires unquantized KV cache")

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("SPS supports text input_ids only")
        tokens = input_ids.shape[0]
        n = self.sps_horizons
        width = n + 1
        real = self.model.embed_input_ids(input_ids)
        predictions = self.sps_prediction_embeddings[:n].to(real.dtype)
        x = torch.cat(
            (real[:, None], predictions[None].expand(tokens, -1, -1)), 1
        ).flatten(0, 1)
        rope_positions = (
            positions[:, None] + torch.arange(width, device=positions.device)[None]
        ).flatten()
        metadata = get_forward_context().attn_metadata
        residual = None
        for layer in self.model.layers:
            if residual is None:
                residual = x
                normalized = layer.input_layernorm(x)
            else:
                normalized, residual = layer.input_layernorm(x, residual)
            attention = layer.self_attn
            qkv, _ = attention.qkv_proj(normalized)
            q, k, v = qkv.split(
                [attention.q_size, attention.kv_size, attention.kv_size], -1
            )
            q = attention.q_norm(
                q.reshape(-1, attention.num_heads, attention.head_dim)
            ).reshape(q.shape)
            k = attention.k_norm(
                k.reshape(-1, attention.num_kv_heads, attention.head_dim)
            ).reshape(k.shape)
            q, k = attention.rotary_emb(rope_positions, q, k)
            q = q.reshape(tokens, width, attention.num_heads, attention.head_dim)
            k = k.reshape(tokens, width, attention.num_kv_heads, attention.head_dim)
            v = v.reshape(tokens, width, attention.num_kv_heads, attention.head_dim)
            real_output = attention.attn(
                q[:, 0].reshape(tokens, -1),
                k[:, 0].reshape(tokens, -1),
                v[:, 0].reshape(tokens, -1),
            )
            if metadata is None:
                prediction_output = torch.zeros_like(q[:, 1:])
            else:
                actual = metadata[attention.attn.layer_name]
                assert isinstance(actual, TritonAttentionMetadata)
                prediction_output = sps_prediction_attention(
                    q[:, 1:],
                    k[:, 1:],
                    v[:, 1:],
                    attention.attn.kv_cache,
                    actual.block_table,
                    positions,
                )
            joined = torch.cat(
                (
                    real_output.reshape(
                        tokens, 1, attention.num_heads, attention.head_dim
                    ),
                    prediction_output,
                ),
                1,
            ).reshape(tokens * width, -1)
            x, _ = attention.o_proj(joined)
            x, residual = layer.post_attention_layernorm(x, residual)
            x = layer.mlp(x)
        x, _ = self.model.norm(x, residual)
        prediction_hidden = x.reshape(tokens, width, -1)[:, 1:]
        self.sps_hidden = prediction_hidden
        self.sps_positions = positions.clone()
        self.sps_forward_count += 1
        return prediction_hidden[:, 0].contiguous()
