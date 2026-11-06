# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse target MTP states as deterministic speculative proposals.

The newly sampled first-slot token is a pending anchor. Later slots in the
same prefix group propose its successors. vLLM's existing rejection sampler
handles these delta-distribution proposals while sampling the SPS target.
"""

import torch


class SPSProposer:
    def __init__(self, vllm_config):
        self.num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("SPS proposer requires one request")
        self.model = None

    def load_model(self, model):
        from vllm.model_executor.models.qwen3_sps import Qwen3SPSForCausalLM

        if not isinstance(model, Qwen3SPSForCausalLM):
            raise ValueError("SPS proposer requires Qwen3SPSForCausalLM")
        if not 0 < self.num_spec_tokens < model.sps_horizons:
            raise ValueError("Requested draft length exceeds trained MTP horizon")
        self.model = model

    @torch.inference_mode()
    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, **kwargs):
        if len(sampled_token_ids) != 1:
            raise ValueError("SPS proposer requires one request")
        if not sampled_token_ids[0]:
            return [[]]
        model = self.model
        if model is None or model.sps_hidden is None:
            raise RuntimeError("SPS target has not produced MTP states")
        # Bookkeeping includes the last sampled correction/bonus, which has
        # not yet been encoded. Reuse the group immediately before that token.
        absolute = int(num_tokens_no_spec[0]) - 2
        row = absolute - int(model.sps_positions[0].item())
        if not 0 <= row < model.sps_hidden.shape[0]:
            raise RuntimeError("Accepted prefix is outside the latest SPS block")
        hidden = model.sps_hidden[row, 1 : self.num_spec_tokens + 1]
        logits = model.compute_logits(hidden)
        return [logits.argmax(-1).tolist()]
