# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The accepted prefix, not the last speculative row, selects the next group."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.sps_prediction import validate_sps_context
from vllm.v1.spec_decode.sps_proposer import SPSProposer


@pytest.mark.parametrize("horizons", [1, 4])
def test_prediction_positions_reserve_rotary_context(horizons):
    validate_sps_context(128 - horizons, horizons, 128)
    with pytest.raises(ValueError, match="RoPE"):
        validate_sps_context(129 - horizons, horizons, 128)


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_next_group_follows_accepted_prefix(accepted):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    proposer = SPSProposer(config)
    logits = torch.zeros(4, 4, 17)
    for row in range(4):
        for slot in range(4):
            logits[row, slot, row * 4 + slot] = 1
    proposer.model = SimpleNamespace(
        sps_hidden=logits,
        sps_positions=torch.arange(10, 14),
        compute_logits=lambda x: x,
    )
    result = proposer.propose([[7] * (accepted + 1)], [12 + accepted], None)
    assert result == [[accepted * 4 + s for s in (1, 2, 3)]]


def test_chunked_prefill_without_a_sample_emits_no_proposals():
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    assert SPSProposer(config).propose([[]], [10], None) == [[]]
