# Qwen3 SPS-MTP pilot

This experimental single-request implementation uses one Qwen3 backbone for
ordinary SPS prediction and MTP proposals. It requires an exported SPS checkpoint
whose architecture is `Qwen3SPSForCausalLM` and whose additional
`sps_prediction_embeddings` parameter contains the trained horizon embeddings.

The first slot defines the SPS verifier. Later slots produce deterministic
proposals for the tokens after the pending first-slot sample. vLLM's ordinary
rejection sampler handles these delta-distribution proposals, including when the
target uses temperature/top-p/top-k sampling. Extra slots see earlier prediction
states within their group, not sampled future tokens. No second draft model is
loaded.

Each target forward processes the real input block and prediction groups through
the same layers. The first slot verifies the candidate block in one traversal;
later slots prepare the next proposals. Only real-token KV is written into the
paged cache. Prediction attention reads causal real KV and its own group's causal
prediction slots. Real-token states never read prediction states.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/exported-sps-checkpoint",
    max_num_seqs=1,
    tensor_parallel_size=1,
    enforce_eager=True,
    async_scheduling=False,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    attention_config={"backend": "TRITON_ATTN"},
    speculative_config={
        "method": "custom_class",
        "model": "vllm.v1.spec_decode.sps_proposer.SPSProposer",
        "num_speculative_tokens": 3,
    },
)
outputs = llm.generate(["Hello"], SamplingParams(temperature=0.7, max_tokens=32))
```

Set `VLLM_USE_V2_MODEL_RUNNER=0` before starting the process. The draft length
must be smaller than the number of trained prediction embeddings. Omit
`speculative_config` to benchmark ordinary first-slot SPS with the same weights.
The pilot supports unquantized FP16/BF16 weights and KV, one request, one GPU,
eager execution and the V1 GPU model runner. Other configurations fail closed.
Production batching, CUDA graph capture, quantization and distributed execution
are outside this pilot. No speedup or unchanged parent-model quality is claimed.

The SPS experiment repository owns training/export and MMLU benchmarking. Pin
both its commit and this vLLM commit in the run manifest. Fresh main's Python and
Torch requirements and matching compiled extensions apply to this implementation;
an older installed vLLM is not a substitute for the modified source.
