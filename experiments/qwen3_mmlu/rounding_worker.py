"""Version-pinned vLLM worker instrumentation; no installed files are modified."""

import functools
import json
import os
import re

import numpy as np
import torch
from rounding_stats import (
    FIELDS,
    SITES,
    connect,
    group_quantile,
    layer_operands,
    threshold,
    validate_eps,
    validate_quantile,
    validate_schedule,
)

from vllm.v1.worker.gpu_worker import Worker


class RoundingWorker(Worker):
    def load_model(self):
        super().load_model()
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

        runner = self.model_runner
        config = self.vllm_config
        model = runner.model
        if not isinstance(model, Qwen3MoeForCausalLM):
            raise ValueError("Rounding experiment supports Qwen3MoeForCausalLM only")
        if (
            not config.model_config.enforce_eager
            or config.parallel_config.world_size != 1
            or config.parallel_config.data_parallel_size != 1
            or config.speculative_config is not None
            or config.lora_config is not None
            or config.model_config.quantization is not None
            or self.use_v2_model_runner
        ):
            raise ValueError("Rounding requires eager, unquantized, single-GPU V1")
        self.eps = validate_eps(os.environ["ROUND_ACTIVATIONS_EPS"])
        self.quantile = validate_quantile(
            os.environ.get("ROUND_ACTIVATIONS_QUANTILE", "0")
        )
        schedule = os.environ.get("ROUND_ACTIVATIONS_QUANTILES")
        self.schedule = validate_schedule(json.loads(schedule)) if schedule else None
        if sum(map(bool, (self.eps, self.quantile, self.schedule))) > 1:
            raise ValueError("Use one of epsilon, quantile, or quantile schedule")
        self.site = os.environ["ROUND_ACTIVATIONS_SITE"]
        if self.site not in SITES:
            raise ValueError(f"Unknown rounding site {self.site!r}")
        self.stats_db = connect(os.environ["ROUND_ACTIVATIONS_DB"])
        self.active_counts = {}
        self.batch_ids = []
        self.batch_sizes = []
        self.layer_counts = {}
        self.executing = False
        self.layers = len(model.model.layers)
        if self.schedule:
            group_quantile(self.schedule, 0, self.layers)
        for index, layer in enumerate(model.model.layers):
            experts = layer.mlp.experts.quant_method.kernel.fused_experts
            if type(experts) is not TritonExperts:
                raise ValueError("Rounding requires the unquantized TritonExperts path")
            if self.site == "mlp":
                original = experts.activation

                def activation(kind, output, inputs, *, original=original, idx=index):
                    original(kind, output, inputs)
                    top_k = model.config.num_experts_per_tok
                    view = output.view(-1, top_k * output.shape[-1])
                    self.record(idx, view)

                experts.activation = activation
            else:

                def block_hook(module, inputs, output, *, idx=index):
                    hidden, residual = output
                    self.record(idx, *layer_operands(self.site, hidden, residual))
                    return hidden, residual

                layer.register_forward_hook(block_hook)

        prepare = runner._prepare_inputs

        @functools.wraps(prepare)
        def prepare_inputs(scheduler_output, *args, **kwargs):
            result = prepare(scheduler_output, *args, **kwargs)
            self.batch_ids = list(runner.input_batch.req_ids)
            self.batch_sizes = [
                scheduler_output.num_scheduled_tokens[rid] for rid in self.batch_ids
            ]
            for i, rid in enumerate(self.batch_ids):
                if rid not in self.active_counts:
                    skipped = min(
                        int(runner.input_batch.num_computed_tokens_cpu[i]),
                        int(runner.input_batch.num_prompt_tokens[i]),
                    )
                    counts = np.zeros((self.layers, len(FIELDS)), dtype=np.int64)
                    self.active_counts[rid] = (counts, skipped)
            return result

        runner._prepare_inputs = prepare_inputs
        execute = runner.execute_model

        @functools.wraps(execute)
        def execute_model(scheduler_output, *args, **kwargs):
            for rid in scheduler_output.finished_req_ids:
                self.active_counts.pop(rid, None)
            self.layer_counts = {}
            self.batch_ids = []
            self.executing = True
            try:
                result = execute(scheduler_output, *args, **kwargs)
                self.flush()
                return result
            finally:
                self.executing = False

        runner.execute_model = execute_model

    def record(self, layer, values, residual=None):
        quantile = self.quantile
        if self.schedule:
            quantile = group_quantile(self.schedule, layer, self.layers)
        counts = threshold(values, self.eps, residual, quantile)
        if self.executing:
            expected = sum(self.batch_sizes)
            if counts.shape[0] != expected or layer in self.layer_counts:
                raise RuntimeError(
                    "Unexpected padded/chunked/duplicate activation layout"
                )
            self.layer_counts[layer] = torch.stack(
                [part.sum(dim=0) for part in counts.split(self.batch_sizes)]
            )

    def flush(self):
        if not self.batch_ids:
            return
        if set(self.layer_counts) != set(range(self.layers)):
            raise RuntimeError("Not all model layers were instrumented")
        counts = torch.stack([self.layer_counts[i] for i in range(self.layers)])
        counts = counts.cpu().numpy()
        rows = []
        for i, rid in enumerate(self.batch_ids):
            total, skipped = self.active_counts[rid]
            total += counts[:, i, :]
            match = re.search(r"(?:^|_)((?:chatcmpl|cmpl)-[a-f0-9]{32})(?:-|$)", rid)
            response_id = match[1] if match else rid
            rows.append((rid, response_id, json.dumps(total.tolist()), skipped))
        with self.stats_db:
            self.stats_db.executemany(
                "INSERT OR REPLACE INTO requests VALUES (?, ?, ?, ?)", rows
            )
