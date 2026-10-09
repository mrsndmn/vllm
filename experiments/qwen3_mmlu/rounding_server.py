# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch the installed vLLM server with experimental activation thresholding."""

import argparse
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path

from embedding_zeroing import validate_config
from rounding_stats import (
    REPLACEMENTS,
    SITES,
    connect,
    validate_eps,
    validate_quantile,
    validate_schedule,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--round-activations-eps",
        "--round_activations_eps",
        required=True,
        type=validate_eps,
    )
    parser.add_argument(
        "--round-activations-quantile",
        "--round_activations_quantile",
        default=0.0,
        type=validate_quantile,
    )
    parser.add_argument(
        "--round-activations-quantiles",
        "--round_activations_quantiles",
        type=lambda text: validate_schedule(text.split(",")),
        help="Comma-separated quantile per equal consecutive layer group",
    )
    parser.add_argument("--round-activations-site", choices=SITES, required=True)
    parser.add_argument("--rounding-db", type=Path, required=True)
    parser.add_argument(
        "--weight-zeroing", type=lambda text: validate_config(json.loads(text))
    )
    parser.add_argument(
        "--round-activations-replacement", choices=REPLACEMENTS, default="zero"
    )
    from mlp_weight_rounding import validate_config as validate_mlp_config

    parser.add_argument(
        "--mlp-weight-rounding", type=lambda text: validate_mlp_config(json.loads(text))
    )
    args, server_args = parser.parse_known_args()
    if args.round_activations_replacement != "zero" and (
        args.round_activations_site != "mlp"
        or args.round_activations_eps
        or args.round_activations_quantiles
    ):
        parser.error("Signed epsilon supports uniform MLP quantiles only")
    modes = (
        args.round_activations_eps,
        args.round_activations_quantile,
        args.round_activations_quantiles,
    )
    if sum(map(bool, modes)) > 1:
        parser.error("use one of epsilon, quantile, or quantile schedule")
    if importlib.metadata.version("vllm") != "0.14.1":
        raise RuntimeError("This experimental wrapper requires installed vLLM 0.14.1")
    if args.rounding_db.exists():
        raise RuntimeError("Use a fresh rounding database for every server start")
    args.rounding_db.parent.mkdir(parents=True, exist_ok=True)
    connect(args.rounding_db).close()
    os.environ.update(
        {
            "ROUND_ACTIVATIONS_REPLACEMENT": args.round_activations_replacement,
            "MLP_WEIGHT_ROUNDING": json.dumps(args.mlp_weight_rounding)
            if args.mlp_weight_rounding
            else "",
            "ROUND_ACTIVATIONS_EPS": str(args.round_activations_eps),
            "ROUND_ACTIVATIONS_QUANTILE": str(args.round_activations_quantile),
            "ROUND_ACTIVATIONS_QUANTILES": json.dumps(args.round_activations_quantiles)
            if args.round_activations_quantiles
            else "",
            "ROUND_ACTIVATIONS_SITE": args.round_activations_site,
            "ROUND_ACTIVATIONS_DB": str(args.rounding_db.resolve()),
            "EMBEDDING_WEIGHT_ZEROING": json.dumps(args.weight_zeroing)
            if args.weight_zeroing
            else "",
            "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        }
    )
    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        *server_args,
        "--worker-cls",
        "rounding_worker.RoundingWorker",
        "--middleware",
        "rounding_api.RoundingMiddleware",
    ]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
