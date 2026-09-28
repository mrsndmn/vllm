"""Launch the installed vLLM server with experimental activation thresholding."""

import argparse
import importlib.metadata
import os
import runpy
import sys
from pathlib import Path

from rounding_stats import connect, validate_eps, validate_quantile


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
        "--round-activations-site", choices=["mlp", "block"], required=True
    )
    parser.add_argument("--rounding-db", type=Path, required=True)
    args, server_args = parser.parse_known_args()
    if args.round_activations_eps and args.round_activations_quantile:
        parser.error("use either an epsilon or a quantile, not both")
    if importlib.metadata.version("vllm") != "0.14.1":
        raise RuntimeError("This experimental wrapper requires installed vLLM 0.14.1")
    if args.rounding_db.exists():
        raise RuntimeError("Use a fresh rounding database for every server start")
    args.rounding_db.parent.mkdir(parents=True, exist_ok=True)
    connect(args.rounding_db).close()
    os.environ.update(
        {
            "ROUND_ACTIVATIONS_EPS": str(args.round_activations_eps),
            "ROUND_ACTIVATIONS_QUANTILE": str(args.round_activations_quantile),
            "ROUND_ACTIVATIONS_SITE": args.round_activations_site,
            "ROUND_ACTIVATIONS_DB": str(args.rounding_db.resolve()),
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
