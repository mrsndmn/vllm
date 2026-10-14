# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ASGI extension exposing rounding counters on completed local API requests."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from rounding_stats import read_stats


class RoundingMiddleware:
    def __init__(self, app):
        self.app = app
        self.path = os.environ["ROUND_ACTIVATIONS_DB"]
        self.site = os.environ["ROUND_ACTIVATIONS_SITE"]
        self.replacement = os.environ.get("ROUND_ACTIVATIONS_REPLACEMENT", "zero")
        mlp = os.environ.get("MLP_WEIGHT_ROUNDING")
        self.mlp_weight_rounding = json.loads(mlp) if mlp else None
        self.eps = float(os.environ["ROUND_ACTIVATIONS_EPS"])
        self.quantile = float(os.environ.get("ROUND_ACTIVATIONS_QUANTILE", "0"))
        schedule = os.environ.get("ROUND_ACTIVATIONS_QUANTILES")
        self.quantiles = json.loads(schedule) if schedule else None
        weights = os.environ.get("EMBEDDING_WEIGHT_ZEROING")
        self.weight_zeroing = json.loads(weights) if weights else None
        channels = os.environ.get("POST_NORM_CHANNELS")
        self.post_norm_channels = json.loads(channels) if channels else None

    async def stats(self, response_id=None):
        result = await asyncio.to_thread(
            read_stats,
            self.path,
            self.site,
            self.eps,
            response_id,
            self.quantile,
            self.quantiles,
        )
        result["replacement"] = self.replacement
        return result

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path in ("/rounding/config", "/rounding/stats"):
            if path.endswith("stats"):
                result = await self.stats()
            else:
                result = {
                    "site": self.site,
                    "epsilon": self.eps,
                    "quantile": self.quantile,
                    "quantiles": self.quantiles,
                    "database": self.path,
                    "replacement": self.replacement,
                }
                if self.mlp_weight_rounding:
                    result["mlp_weight_rounding"] = self.mlp_weight_rounding
                    result["mlp_weight_rounding_report"] = json.loads(
                        Path(self.path).with_suffix(".mlp-weights.json").read_text()
                    )
                if self.post_norm_channels:
                    result["post_norm_channels"] = self.post_norm_channels
                    result["post_norm_channels_report"] = json.loads(
                        Path(self.path).with_suffix(".norm-channels.json").read_text()
                    )
                if self.weight_zeroing:
                    result["weight_zeroing"] = self.weight_zeroing
                    result["weight_zeroing_report"] = json.loads(
                        Path(self.path).with_suffix(".weights.json").read_text()
                    )
            body = json.dumps(result).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        if path not in ("/v1/completions", "/v1/chat/completions"):
            return await self.app(scope, receive, send)
        uid = uuid.uuid4().hex
        response_id = ("chatcmpl-" if "/chat/" in path else "cmpl-") + uid
        scope = dict(scope)
        scope["headers"] = [
            (key, value)
            for key, value in scope["headers"]
            if key.lower() != b"x-request-id"
        ] + [(b"x-request-id", uid.encode())]
        started = None
        buffer = b""
        streaming = False
        success = False
        stream_metadata = {}

        async def intercept(message):
            nonlocal started, buffer, streaming, success
            if message["type"] == "http.response.start":
                started = dict(message)
                success = message["status"] == 200
                streaming = any(
                    key == b"content-type" and b"text/event-stream" in value
                    for key, value in message["headers"]
                )
                started["headers"] = [
                    (k, v) for k, v in message["headers"] if k != b"content-length"
                ]
                await send(started)
                return
            if message["type"] != "http.response.body" or not success:
                await send(message)
                return
            buffer += message.get("body", b"")
            if streaming:
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    if event.strip() == b"data: [DONE]":
                        result = {
                            **stream_metadata,
                            "id": response_id,
                            "choices": [],
                            "rounding_statistics": await self.stats(response_id),
                        }
                        extra = b"data: " + json.dumps(result).encode() + b"\n\n"
                        await send(
                            {
                                "type": "http.response.body",
                                "body": extra,
                                "more_body": True,
                            }
                        )
                    elif event.startswith(b"data: "):
                        payload = json.loads(event[6:])
                        stream_metadata.update(
                            {
                                key: payload[key]
                                for key in ("object", "created", "model")
                                if key in payload
                            }
                        )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": event + b"\n\n",
                            "more_body": True,
                        }
                    )
                if not message.get("more_body", False):
                    await send({"type": "http.response.body", "body": buffer})
            elif not message.get("more_body", False):
                result = json.loads(buffer)
                result["rounding_statistics"] = await self.stats(response_id)
                await send(
                    {"type": "http.response.body", "body": json.dumps(result).encode()}
                )

        await self.app(scope, receive, intercept)
