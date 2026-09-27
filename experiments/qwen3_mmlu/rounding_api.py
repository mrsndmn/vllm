"""ASGI extension exposing rounding counters on completed local API requests."""

import asyncio
import json
import os
import uuid

from rounding_stats import read_stats


class RoundingMiddleware:
    def __init__(self, app):
        self.app = app
        self.path = os.environ["ROUND_ACTIVATIONS_DB"]
        self.site = os.environ["ROUND_ACTIVATIONS_SITE"]
        self.eps = float(os.environ["ROUND_ACTIVATIONS_EPS"])

    async def stats(self, response_id=None):
        return await asyncio.to_thread(
            read_stats, self.path, self.site, self.eps, response_id
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path in ("/rounding/config", "/rounding/stats"):
            if path.endswith("stats"):
                result = await self.stats()
            else:
                result = {"site": self.site, "epsilon": self.eps, "database": self.path}
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
