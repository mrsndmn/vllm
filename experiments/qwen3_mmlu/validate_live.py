"""Capture original outputs or validate an instrumented local server."""

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
import uuid

from run import ARTIFACTS, CONFIG, URL, get_json

PROMPTS = ["The capital of France is", "A number " * 600 + " equals"]
REFERENCE = ARTIFACTS / "rounding-reference.json"


def request(body, path="/v1/completions"):
    body = {"model": CONFIG["model"], "temperature": 0, "max_tokens": 3, **body}
    req = urllib.request.Request(
        URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=240) as response:
        return response.read() if body.get("stream") else json.load(response)


def signature(response):
    return [(choice["text"], choice.get("logprobs")) for choice in response["choices"]]


def check_counts(response, config):
    stats = response["rounding_statistics"]
    layers = stats["layers"]
    assert len(layers) == 48, stats
    width = 6144 if config["site"] == "mlp" else 2048
    assert len({layer["tokens"] for layer in layers}) == 1
    for layer in layers:
        assert layer["tokens"] > 0
        assert layer["floats"] == layer["tokens"] * width
        assert 0 <= layer["newly_zeroed_floats"] <= layer["floats"]
        assert 0 <= layer["affected_tokens"] <= layer["tokens"]
        if config["epsilon"] == 0:
            assert layer["newly_zeroed_floats"] == 0
    if config["epsilon"] > 0:
        assert sum(layer["newly_zeroed_floats"] for layer in layers) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "check"])
    args = parser.parse_args()
    deadline = time.monotonic() + 600
    while True:
        try:
            get_json("/v1/models")
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    salt = uuid.uuid4().hex
    responses = [
        request({"prompt": prompt, "logprobs": 3, "cache_salt": salt})
        for prompt in PROMPTS
    ]
    if args.action == "capture":
        if REFERENCE.exists():
            raise RuntimeError("Reference already exists; preserve the original")
        REFERENCE.write_text(json.dumps(responses, indent=2))
        print(f"Captured original server outputs: {REFERENCE}")
        return
    config = get_json("/rounding/config")
    for response in responses:
        check_counts(response, config)
    if config["epsilon"] == 0:
        original = json.loads(REFERENCE.read_text())
        assert [signature(r) for r in original] == [signature(r) for r in responses]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(request, {"prompt": p, "cache_salt": salt}) for p in PROMPTS
        ]
        concurrent_responses = [future.result() for future in futures]
    for response in concurrent_responses:
        check_counts(response, config)
    assert concurrent_responses[0]["id"] != concurrent_responses[1]["id"]
    processed_and_cached = [
        r["rounding_statistics"]["layers"][0]["tokens"]
        + r["rounding_statistics"]["initial_cached_tokens_excluded"]
        for r in concurrent_responses
    ]
    assert processed_and_cached[0] < processed_and_cached[1]
    multiple = request(
        {"prompt": ["One", "Two"], "n": 2, "temperature": 0.1, "seed": 0}
    )
    check_counts(multiple, config)
    assert multiple["rounding_statistics"]["engine_sequences"] == 4
    chat = request(
        {"messages": [{"role": "user", "content": "Say hello"}]}, "/v1/chat/completions"
    )
    check_counts(chat, config)
    stream = request({"prompt": "Three", "stream": True})
    events = [
        event[6:] for event in stream.split(b"\n\n") if event.startswith(b"data: ")
    ]
    assert events[-1] == b"[DONE]"
    check_counts(json.loads(events[-2]), config)
    try:
        request({"prompt": "Invalid", "n": 0})
    except urllib.error.HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Invalid request was accepted")
    result = {
        "configuration": config,
        "checks": "passed",
        "responses": responses,
        "multi_prompt": multiple,
        "chat": chat,
        "stream_statistics": json.loads(events[-2]),
    }
    path = ARTIFACTS / f"validation-{config['site']}-{config['epsilon']}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Live checks passed: {path}")


if __name__ == "__main__":
    main()
