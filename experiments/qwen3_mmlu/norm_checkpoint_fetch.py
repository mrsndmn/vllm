# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fetch an indexed checkpoint's decoder norms using strict HTTP byte ranges."""

import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import regex as re
import requests
import torch
from huggingface_hub import hf_hub_url
from safetensors.torch import save_file


def read_range(session, url, first, last):
    """Reject full-file responses before reading any weight payload."""
    expected = last - first + 1
    with session.get(
        url + f"?norm_range={first}-{last}",
        headers={
            "Range": f"bytes={first}-{last}",
            "Accept-Encoding": "identity",
            "User-Agent": "vllm-norm-analysis",
        },
        stream=True,
        timeout=(10, 60),
    ) as response:
        match = re.fullmatch(
            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
        )
        if (
            response.status_code != 206
            or not match
            or (int(match[1]), int(match[2])) != (first, last)
            or int(match[3]) <= last
        ):
            raise ValueError("Server did not honor the requested tensor byte range")
        data = response.raw.read(expected + 1)
        if len(data) != expected:
            raise ValueError("Unexpected tensor byte-range length")
        return data


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_metadata(config, filename):
    """Read a bounded JSON metadata file at the pinned revision."""
    url = hf_hub_url(config["model"], filename, revision=config["revision"])
    limit = config["max_remote_metadata_bytes"]
    with requests.get(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "vllm-norm-analysis"},
        stream=True,
        timeout=(10, 60),
    ) as response:
        if response.status_code != 200:
            raise ValueError(f"Could not retrieve checkpoint metadata: {filename}")
        data = response.raw.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Checkpoint metadata exceeds the configured limit")
        return data


def fetch_norm_checkpoint(config, families):
    """Persist raw norm tensors and source metadata in a separate subset cache."""
    target = Path(config["snapshot"])
    marker = target / "norm-source.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["model"] != config["model"] or saved["revision"] != config["revision"]:
            raise ValueError("Normalization subset cache has a different source")
        for filename, digest in saved["files"].items():
            if sha256((target / filename).read_bytes()) != digest:
                raise ValueError(f"Normalization subset cache is corrupted: {filename}")
        return
    sources = {}
    for filename in ["config.json", "model.safetensors.index.json"]:
        sources[filename] = fetch_metadata(config, filename)
    metadata = json.loads(sources["config.json"])
    for key in config["text_config_path"]:
        metadata = metadata[key]
    depth, width = metadata["num_hidden_layers"], metadata["hidden_size"]
    index = json.loads(sources["model.safetensors.index.json"])["weight_map"]
    shards = defaultdict(list)
    for layer in range(depth):
        for family in families:
            name = f"{config['prefix']}.{layer}.{family}.weight"
            shards[index[name]].append(name)
    tensors, receipts = {}, []
    byte_count = 0
    dtypes = {
        "BF16": (torch.bfloat16, 2),
        "F16": (torch.float16, 2),
        "F32": (torch.float32, 4),
    }
    with requests.Session() as session:
        for shard, names in shards.items():
            url = hf_hub_url(config["model"], shard, revision=config["revision"])
            prefix = read_range(session, url, 0, 7)
            size = struct.unpack("<Q", prefix)[0]
            if not 0 < size <= config["max_remote_header_bytes"]:
                raise ValueError("Safetensors header exceeds the configured limit")
            header_bytes = read_range(session, url, 8, size + 7)
            header = json.loads(header_bytes)
            byte_count += 8 + size
            intervals = []
            for name in names:
                entry = header[name]
                if entry["shape"] != [width] or entry["dtype"] not in dtypes:
                    raise ValueError(f"Invalid remote decoder norm: {name}")
                start, end = entry["data_offsets"]
                if end - start != width * dtypes[entry["dtype"]][1] or start < 0:
                    raise ValueError(f"Invalid remote tensor offsets: {name}")
                intervals.append((start, end, name))
            groups = []
            for start, end, name in sorted(intervals):
                if groups and groups[-1][1] == start:
                    groups[-1][1] = end
                    groups[-1][2].append((start, end, name))
                else:
                    groups.append([start, end, [(start, end, name)]])
            for start, end, entries in groups:
                data = read_range(session, url, size + 8 + start, size + 7 + end)
                byte_count += len(data)
                for lo, hi, name in entries:
                    raw = data[lo - start : hi - start]
                    dtype = dtypes[header[name]["dtype"]][0]
                    tensors[name] = torch.frombuffer(
                        bytearray(raw), dtype=dtype
                    ).clone()
                    receipts.append(
                        {
                            "tensor": name,
                            "shard": shard,
                            "byte_range": [size + 8 + lo, size + 7 + hi],
                            "sha256": sha256(raw),
                        }
                    )
            print(
                f"Fetched {len(names)} norm vectors from {shard}; "
                f"{byte_count:,} header/tensor bytes",
                flush=True,
            )
    target.mkdir(parents=True, exist_ok=True)
    tensor_file = "decoder-norms.safetensors"
    temporary = target / "decoder-norms.tmp.safetensors"
    save_file(
        tensors,
        temporary,
        metadata={"model": config["model"], "revision": config["revision"]},
    )
    temporary.replace(target / tensor_file)
    files = {
        "config.json": sources["config.json"],
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {name: tensor_file for name in tensors}}
        ).encode(),
    }
    for filename, data in files.items():
        temporary = target / (filename + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target / filename)
    receipt = {
        "model": config["model"],
        "revision": config["revision"],
        "tensor_count": len(tensors),
        "header_tensor_bytes": byte_count,
        "source_metadata_sha256": {n: sha256(b) for n, b in sources.items()},
        "tensors": receipts,
        "files": {n: sha256((target / n).read_bytes()) for n in [tensor_file, *files]},
    }
    temporary = marker.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(marker)
