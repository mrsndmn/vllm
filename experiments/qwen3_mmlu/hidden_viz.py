"""Activation token stats of three chat_baseline responses as an HTML heatmap.

Local analysis outside the run.py registry. Teacher-forces the reference
responses of three generative-check questions through the HF model (one forward
pass each) and keeps, for every token, the activations of five sites (SITES): the
49 hidden states (embedding output and residual stream after each decoder layer,
before the final norm) and, per layer, the residual r, the MoE output, the expert
outputs, and the experts' SiLU-and-multiply outputs. Each token row is reduced to
nine token stats, each site row to a histogram of all its values, and everything
is written to one HTML file with the data embedded (Plotly loads from a CDN).
"""

import argparse
import base64
import datetime
import json
import math
import subprocess
from pathlib import Path

import chat_eval
import numpy as np
import torch
import torch.nn.functional as F

import run

REFERENCE = "chat_baseline"
# First question of each of the first three subjects in the seeded check.
SAMPLES = (("abstract_algebra", 11), ("anatomy", 14), ("astronomy", 6))
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
STATS = ("min", "max", "mean", "median", "q10", "q25", "q50", "q75", "q90")
# Viewer sites and their descriptions. `block` has the 49 hidden states; the others
# one row per decoder layer, shown as state s = layer + 1 so rows line up.
SITES = {
    "block": "hidden state: residual stream after the layer (r + MoE output); "
    "state 0 = embedding",
    "residual": "residual r: stream after the layer's attention, before the MoE add",
    "moe_out": "MoE output: router-weighted sum of the selected experts' outputs",
    "experts": "expert outputs: each selected expert's down_proj output, before its "
    "routing weight (8 × 2048 per token)",
    "mlp": "mlp site: each selected expert's SiLU(gate)·up, the input of down_proj "
    "(8 × 768 per token)",
}
DEFAULT_STATES = "0,8,16,24,32,40,48"
# <|im_start|> and "user" open every prompt with massive activations (|x| ~ 1e3);
# the viewer leaves them out of the colour range by default.
DEFAULT_SKIP = 2
# |x| bin edges of the per-state value distributions: 10 per decade, 1e-5 to 1e4.
HIST_EDGES = 10.0 ** (np.arange(91) / 10 - 5)
# Linear distributions: LINEAR_BINS bins over ± each site row's LINEAR_REACH
# quantile of |x| (the few larger values, mostly massive activations, are left out).
LINEAR_BINS = 201
LINEAR_REACH = 0.995
PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
TEMPLATE = Path(__file__).with_name("hidden_viz.html")


def token_stats(rows):
    """Nine token stats per row: signed min, max, mean, median, then the
    q10-q90 magnitude quantiles, each the row's floor(q * n)-th smallest |x| (the
    quantile-zeroing cutoff; at least the smallest). Computed in FP32."""
    values = rows.float()
    width = values.shape[1]
    ordered = values.sort(dim=1).values
    median = (ordered[:, (width - 1) // 2] + ordered[:, width // 2]) / 2
    magnitude = values.abs().sort(dim=1).values
    ranks = [max(math.floor(q * width), 1) - 1 for q in QUANTILES]
    return torch.stack(
        [ordered[:, 0], ordered[:, -1], values.mean(dim=1), median]
        + [magnitude[:, rank] for rank in ranks],
        dim=1,
    )


def value_histogram(rows):
    """Counts [2, len(HIST_EDGES)] of all values of one site row (every token):
    row 0 negative and row 1 non-negative values, by |x| bin. Bin 0 holds
    |x| < HIST_EDGES[0] (zeros included), bin i HIST_EDGES[i-1] <= |x| <
    HIST_EDGES[i], and the last bin also everything above."""
    values = rows.float().flatten()
    edges = torch.tensor(HIST_EDGES, dtype=torch.float32, device=values.device)
    bins = torch.bucketize(values.abs(), edges, right=True)
    bins = bins.clamp(max=len(HIST_EDGES) - 1)
    index = (values >= 0).long() * len(HIST_EDGES) + bins
    return torch.bincount(index, minlength=2 * len(HIST_EDGES)).view(2, -1)


def linear_histogram(rows):
    """(counts [LINEAR_BINS], reach) of one site row's signed values on a linear
    axis over [-reach, reach], reach being the row's LINEAR_REACH quantile of |x|
    (its ceil(q * n)-th smallest); values beyond it are not counted."""
    values = rows.float().flatten()
    rank = max(math.ceil(LINEAR_REACH * values.numel()), 1)
    reach = values.abs().kthvalue(rank).values.item() or 1e-30
    return torch.histc(values, LINEAR_BINS, -reach, reach).long(), reach


def pick(records):
    """The SAMPLES records of a reference run, in SAMPLES order."""
    by_key = {(r["subject"], r["index"]): r for r in records}
    missing = [key for key in SAMPLES if key not in by_key]
    if missing:
        raise ValueError(f"{REFERENCE} has no records for {missing}")
    return [by_key[key] for key in SAMPLES]


def sequence(record, tokenizer):
    """(prompt ids, response ids, scored mask) of one reference response.

    The model-written first segment must re-tokenize to the recorded completion
    token count (vLLM counts the end-of-turn token of a natural stop).
    """
    prompt = tokenizer.encode(record["prompt"], add_special_tokens=False)
    segments = chat_eval.reference_segments(record, tokenizer)
    if len(segments[0][0]) != record["reasoning_tokens"]:
        raise ValueError(
            f"{record['subject']} #{record['index']}: response re-tokenizes to "
            f"{len(segments[0][0])} tokens, recorded {record['reasoning_tokens']}"
        )
    response = [token for ids, _ in segments for token in ids]
    scored = [flag for ids, flag in segments for _ in ids]
    return prompt, response, scored


def expert_rows(moe, hidden):
    """(mlp rows [T, k * I], expert-output rows [T, k * H], MoE output [T, H]) of
    one sparse-MoE block, recomputed exactly as HF's forward: the selected experts'
    SiLU(gate) * up (down_proj's input) and down_proj outputs in top-k order, and
    their routing-weighted sum."""
    x = hidden.reshape(-1, hidden.shape[-1])
    weights = F.softmax(moe.gate(x), dim=1, dtype=torch.float)
    weights, selected = torch.topk(weights, moe.top_k, dim=-1)
    if moe.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)
    mask = F.one_hot(selected, num_classes=moe.num_experts).permute(2, 1, 0)
    width = moe.experts[0].down_proj.in_features
    inner = x.new_zeros(x.shape[0], moe.top_k, width)
    outer = x.new_zeros(x.shape[0], moe.top_k, x.shape[1])
    mixed = torch.zeros_like(x)
    for index in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
        expert = moe.experts[index]
        slot, token = torch.where(mask[index])
        state = x[token]
        act = expert.act_fn(expert.gate_proj(state)) * expert.up_proj(state)
        out = expert.down_proj(act)
        inner[token, slot] = act
        outer[token, slot] = out
        mixed.index_add_(0, token, (out * weights[token, slot, None]).to(x.dtype))
    return inner.flatten(1), outer.flatten(1), mixed


def capture(model, ids):
    """(activations by site, next-token argmax [T]) of one teacher-forced sequence.

    Each site maps to [rows, T, width]. `block` holds the 49 hidden states: HF's
    output_hidden_states has the embedding output and the outputs of all but the
    last decoder layer, then the final-norm output, so the last layer's pre-norm
    output is taken as the final norm's input. The other sites have one row per
    layer: `residual` is the post-attention norm's input, `moe_out` the sparse-MoE
    block's output, and `experts`/`mlp` come from expert_rows, whose weighted sum
    must reproduce `moe_out` exactly.
    """
    final, layers = [], {site: [] for site in SITES if site != "block"}

    def moe_hook(module, args, output):
        inner, outer, mixed = expert_rows(module, args[0])
        moe_out = output[0].reshape(mixed.shape)
        if not torch.equal(mixed, moe_out):
            difference = (mixed - moe_out).abs().max().item()
            raise RuntimeError(
                f"Expert outputs do not sum to the MoE output ({difference})"
            )
        layers["moe_out"].append(moe_out)
        layers["experts"].append(outer)
        layers["mlp"].append(inner)

    handles = [
        model.model.norm.register_forward_pre_hook(
            lambda module, args: final.append(args[0])
        )
    ]
    for layer in model.model.layers:
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda module, args: layers["residual"].append(args[0][0])
            )
        )
        handles.append(layer.mlp.register_forward_hook(moe_hook))
    try:
        with torch.no_grad():
            output = model(
                torch.tensor([ids], device=model.device), output_hidden_states=True
            )
    finally:
        for handle in handles:
            handle.remove()
    hidden = output.hidden_states
    count = model.config.num_hidden_layers
    if len(final) != 1 or len(hidden) != count + 1:
        raise RuntimeError("Unexpected hidden-state layout")
    if any(len(rows) != count for rows in layers.values()):
        raise RuntimeError("Not every layer recorded every site")
    with torch.no_grad():
        if not torch.equal(model.model.norm(final[0]), hidden[-1]):
            raise RuntimeError("Last hidden state is not the final norm's input")
    sites = {"block": torch.cat([*hidden[:-1], final[0]])}
    sites.update({site: torch.stack(rows) for site, rows in layers.items()})
    return sites, output.logits[0].argmax(dim=-1)


def token_text(tokenizer, token):
    """Readable text of one token; byte fragments fall back to the BPE symbol."""
    text = tokenizer.decode([token])
    return tokenizer.convert_ids_to_tokens(token) if "�" in text else text


def encode(array, dtype="<f4"):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def render(samples, meta):
    """Standalone HTML: the template with the data JSON embedded.

    Every "<" is escaped so no text (a closing tag, "<!--") can end the script."""
    payload = json.dumps({**meta, "samples": samples}).replace("<", "\\u003c")
    page = TEMPLATE.read_text()
    return page.replace("%%PLOTLY%%", PLOTLY).replace("%%DATA%%", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, help="default <artifacts>/hidden-viz/<time>"
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = run.CONFIG
    output = args.output or run.ARTIFACTS / "hidden-viz" / time_stamp()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["snapshot"])
    records = pick(chat_eval.reference_records(REFERENCE))
    model = AutoModelForCausalLM.from_pretrained(
        config["snapshot"], dtype=torch.bfloat16, device_map=args.device
    ).eval()

    samples, arrays = [], {}
    for record in records:
        prompt, response, scored = sequence(record, tokenizer)
        ids = prompt + response
        activations, argmax = capture(model, ids)
        key = f"{record['subject']}_{record['index']}"
        sites = {}
        for site, rows in activations.items():
            stats = torch.stack([token_stats(row) for row in rows]).cpu()
            if not stats.isfinite().all():
                raise RuntimeError(f"Non-finite {site} token stats")
            hist = torch.stack([value_histogram(row) for row in rows]).cpu()
            if hist.sum() != rows.numel():
                raise RuntimeError(f"{site} histograms do not cover every value")
            linear = [linear_histogram(row) for row in rows]
            counts = torch.stack([row for row, _ in linear]).cpu()
            reach = [value for _, value in linear]
            arrays[f"{key}_{site}_stats"] = stats.numpy()
            arrays[f"{key}_{site}_hist"] = hist.numpy()
            arrays[f"{key}_{site}_linear"] = counts.numpy()
            arrays[f"{key}_{site}_reach"] = np.array(reach)
            sites[site] = {
                "rows": rows.shape[0],
                "width": rows.shape[2],
                # State of the first row: 0 (embedding) for block, 1 (layer 0) else.
                "first_state": model.config.num_hidden_layers + 1 - rows.shape[0],
                "stats": encode(stats.numpy()),
                "hist": encode(hist.numpy(), "<i4"),
                "linear": encode(counts.numpy(), "<i4"),
                "reach": reach,
            }
        del activations
        # Position t predicts token t + 1; count model-written response tokens.
        predicted = argmax[len(prompt) - 1 : len(ids) - 1].tolist()
        hits = [p == t for p, t, s in zip(predicted, response, scored) if s]
        arrays[f"{key}_ids"] = np.array(ids)
        samples.append(
            {
                **{
                    field: record[field]
                    for field in (
                        "subject",
                        "index",
                        "question",
                        "choices",
                        "answer",
                        "prediction",
                        "correct",
                        "mode",
                    )
                },
                "tokens": [token_text(tokenizer, token) for token in ids],
                "prompt_tokens": len(prompt),
                "argmax_agreement": sum(hits) / len(hits),
                "sites": sites,
            }
        )
        print(
            f"{key}: {len(prompt)} prompt + {len(response)} response tokens, "
            f"argmax agreement {sum(hits)}/{len(hits)}",
            flush=True,
        )

    marker = run.job_root(REFERENCE, None) / "done.json"
    meta = {
        "model": config["model"],
        "reference": json.loads(marker.read_text())["output"],
        "commit": commit(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "stat_names": STATS,
        "sites": [{"name": k, "description": v} for k, v in SITES.items()],
        "states": model.config.num_hidden_layers + 1,
        "default_states": DEFAULT_STATES,
        "default_skip": DEFAULT_SKIP,
        "hist_edges": HIST_EDGES.tolist(),
        "linear_bins": LINEAR_BINS,
        "linear_reach": LINEAR_REACH,
    }
    np.savez_compressed(
        output / "stats.npz",
        stat_names=np.array(STATS),
        hist_edges=HIST_EDGES,
        **arrays,
    )
    page = output / "hidden-states.html"
    page.write_text(render(samples, meta))
    print(f"{page} ({page.stat().st_size / 2**20:.1f} MiB)")


def time_stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def commit():
    """HEAD, marked -dirty if the experiment directory has uncommitted changes."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True
        ).stdout.strip()

    return git("rev-parse", "HEAD") + (
        "-dirty" if git("status", "--porcelain", ".") else ""
    )


if __name__ == "__main__":
    main()
