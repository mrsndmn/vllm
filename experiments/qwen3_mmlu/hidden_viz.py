"""Hidden-state token stats of three chat_baseline responses as an HTML heatmap.

Local analysis outside the run.py registry. Teacher-forces the reference
responses of three generative-check questions through the HF model (one forward
pass each), keeps every token's 49 hidden states (the embedding output and the
residual stream after each decoder layer, all before the final norm), reduces
each to nine token stats, and writes one HTML file with the data embedded
(Plotly loads from a CDN).
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

import run

REFERENCE = "chat_baseline"
# First question of each of the first three subjects in the seeded check.
SAMPLES = (("abstract_algebra", 11), ("anatomy", 14), ("astronomy", 6))
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
STATS = ("min", "max", "mean", "median", "q10", "q25", "q50", "q75", "q90")
DEFAULT_STATES = "0,8,16,24,32,40,48"
# <|im_start|> and "user" open every prompt with massive activations (|x| ~ 1e3);
# the viewer leaves them out of the colour range by default.
DEFAULT_SKIP = 2
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


def capture(model, ids):
    """(49 hidden states [49, T, H], next-token argmax [T]) of one sequence.

    HF's output_hidden_states holds the embedding output and the outputs of all
    but the last decoder layer, then the final-norm output; the last layer's
    pre-norm output is the final norm's input, taken with a hook.
    """
    final = []
    handle = model.model.norm.register_forward_pre_hook(
        lambda module, args: final.append(args[0])
    )
    try:
        with torch.no_grad():
            output = model(
                torch.tensor([ids], device=model.device), output_hidden_states=True
            )
    finally:
        handle.remove()
    hidden = output.hidden_states
    if len(final) != 1 or len(hidden) != model.config.num_hidden_layers + 1:
        raise RuntimeError("Unexpected hidden-state layout")
    with torch.no_grad():
        if not torch.equal(model.model.norm(final[0]), hidden[-1]):
            raise RuntimeError("Last hidden state is not the final norm's input")
    states = torch.cat([*hidden[:-1], final[0]])
    return states, output.logits[0].argmax(dim=-1)


def token_text(tokenizer, token):
    """Readable text of one token; byte fragments fall back to the BPE symbol."""
    text = tokenizer.decode([token])
    return tokenizer.convert_ids_to_tokens(token) if "�" in text else text


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


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
        states, argmax = capture(model, ids)
        stats = torch.stack([token_stats(state) for state in states])
        if not stats.isfinite().all():
            raise RuntimeError("Non-finite token stats")
        # Position t predicts token t + 1; count model-written response tokens.
        predicted = argmax[len(prompt) - 1 : len(ids) - 1].tolist()
        hits = [p == t for p, t, s in zip(predicted, response, scored) if s]
        key = f"{record['subject']}_{record['index']}"
        arrays[f"{key}_stats"] = stats.cpu().numpy()
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
                "states": stats.shape[0],
                "stats": encode(stats.cpu().numpy()),
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
        "default_states": DEFAULT_STATES,
        "default_skip": DEFAULT_SKIP,
    }
    np.savez_compressed(output / "stats.npz", stat_names=np.array(STATS), **arrays)
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
