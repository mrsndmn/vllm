# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GSM8K data and numeric-answer protocol for the generative chat evaluator."""

import random
from decimal import Decimal

import regex as re

TASK = "gsm8k_chat"
PROMPT = (
    "Solve the following math word problem. Think step by step, but keep your "
    "reasoning short. The last line should be 'Answer: NUMBER', where NUMBER is "
    "only the final numeric answer, without units or an expression.\n\n{question}"
)
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def numeric_answer(text):
    """Normalize a complete numeric answer; reject expressions and ambiguity."""
    text = text.strip().strip("*_`\"'").strip()
    if text.startswith(r"\boxed{") and text.endswith("}"):
        text = text[7:-1].strip()
    text = text.strip("$").strip()
    if not NUMBER.fullmatch(text):
        return None
    value = Decimal(text.replace(",", ""))
    if value == 0:
        return "0"
    normalized = format(value, "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def final_answer(text):
    """Read the last explicitly marked answer, never a reasoning calculation."""
    lines = [
        line for line in text.splitlines() if re.search(r"(?i)answer\s*[:：]", line)
    ]
    if not lines:
        return None
    answer = re.split(r"(?i)answer\s*[:：]", lines[-1])[-1]
    return numeric_answer(answer)


def load_test_split(config):
    from datasets import load_dataset

    return load_dataset(
        config["chat_dataset"],
        config["chat_dataset_config"],
        split=config["chat_split"],
        revision=config["chat_dataset_revision"],
    )


def select_questions(config, limit=None, load=None):
    """Use the full test split, or a reproducible smoke subset."""
    rows = (load or load_test_split)(config)
    if len(rows) != config["chat_expected_questions"]:
        raise ValueError("Unexpected GSM8K test split size")
    indices = range(len(rows))
    if limit is not None:
        if not 0 < limit <= len(rows):
            raise ValueError("GSM8K limit must be between 1 and the test split size")
        indices = sorted(random.Random(config["chat_seed"]).sample(indices, limit))
    picked = []
    for index in indices:
        row = rows[index]
        _, separator, gold = row["answer"].rpartition("####")
        answer = numeric_answer(gold) if separator else None
        if answer is None:
            raise ValueError(f"Invalid GSM8K gold answer at index {index}")
        picked.append(
            {
                "subject": "gsm8k",
                "index": index,
                "question": row["question"].strip(),
                "answer": answer,
            }
        )
    return picked
