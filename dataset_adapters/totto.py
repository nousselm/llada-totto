"""
ToTTo adapter.

Contains only dataset-specific logic.
"""

import json
import random


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def serialize_table(table, highlighted_cells):
    headers = [c["value"] for c in table[0]]
    parts = []

    for row, col in highlighted_cells:
        value = table[row][col]["value"]
        header = headers[col]
        parts.append(f"{header}: {value}")

    return " | ".join(parts)


def convert_example(example):
    reference = example["sentence_annotations"][0]["final_sentence"]

    return {
        "table_id": example.get("table_page_title", ""),
        "serialized_table": serialize_table(
            example["table"],
            example["highlighted_cells"],
        ),
        "reference": reference,
    }


def build_subset(examples, size, seed=42):
    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)

    subset = []
    for ex in examples:
        try:
            subset.append(convert_example(ex))
        except Exception:
            continue

        if len(subset) >= size:
            break

    return subset
