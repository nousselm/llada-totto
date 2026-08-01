#!/usr/bin/env python3
"""Prepare train/eval subsets from ToTTo."""

import json

# Read the first 3 lines
with open("totto_data/totto_train_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        exemple = json.loads(line)  # each line -> Python dict
        print(f"\n=== Example {i} ===")
        print(exemple)
        if i >= 2:
            break

import json
with open("totto_data/totto_train_data.jsonl", "r") as f:
    exemple = json.loads(f.readline())  # first line only
    print("Keys :", list(exemple.keys()))

import json

def serialize_table(table, highlighted_cells):
    """
    Serializes highlighted cells into linear text.
    Converts a ToTTo table into structured linear text.
    Focuses on highlighted_cells, which are relevant for the description.

    Format: "header1 : value1 | header2 : value2 | ..."
    """
    # Extract headers from row 0
    headers = [cell["value"].strip() for cell in table[0]]

    parts = []
    for (row_idx, col_idx) in highlighted_cells:
        if row_idx >= len(table) or col_idx >= len(table[row_idx]):
            continue
        value = table[row_idx][col_idx]["value"].strip()
        header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
        if value:
            parts.append(f"{header} : {value}")

    return " | ".join(parts) if parts else "no data"


# Test on the first 3 examples
with open("totto_data/totto_train_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        serialized = serialize_table(ex["table"], ex["highlighted_cells"])
        reference = ex["sentence_annotations"][0]["final_sentence"]
        print(f"\n=== Example {i} ===")
        print(f"TABLE   : {serialized}")
        print(f"REF     : {reference}")
        if i >= 2:
            break

def load_jsonl(path):
    """Loads a jsonl file into a list of dictionaries"""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Load datasets
train_data = load_jsonl("totto_data/totto_train_data.jsonl")
validation_data = load_jsonl("totto_data/totto_dev_data.jsonl")  # Note: dev = validation

# Dictionary of splits
dataset = {
    "train": train_data,
    "validation": validation_data
}

import random
# Building subsets
def build_subset(split, n):
    """Extracts n valid examples (with at least one highlighted_cell)."""
    subset = []

    for ex in dataset[split]:
        if len(ex["highlighted_cells"]) == 0:
            continue
        serialized = serialize_table(ex["table"], ex["highlighted_cells"])
        reference = ex["sentence_annotations"][0]["final_sentence"]
        if not serialized or not reference:
            continue
        subset.append({
            "table_id": ex.get("table_page_title", ""),
            "serialized_table": serialized,
            "reference": reference,
        })

        if len(subset) >= n:
            break

    return subset

# Build the two subsets
random.seed(42)
train_subset = build_subset("train", 5000)
eval_subset  = build_subset("validation", 500)
print(f"Train subset : {len(train_subset)} examples")
print(f"Eval subset  : {len(eval_subset)} examples")

# Verification
print("\n--- Example eval[0] ---")
print("Table    :", eval_subset[0]["serialized_table"][:120], "...")
print("Ref      :", eval_subset[0]["reference"])

random.shuffle(dataset["train"])
random.shuffle(dataset["validation"])


from pathlib import Path
import os

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "train_subset.json", "w", encoding="utf-8") as f:
    json.dump(train_subset, f, ensure_ascii=False, indent=2)

with open(output_dir / "eval_subset.json", "w", encoding="utf-8") as f:
    json.dump(eval_subset, f, ensure_ascii=False, indent=2)

print(f"Saved subsets to {output_dir.resolve()}")
