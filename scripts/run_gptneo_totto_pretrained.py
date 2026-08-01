#!/usr/bin/env python3
"""Converted from gptneo_notebook2.ipynb."""

import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("rouge_score")
pip("nltk")
pip("transformers")
pip("accelerate")

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


import json, random, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    pipeline,
)
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rs_module


with open(f"{FOLDER}eval_subset.json")  as f: eval_data  = json.load(f)
with open(f"{FOLDER}eval_subset_100.json")  as f: eval_data_100  = json.load(f)
with open(f"{FOLDER}train_subset.json") as f: train_data = json.load(f)

print(f"Train : {len(train_data)} | Eval : {len(eval_data)} | Eval_100 : {len(eval_data_100)}")


class TableDatasetKShot(Dataset):
    """
    Builds a few-shot prompt with k examples drawn from `pool`.
    k=0 → zero-shot.
    """
    def __init__(self, data, pool, tokenizer, max_length, k=0, seed=42):
        self.data       = data
        self.pool       = pool
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.k          = k
        self.seed       = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        table = item["serialized_table"]
        label = item["reference"]

        rng = random.Random(self.seed + idx)
        examples = rng.sample(self.pool, k=self.k) if self.k > 0 else []
        prompt   = self._build_prompt(table, examples)

        enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        lbl = self.tokenizer(
            label, truncation=True, max_length=128,
            padding="max_length", return_tensors="pt",
        )
        return {
            "table":          table,
            "prompt":         prompt,
            "labels_str":     label,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         lbl["input_ids"].squeeze(0),
        }

    def _build_prompt(self, table, examples):
        prompt = ""
        for ex in examples:
            prompt += f"Table: {ex['serialized_table']}\nDescription: {ex['reference']}\n\n"
        prompt += f"Table: {table}\nDescription:"
        return prompt


def generate_batch(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    dec_out    = tokenizer.batch_decode(outputs,   skip_special_tokens=True)
    dec_prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    cleaned = []
    for p, full in zip(dec_prompt, dec_out):
        tail = full[len(p):].strip() if full.startswith(p) else full.strip()
        cleaned.append(tail)
    return cleaned


_rouge  = rs_module.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1

def _bleu_rouge(pred, ref):
    try:
        bleu = sentence_bleu([word_tokenize(ref)], word_tokenize(pred),
                             weights=(0.25,0.25,0.25,0.25),
                             smoothing_function=_smooth)
    except ZeroDivisionError:
        bleu = 0.0
    sc = _rouge.score(ref, pred)
    return bleu, sc["rouge1"].fmeasure, sc["rouge2"].fmeasure, sc["rougeL"].fmeasure

def bleu_rouge_batch(preds, refs):
    vals = [_bleu_rouge(p, r) for p, r in zip(preds, refs)]
    b,r1,r2,rl = zip(*vals)
    return np.mean(b), np.mean(r1), np.mean(r2), np.mean(rl)


print("Loading BERTScore backbone (bert-base-uncased) …")
_bt   = AutoTokenizer.from_pretrained("bert-base-uncased")
_bm   = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()

def _embed(texts):
    enc = _bt(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        return _bm(**enc).last_hidden_state

def bertscore_batch(preds, refs):
    ep, er = _embed(preds), _embed(refs)
    scores = []
    for i in range(len(preds)):
        p = F.normalize(ep[i], dim=1)
        r = F.normalize(er[i], dim=1)
        sim  = torch.mm(p, r.T)
        prec = sim.max(dim=1).values.mean()
        rec  = sim.max(dim=0).values.mean()
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        scores.append(f1.item())
    return float(np.mean(scores))


print("Loading NLI model (cross-encoder/nli-MiniLM2-L6-H768) …")
_nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768",
    device=0 if DEVICE == "cuda" else -1,
    top_k=None,
)

def nli_faithfulness_batch(tables, preds, max_chars=400):
    pairs = [f"{t[:max_chars]} [SEP] {p[:max_chars]}"
             for t, p in zip(tables, preds)]
    scores = []
    for result in _nli(pairs, truncation=True, max_length=512):
        ls = {r["label"]: r["score"] for r in result}
        scores.append(ls.get("ENTAILMENT", 0.0))
    return float(np.mean(scores))



MODEL_NAME = "EleutherAI/gpt-neo-1.3B"

print("\n" + "="*55)
print(f"  Loading {MODEL_NAME}")
print("="*55)

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token    = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.pad_token_id = tok.eos_token_id

total = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Loaded — {total:.0f}M parameters on {DEVICE}")

print("\nCalibrating loaders …")
loaders, mlens = make_loaders(
    eval_data_100, train_data, tok,
    k_values=K_VALUES, ctx_limit=2048, batch_size=BATCH
)

results_gptneo = evaluate_model(model, tok, loaders, K_VALUES, max_new_tokens=64)


out_path = f"{FOLDER}results_gptneo_100.json"
with open(out_path, "w") as f:
    json.dump(results_gptneo, f, indent=2)
print(f"\nResults saved → {out_path}")


import pandas as pd

NB_QUAL = 50
model.eval()
for k in K_VALUES:
    rows, count = [], 0
    with torch.no_grad():
        for batch in loaders[k]:
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            preds = generate_batch(model, tok, ids, mask, 64)
            for table, ref, pred in zip(batch["table"],
                                        batch["labels_str"], preds):
                rows.append({"table": table, "reference": ref, "prediction": pred})
                count += 1
            if count >= NB_QUAL:
                break
    path = f"{FOLDER}gptneo_k{k}_qualitative_100.csv"
    pd.DataFrame(rows[:NB_QUAL]).to_csv(path, index=False)
    print(f"  Qualitative k={k} → {path}")


print("\n" + "="*55)
print("  GPT-NEO RESULTS SUMMARY")
print("="*55)
header = f"{'k':>4} | {'BLEU':>7} | {'ROUGE-L':>7} | {'BERTScore':>9} | {'NLI':>7}"
print(header)
print("─" * len(header))
for k in K_VALUES:
    r = results_gptneo[k]
    print(f"{k:>4} | {r['bleu']:>7.4f} | {r['rouge_l']:>7.4f} "
          f"| {r['bertscore']:>9.4f} | {r['nli']:>7.4f}")


del model
gc.collect()
torch.cuda.empty_cache()
print("\nGPT-Neo unloaded. GPU memory freed.")

