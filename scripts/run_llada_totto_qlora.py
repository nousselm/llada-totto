#!/usr/bin/env python3
"""
run_llada_totto_pretrained.py

Reference evaluation pipeline for the QLoRA fine-tuned LLaDA-8B-Instruct model.

This script intentionally mirrors the QLoRA evaluation pipeline so that
the Loads the pretrained base model together with a QLoRA adapter.
Outputs are written under outputs/qlora/.
"""

import os
import json
import random
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
import evaluate
import sys
# ============================================================
# CONFIG (EDIT HERE)
# ============================================================

LLADA_REPO = r"C:\Users\nouss\Desktop\gen IA\Masked-Diffusion-Language-Models-for-Structured-to-Text-Generation-Tabular-Data\LLaDA"
if LLADA_REPO not in sys.path:
    sys.path.append(LLADA_REPO)
            # <-- PATH TO OFFICIAL REPO
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
LORA_PATH = "./outputs/llada_totto_qlora/final"

OUTPUT_DIR = "./outputs/qlora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
USE_FIXED_SEED = True

SHOT_SETS = [0, 1, 3, 5, 10]

GEN_LENGTH = 64
GEN_STEPS = 64
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"

DEBUG = True

# DATAPATH = "/data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LOAD DATA
# # ============================================================
from pathlib import Path

data_folder = Path('data')

eval_path = data_folder / 'eval_subset.json'
train_path = data_folder / 'train_subset.json'

with open(eval_path, "r", encoding="utf-8") as f:
    eval_subset = json.load(f)

with open(train_path, "r", encoding="utf-8") as f:
    train_subset = json.load(f)

if USE_FIXED_SEED:
    random.Random(SEED).shuffle(train_subset)

# ============================================================
# SAMPLES
# ============================================================
if DEBUG:
    N_SAMPLES = 5
else:
    N_SAMPLES = len(eval_subset)

# ============================================================
# EMBEDDINGS
# ============================================================


def format_table(example):
    """
    Convertit serialized_table en input textuel propre
    (important pour LLaDA + BLEU stable)
    """
    return f"Table: {example['serialized_table']}\n"

train_texts = [format_table(x) for x in train_subset]
print("Chargement du modèle de retrieval...")

retrieval_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

retrieval_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
).to(DEVICE)

retrieval_model.eval()

print("Retrieval chargé.")
@torch.no_grad()
def encode_texts(texts, batch_size=32):
    """
    Encode une liste de textes avec all-MiniLM-L6-v2
    sans utiliser sentence-transformers.
    """

    embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        encoded = retrieval_tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        outputs = retrieval_model(**encoded, return_dict=True)

        token_embeddings = outputs.last_hidden_state

        attention_mask = encoded["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sentence_embeddings = (
            token_embeddings * mask
        ).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        embeddings.append(sentence_embeddings)

    return F.normalize(
        torch.cat(embeddings, dim=0),
        p=2,
        dim=1
    )

train_embeddings = encode_texts(train_texts)   


print("Train embeddings shape:", train_embeddings.shape)


def retrieve(query, top_k=10):

    q = encode_texts([query])[0]

    sims = torch.matmul(train_embeddings, q)

    idx = torch.argsort(sims, descending=True)

    return idx[:top_k].cpu().numpy()

# ============================================================
# NESTED SHOTS
# ============================================================
print("Building nested shot retrieval...")

nested = {}

for i, ex in enumerate(eval_subset):
    q = format_table(ex)
    top = retrieve(q, 10)
    nested[i] = {
        0: [],
        1: top[:1].tolist(),
        3: top[:3].tolist(),
        5: top[:5].tolist(),
        10: top[:10].tolist(),
    }

print("Nested retrieval terminé.")

print("Libération du modèle de retrieval...")

del retrieval_model
del retrieval_tokenizer

import gc
gc.collect()

torch.cuda.empty_cache()

print("Mémoire GPU libérée.")
# ============================================================
# LOAD LLaDA
# ============================================================
from generate import generate



print("Chargement du tokenizer LLaDA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)   

print("Chargement de LLaDA en 4 bits...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=(
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    ),
)

base_model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

print("Chargement de l'adapter QLoRA...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()

print("Adapter QLoRA chargé.")


if torch.cuda.is_available():
    print(
        f"VRAM utilisée : {torch.cuda.memory_allocated()/1024**3:.2f} Go"
    )

    print(
        f"VRAM réservée : {torch.cuda.memory_reserved()/1024**3:.2f} Go"
    )

MASK_ID = 126336

if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"

assert tokenizer.pad_token_id != MASK_ID

# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert at generating factual natural language "
    "descriptions from structured tables."
)

def build_prompt(example, retrieved_examples=None):

    prompt = SYSTEM_PROMPT + "\n\n"

    if retrieved_examples:

        prompt += "Examples:\n\n"

        for i, ex in enumerate(retrieved_examples, 1):

            prompt += f"Example {i}\n"
            prompt += "Table:\n"
            prompt += ex["serialized_table"]
            prompt += "\n"

            prompt += "Description:\n"
            prompt += ex["reference"]
            prompt += "\n\n"

    prompt += "Generate one faithful description for the following table."
    prompt += "\n\n"

    prompt += "Table:\n"
    prompt += example["serialized_table"]
    prompt += "\n\nDescription:"

    return prompt


def tokenize(prompt):
    msg = [{"role": "user", "content": prompt}]
    txt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

    enc = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_llada(prompt):
    input_ids, attn = tokenize(prompt)

    out = generate(
        model=model,
        prompt=input_ids,
        attention_mask=attn,
        steps=GEN_STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=TEMPERATURE,
        cfg_scale=CFG_SCALE,
        remasking=REMASKING,
        mask_id=MASK_ID,
    )

    return tokenizer.batch_decode(
        out[:, input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

def predict(i, shot):

    ex = eval_subset[i]
    retrieved = []

    if shot > 0:
        idxs = nested[i][shot]
        retrieved = [train_subset[j] for j in idxs]

    prompt = build_prompt(ex, retrieved)

    if i == 0:
        print("=" * 80)
        print(prompt[:1000])
        print("=" * 80)

    prediction = generate_llada(prompt)

    if i == 0:
        print(prediction)

    return prediction
# ============================================================
# SAVE RETRIEVAL IDS
# ============================================================
with open(
    os.path.join(OUTPUT_DIR, "retrieval_ids.json"),
    "w"
) as f:
    json.dump(nested, f, indent=2)

# ============================================================
# RUN GENERATION
# ============================================================

predictions = {s: [] for s in SHOT_SETS}

for s in SHOT_SETS:

    print(f"\nGenerating {s}-shot...")

    for i in tqdm(range(N_SAMPLES)):
        predictions[s].append(predict(i, s))

with open(
    os.path.join(OUTPUT_DIR, "predictions.json"),
    "w"
) as f:
    json.dump(predictions, f, indent=2)

# ============================================================
# SELF-BLEU
# ============================================================

from nltk.translate.bleu_score import sentence_bleu

def compute_self_bleu(predictions):

    scores = []

    if len(predictions) < 2:
        return 0.0

    for i, pred in enumerate(predictions):

        refs = predictions[:i] + predictions[i+1:]
        refs = [r.split() for r in refs]

        score = sentence_bleu(
            refs,
            pred.split()
        )

        scores.append(score)

    return float(np.mean(scores))

# ============================================================
# METRICS
# ============================================================

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

refs = [x["reference"] for x in eval_subset[:N_SAMPLES]]

results = {}

for s in SHOT_SETS:
    preds = predictions[s]

    b = bleu.compute(predictions=preds, references=[[r] for r in refs])
    r = rouge.compute(predictions=preds, references=refs)
    br = bertscore.compute(predictions=preds, references=refs, lang="en")

    results[s] = {
        "BLEU": b["score"],
        "ROUGE1": r["rouge1"],
        "ROUGE2": r["rouge2"],
        "ROUGEL": r["rougeL"],
        "BERTScore": float(np.mean(br["f1"])),
        "SelfBLEU": compute_self_bleu(preds)
        
    }

# ============================================================
# SAVE RESULTS
# ============================================================

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(results)

# ============================================================
# EXPORT TO CSV METRICS
# ============================================================
import pandas as pd

table = pd.DataFrame(results).T

table.index = [
    f"{k}-shot"
    for k in SHOT_SETS
]

csv_path = os.path.join(
    OUTPUT_DIR,
    "results_llada.csv"
)

table.to_csv(csv_path)

print(table)
print(csv_path)

# ============================================================
# EXPORT TO CSV PREDICTIONS
# ============================================================

rows = []

for shot in SHOT_SETS:

    for i in range(N_SAMPLES):

        rows.append({
            "shot": shot,
            "table": eval_subset[i]["serialized_table"],
            "reference": refs[i],
            "prediction": predictions[shot][i]
        })

analysis = pd.DataFrame(rows)

analysis_path = os.path.join(
    OUTPUT_DIR,
    "generation_analysis.csv"
)

analysis.to_csv(
    analysis_path,
    index=False
)

print(analysis_path)