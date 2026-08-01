#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Évalue des générations LLaDA sauvegardées dans predictions.json.

Exemples :
    python evaluate_predictions.py --model qlora
    python evaluate_predictions.py --model pretrained
    python evaluate_predictions.py --input outputs/qlora/predictions.json
    python evaluate_predictions.py --model qlora --skip-nli
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import pipeline

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

PRED_KEYS = (
    "prediction", "pred", "generated_text", "generation", "output",
    "decoded", "text", "hypothesis"
)
REF_KEYS = (
    "reference", "ref", "target", "target_text", "gold",
    "gold_text", "answer", "sentence", "final_sentence"
)
SHOT_KEYS = ("shot", "shots", "n_shots", "num_shots", "k", "setting")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(normalize_text(x) for x in value if normalize_text(x))
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else 0.0


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def distinct_n(texts: Sequence[str], n: int) -> float:
    all_ngrams: List[Tuple[str, ...]] = []
    for text in texts:
        tokens = tokenize(text)
        all_ngrams.extend(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def repetition_rate(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(tokens)


def compute_self_bleu(predictions: Sequence[str], max_samples: int = 200) -> float:
    predictions = [normalize_text(x) for x in predictions if normalize_text(x)][:max_samples]
    if len(predictions) < 2:
        return 0.0

    smoothing = SmoothingFunction().method1
    tokenized = [tokenize(x) for x in predictions]
    scores: List[float] = []

    for i, hypothesis in enumerate(tokenized):
        references = [tokens for j, tokens in enumerate(tokenized) if j != i and tokens]
        if hypothesis and references:
            scores.append(sentence_bleu(
                references,
                hypothesis,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            ))
    return safe_mean(scores)


def first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def extract_reference_from_eval_item(item: Dict[str, Any]) -> str:
    direct = first_present(item, REF_KEYS)
    if direct is not None:
        return normalize_text(direct)

    for key in ("sentence_annotations", "annotations"):
        value = item.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            candidate = first_present(value[0], ("final_sentence", "sentence", "text"))
            if candidate is not None:
                return normalize_text(candidate)

    for key in ("target", "targets", "references"):
        value = item.get(key)
        if isinstance(value, list) and value:
            return normalize_text(value[0])
        if isinstance(value, dict):
            candidate = first_present(value, REF_KEYS)
            if candidate is not None:
                return normalize_text(candidate)
    return ""


def load_eval_references(eval_path: Optional[Path]) -> List[str]:
    if eval_path is None or not eval_path.exists():
        return []

    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "examples", "items", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        return []

    return [extract_reference_from_eval_item(x) if isinstance(x, dict) else "" for x in data]


def normalize_shot_label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = normalize_text(value)
    match = re.search(r"(\d+)", text)
    return f"{match.group(1)}-shot" if match else (text or "unknown")


def parse_record_list(records: Sequence[Any], fallback_refs: Sequence[str], default_shot: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for i, item in enumerate(records):
        if isinstance(item, str):
            rows.append({
                "shot": default_shot,
                "prediction": normalize_text(item),
                "reference": fallback_refs[i] if i < len(fallback_refs) else "",
            })
            continue

        if not isinstance(item, dict):
            continue

        pred = normalize_text(first_present(item, PRED_KEYS))
        ref = normalize_text(first_present(item, REF_KEYS))
        shot = normalize_shot_label(first_present(item, SHOT_KEYS) or default_shot)

        if not ref and i < len(fallback_refs):
            ref = fallback_refs[i]
        if pred:
            rows.append({"shot": shot, "prediction": pred, "reference": ref})
    return rows


def parse_predictions(raw: Any, fallback_refs: Sequence[str]) -> List[Dict[str, str]]:
    if isinstance(raw, list):
        return parse_record_list(raw, fallback_refs, "unknown")

    if not isinstance(raw, dict):
        raise ValueError("Le JSON doit contenir une liste ou un dictionnaire.")

    if isinstance(raw.get("predictions"), list):
        refs = raw.get("references")
        refs = [normalize_text(x) for x in refs] if isinstance(refs, list) else list(fallback_refs)
        shot = normalize_shot_label(first_present(raw, SHOT_KEYS))
        return parse_record_list(raw["predictions"], refs, shot)

    rows: List[Dict[str, str]] = []
    ignored = {"metadata", "config", "retrieval_ids", "references", "args", "timings", "generation_times"}

    for key, value in raw.items():
        if key in ignored:
            continue
        shot = normalize_shot_label(key)

        if isinstance(value, list):
            rows.extend(parse_record_list(value, fallback_refs, shot))
        elif isinstance(value, dict):
            nested = first_present(value, ("predictions", "generations", "outputs", "results", "samples"))
            nested_refs = value.get("references", fallback_refs)
            nested_refs = [normalize_text(x) for x in nested_refs] if isinstance(nested_refs, list) else list(fallback_refs)
            if isinstance(nested, list):
                rows.extend(parse_record_list(nested, nested_refs, shot))
            elif any(k in value for k in PRED_KEYS):
                rows.extend(parse_record_list([value], fallback_refs, shot))

    if not rows:
        raise ValueError("Impossible de détecter les prédictions dans predictions.json.")
    return rows


def compute_nli_scores(predictions: Sequence[str], references: Sequence[str], model_name: str, batch_size: int):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Chargement du modèle NLI : {model_name}")

    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        top_k=None,
    )

    inputs = [{"text": ref, "text_pair": pred} for pred, ref in zip(predictions, references)]
    outputs = classifier(inputs, batch_size=batch_size, truncation=True, max_length=512)

    entailments, neutrals, contradictions = [], [], []
    for sample_scores in outputs:
        label_scores = {str(x["label"]).lower(): float(x["score"]) for x in sample_scores}
        entailments.append(next((v for k, v in label_scores.items() if "entail" in k), 0.0))
        neutrals.append(next((v for k, v in label_scores.items() if "neutral" in k), 0.0))
        contradictions.append(next((v for k, v in label_scores.items() if "contrad" in k), 0.0))

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return entailments, neutrals, contradictions


def evaluate_group(
    shot: str,
    group: pd.DataFrame,
    sacrebleu_metric,
    rouge_metric,
    bertscore_metric,
    language: str,
    bertscore_batch_size: int,
    skip_nli: bool,
    nli_model: str,
    nli_batch_size: int,
):
    pairs = [
        (normalize_text(p), normalize_text(r))
        for p, r in zip(group["prediction"], group["reference"])
        if normalize_text(p) and normalize_text(r)
    ]
    if not pairs:
        raise ValueError(f"Aucune paire prédiction/référence valide pour {shot}.")

    predictions = [p for p, _ in pairs]
    references = [r for _, r in pairs]
    print(f"\nÉvaluation de {shot} — {len(predictions)} exemples")

    bleu = sacrebleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    rouge = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
    bert = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang=language,
        batch_size=bertscore_batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    entailments = [0.0] * len(predictions)
    neutrals = [0.0] * len(predictions)
    contradictions = [0.0] * len(predictions)
    if not skip_nli:
        entailments, neutrals, contradictions = compute_nli_scores(
            predictions, references, nli_model, nli_batch_size
        )

    rows = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_tokens, ref_tokens = tokenize(pred), tokenize(ref)
        rows.append({
            "shot": shot,
            "index": i,
            "prediction": pred,
            "reference": ref,
            "exact_match": float(pred.lower() == ref.lower()),
            "prediction_word_count": len(pred_tokens),
            "reference_word_count": len(ref_tokens),
            "length_ratio": len(pred_tokens) / max(len(ref_tokens), 1),
            "repetition_rate": repetition_rate(pred),
            "bertscore_precision": float(bert["precision"][i]),
            "bertscore_recall": float(bert["recall"][i]),
            "bertscore_f1": float(bert["f1"][i]),
            "nli_entailment": float(entailments[i]),
            "nli_neutral": float(neutrals[i]),
            "nli_contradiction": float(contradictions[i]),
        })

    analysis_df = pd.DataFrame(rows)
    result = {
        "shot": shot,
        "num_examples": len(predictions),
        "BLEU": float(bleu["score"]),
        "ROUGE-1": float(rouge["rouge1"]),
        "ROUGE-2": float(rouge["rouge2"]),
        "ROUGE-L": float(rouge["rougeL"]),
        "BERTScore_P": safe_mean(bert["precision"]),
        "BERTScore_R": safe_mean(bert["recall"]),
        "BERTScore_F1": safe_mean(bert["f1"]),
        "SelfBLEU": compute_self_bleu(predictions),
        "ExactMatch": float(analysis_df["exact_match"].mean()),
        "Distinct-1": distinct_n(predictions, 1),
        "Distinct-2": distinct_n(predictions, 2),
        "AveragePredictionWords": float(analysis_df["prediction_word_count"].mean()),
        "AverageReferenceWords": float(analysis_df["reference_word_count"].mean()),
        "AverageLengthRatio": float(analysis_df["length_ratio"].mean()),
        "AverageRepetitionRate": float(analysis_df["repetition_rate"].mean()),
        "NLI_Entailment": float(analysis_df["nli_entailment"].mean()),
        "NLI_Neutral": float(analysis_df["nli_neutral"].mean()),
        "NLI_Contradiction": float(analysis_df["nli_contradiction"].mean()),
    }
    return result, analysis_df


def shot_sort_key(label: str):
    match = re.search(r"(\d+)", label)
    return (int(match.group(1)) if match else 10**9, label)


def build_parser():
    parser = argparse.ArgumentParser(description="Évalue les prédictions LLaDA sans relancer la génération.")
    parser.add_argument("--model", choices=("pretrained", "qlora"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval_subset.json"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli-batch-size", type=int, default=4)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    return parser


def main():
    args = build_parser().parse_args()
    if args.input:
        input_path = args.input
    elif args.model:
        input_path = Path("outputs") / args.model / "predictions.json"
    else:
        raise ValueError("Indique --model qlora/pretrained ou --input <chemin>.")

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path.resolve()}")

    print(f"Lecture des prédictions : {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    fallback_refs = load_eval_references(args.eval_file)
    if fallback_refs:
        print(f"{len(fallback_refs)} références chargées depuis {args.eval_file}")
    else:
        print("Aucune référence externe chargée ; utilisation des références du JSON.")

    data = pd.DataFrame(parse_predictions(raw, fallback_refs))
    data["prediction"] = data["prediction"].map(normalize_text)
    data["reference"] = data["reference"].map(normalize_text)
    data = data[data["prediction"] != ""].reset_index(drop=True)

    print(f"{len(data)} générations détectées.")
    print("Groupes :", ", ".join(sorted(data["shot"].unique(), key=shot_sort_key)))

    print("Chargement de SacreBLEU, ROUGE et BERTScore...")
    sacrebleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    results, analyses = [], []
    for shot in sorted(data["shot"].unique(), key=shot_sort_key):
        result, analysis = evaluate_group(
            shot,
            data[data["shot"] == shot].reset_index(drop=True),
            sacrebleu_metric,
            rouge_metric,
            bertscore_metric,
            args.language,
            args.bertscore_batch_size,
            args.skip_nli,
            args.nli_model,
            args.nli_batch_size,
        )
        results.append(result)
        analyses.append(analysis)

    output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    analysis_df = pd.concat(analyses, ignore_index=True)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    results_df.to_csv(output_dir / "results_llada.csv", index=False, encoding="utf-8-sig")
    analysis_df.to_csv(output_dir / "generation_analysis.csv", index=False, encoding="utf-8-sig")

    print("\nÉvaluation terminée.")
    print(f"- {output_dir / 'results.json'}")
    print(f"- {output_dir / 'results_llada.csv'}")
    print(f"- {output_dir / 'generation_analysis.csv'}")

    cols = ["shot", "num_examples", "BLEU", "ROUGE-L", "BERTScore_F1", "NLI_Entailment", "NLI_Contradiction"]
    print("\nRésumé :")
    print(results_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
