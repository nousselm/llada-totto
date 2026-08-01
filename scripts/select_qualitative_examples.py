#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
select_qualitative_examples_v2.py

Objectif
--------
1. Charger toutes les sorties GPT-2, GPT-Neo, MDLM, LLaDA et LLaDA+QLoRA.
2. Aligner uniquement les mêmes exemples.
3. Évaluer chaque génération avec des critères homogènes :
   - similarité sémantique ;
   - ROUGE-L ;
   - F1 lexical ;
   - fidélité des nombres ;
   - couverture des éléments de la table ;
   - NLI : entailment / neutral / contradiction ;
   - répétition ;
   - fuite du prompt.
4. Classer :
   - les meilleurs exemples pour une comparaison qualitative (Table 3) ;
   - les meilleurs vrais cas d'échec (Table 4).
5. Exporter les générations exactes et du LaTeX directement exploitable.

Important
---------
Le script ne réécrit jamais les prédictions. Les textes exportés sont exactement
ceux présents dans les CSV. La coloration mot à mot doit ensuite être vérifiée
manuellement, car une détection automatique de spans hallucinés n'est pas
suffisamment fiable pour une publication scientifique.

Dépendances
-----------
pip install pandas numpy torch transformers tqdm

Exécution
---------
python select_qualitative_examples_v2.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs" / "qualitative_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = [0, 1, 3, 5, 10]

# Shots retenus dans la comparaison principale.
# Modifie-les uniquement si ta Table 2 finale donne d'autres meilleurs shots.
BEST_SHOTS = {
    "GPT-2": 1,
    "GPT-Neo": 3,
    "MDLM": 10,
    "LLaDA": 5,
    "LLaDA + QLoRA": 1,
}

FILES = {
    ("GPT-2", 0): BASE_DIR / "gpt2_k0_qualitative_100.csv",
    ("GPT-2", 1): BASE_DIR / "gpt2_k1_qualitative_100.csv",
    ("GPT-2", 3): BASE_DIR / "gpt2_k3_qualitative_100.csv",
    ("GPT-2", 5): BASE_DIR / "gpt2_k5_qualitative_100.csv",
    ("GPT-2", 10): BASE_DIR / "gpt2_k10_qualitative_100.csv",

    ("GPT-Neo", 0): BASE_DIR / "gptneo_k0_qualitative.csv",
    ("GPT-Neo", 1): BASE_DIR / "gptneo_k1_qualitative.csv",
    ("GPT-Neo", 3): BASE_DIR / "gptneo_k3_qualitative.csv",
    ("GPT-Neo", 5): BASE_DIR / "gptneo_k5_qualitative.csv",
    ("GPT-Neo", 10): BASE_DIR / "gptneo_k10_qualitative.csv",

    ("MDLM", None): BASE_DIR / "generation_analysis_MDLM_with_nli.csv",
    ("LLaDA", None): BASE_DIR / "generation_analysis_LLaDA_with_nli.csv",
    ("LLaDA + QLoRA", None): BASE_DIR / "generation_analysis_LLaDA_QLoRA_with_nli.csv",
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

BATCH_SIZE_EMBEDDING = 32
BATCH_SIZE_NLI = 16
MAX_LENGTH_EMBEDDING = 256
MAX_LENGTH_NLI = 384

TOP_K = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ORDER = ["GPT-2", "GPT-Neo", "MDLM", "LLaDA", "LLaDA + QLoRA"]

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by",
    "and", "or", "as", "is", "was", "were", "be", "been", "being", "with",
    "that", "this", "these", "those", "it", "its", "into", "during", "after",
    "before", "while", "who", "which", "where", "when", "had", "has", "have",
}


# ============================================================
# 2. OUTILS TEXTE
# ============================================================

def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\u200b", " ").strip()
    return re.sub(r"\s+", " ", text)


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s.%+\-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w]+\b", normalize_text(text), flags=re.UNICODE)


def content_tokens(text: str) -> List[str]:
    return [
        token for token in tokenize(text)
        if token not in STOPWORDS and (len(token) > 2 or token.isdigit())
    ]


def extract_numbers(text: str) -> List[str]:
    pattern = r"(?<!\w)[+\-]?\d+(?:[.,]\d+)?%?"
    return [
        match.replace(",", "")
        for match in re.findall(pattern, clean_text(text))
    ]


def multiset_prf(reference_items: Sequence[str],
                 prediction_items: Sequence[str]) -> Tuple[float, float, float]:
    ref = Counter(reference_items)
    pred = Counter(prediction_items)
    overlap = sum((ref & pred).values())

    precision = overlap / sum(pred.values()) if pred else 0.0
    recall = overlap / sum(ref.values()) if ref else 1.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def lexical_f1(reference: str, prediction: str) -> float:
    return multiset_prf(tokenize(reference), tokenize(prediction))[2]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )


def repetition_rate(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def prompt_leakage_score(text: str) -> float:
    lowered = clean_text(text).lower()
    markers = [
        "\ntable:", "\ndescription:", "table :", "description :",
        "reference:", "input table:", "###", "assistant:"
    ]
    hits = sum(marker in lowered for marker in markers)
    return min(hits / 2.0, 1.0)


def length_ratio(reference: str, prediction: str) -> float:
    ref_len = max(len(tokenize(reference)), 1)
    return len(tokenize(prediction)) / ref_len


def length_penalty(ratio: float) -> float:
    # Pas de pénalité autour de 1 ; croissance progressive si trop court/long.
    if 0.65 <= ratio <= 1.8:
        return 0.0
    return min(abs(math.log(max(ratio, 1e-4))) / 3.0, 1.0)


def table_content_coverage(table: str, prediction: str) -> float:
    table_items = content_tokens(table)
    pred_items = content_tokens(prediction)
    if not table_items:
        return 0.0
    return multiset_prf(table_items, pred_items)[1]


def numeric_scores(table: str, prediction: str) -> Tuple[float, float, float]:
    return multiset_prf(extract_numbers(table), extract_numbers(prediction))


def hallucinated_number_rate(table: str, prediction: str) -> float:
    table_numbers = Counter(extract_numbers(table))
    pred_numbers = Counter(extract_numbers(prediction))
    if not pred_numbers:
        return 0.0
    unsupported = sum((pred_numbers - table_numbers).values())
    return unsupported / sum(pred_numbers.values())


# ============================================================
# 3. CHARGEMENT ET ALIGNEMENT
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Place le script dans le même dossier que les CSV ou modifie BASE_DIR/FILES."
        )


def standardize_small_model_file(
    path: Path,
    model: str,
    shot: int,
) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    required = {"table", "reference", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: colonnes manquantes {sorted(missing)}")

    # Ces fichiers ne possèdent pas d'identifiant explicite.
    # Leur ordre est supposé correspondre aux indices 0...N-1 du jeu d'évaluation.
    out = pd.DataFrame({
        "example_id": np.arange(len(df), dtype=int),
        "model": model,
        "shot": shot,
        "table": df["table"].map(clean_text),
        "reference": df["reference"].map(clean_text),
        "prediction": df["prediction"].map(clean_text),
    })
    return out


def standardize_diffusion_file(path: Path, model: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    id_col = "example_id" if "example_id" in df.columns else "index"
    required = {"shot", id_col, "reference", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: colonnes manquantes {sorted(missing)}")

    # Les fichiers LLaDA/QLoRA ne contiennent parfois pas la colonne table.
    # Elle sera récupérée ensuite depuis MDLM ou GPT.
    table_values = (
        df["table"].map(clean_text)
        if "table" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    out = pd.DataFrame({
        "example_id": pd.to_numeric(df[id_col], errors="raise").astype(int),
        "model": model,
        "shot": pd.to_numeric(df["shot"], errors="raise").astype(int),
        "table": table_values,
        "reference": df["reference"].map(clean_text),
        "prediction": df["prediction"].map(clean_text),
    })

    # On conserve les NLI déjà présents uniquement à titre de contrôle.
    for col in ["nli_entailment", "nli_neutral", "nli_contradiction", "nli_label"]:
        if col in df.columns:
            out[f"original_{col}"] = df[col].values

    return out


def load_all_generations() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for (model, shot), path in FILES.items():
        if model in {"GPT-2", "GPT-Neo"}:
            assert shot is not None
            frames.append(standardize_small_model_file(path, model, int(shot)))
        else:
            frames.append(standardize_diffusion_file(path, model))

    generations = pd.concat(frames, ignore_index=True)

    # Dictionnaire table/référence par example_id, prioritairement depuis MDLM.
    metadata_candidates = generations[
        generations["table"].astype(str).str.len() > 0
    ][["example_id", "table", "reference"]].drop_duplicates("example_id")

    table_map = metadata_candidates.set_index("example_id")["table"].to_dict()
    reference_map = metadata_candidates.set_index("example_id")["reference"].to_dict()

    generations["table"] = generations.apply(
        lambda r: r["table"] or table_map.get(r["example_id"], ""),
        axis=1,
    )
    generations["reference"] = generations.apply(
        lambda r: r["reference"] or reference_map.get(r["example_id"], ""),
        axis=1,
    )

    # Vérifications d'unicité.
    duplicate_mask = generations.duplicated(
        ["model", "shot", "example_id"], keep=False
    )
    if duplicate_mask.any():
        duplicates = generations.loc[
            duplicate_mask, ["model", "shot", "example_id"]
        ]
        raise ValueError(
            "Doublons modèle/shot/exemple détectés :\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )

    return generations.sort_values(
        ["example_id", "model", "shot"]
    ).reset_index(drop=True)


# ============================================================
# 4. SIMILARITÉ SÉMANTIQUE
# ============================================================

@torch.inference_mode()
def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    vectors = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embeddings",
    ):
        batch = list(texts[start:start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu())

    return torch.cat(vectors, dim=0)


def add_semantic_similarity(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Chargement du modèle d'embedding sur {DEVICE}: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE).eval()

    # Encoder chaque texte unique une seule fois.
    unique_texts = pd.Index(
        pd.concat([df["reference"], df["prediction"]], ignore_index=True).unique()
    )
    embeddings = encode_texts(
        unique_texts.tolist(),
        tokenizer,
        model,
        BATCH_SIZE_EMBEDDING,
        MAX_LENGTH_EMBEDDING,
    )
    embedding_map = {
        text: embeddings[i]
        for i, text in enumerate(unique_texts)
    }

    similarities = []
    for reference, prediction in zip(df["reference"], df["prediction"]):
        score = float(torch.dot(
            embedding_map[reference],
            embedding_map[prediction],
        ).item())
        similarities.append(max(min(score, 1.0), -1.0))

    out = df.copy()
    out["semantic_similarity"] = similarities

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


# ============================================================
# 5. NLI HOMOGÈNE POUR TOUS LES MODÈLES
# ============================================================

def infer_nli_label_indices(model) -> Dict[str, int]:
    labels = {
        int(idx): str(label).lower()
        for idx, label in model.config.id2label.items()
    }

    result: Dict[str, int] = {}
    for idx, label in labels.items():
        if "entail" in label:
            result["entailment"] = idx
        elif "neutral" in label:
            result["neutral"] = idx
        elif "contrad" in label:
            result["contradiction"] = idx

    if set(result) != {"entailment", "neutral", "contradiction"}:
        raise ValueError(
            f"Impossible d'identifier les labels NLI : {model.config.id2label}"
        )
    return result


@torch.inference_mode()
def add_nli_scores(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Chargement du modèle NLI sur {DEVICE}: {NLI_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL
    ).to(DEVICE).eval()

    label_indices = infer_nli_label_indices(model)

    entailment_scores: List[float] = []
    neutral_scores: List[float] = []
    contradiction_scores: List[float] = []

    references = df["reference"].tolist()
    predictions = df["prediction"].tolist()

    for start in tqdm(
        range(0, len(df), BATCH_SIZE_NLI),
        desc="NLI",
    ):
        premises = references[start:start + BATCH_SIZE_NLI]
        hypotheses = predictions[start:start + BATCH_SIZE_NLI]

        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH_NLI,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()

        entailment_scores.extend(
            probabilities[:, label_indices["entailment"]].tolist()
        )
        neutral_scores.extend(
            probabilities[:, label_indices["neutral"]].tolist()
        )
        contradiction_scores.extend(
            probabilities[:, label_indices["contradiction"]].tolist()
        )

    out = df.copy()
    out["nli_entailment"] = entailment_scores
    out["nli_neutral"] = neutral_scores
    out["nli_contradiction"] = contradiction_scores

    labels = np.array(["entailment", "neutral", "contradiction"])
    matrix = out[
        ["nli_entailment", "nli_neutral", "nli_contradiction"]
    ].to_numpy()
    out["nli_label"] = labels[matrix.argmax(axis=1)]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


# ============================================================
# 6. MÉTRIQUES ET SCORE PAR GÉNÉRATION
# ============================================================

def add_surface_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rouge_l_f1"] = [
        rouge_l_f1(r, p)
        for r, p in zip(out["reference"], out["prediction"])
    ]
    out["lexical_f1"] = [
        lexical_f1(r, p)
        for r, p in zip(out["reference"], out["prediction"])
    ]

    numeric = [
        numeric_scores(t, p)
        for t, p in zip(out["table"], out["prediction"])
    ]
    out["number_precision"] = [x[0] for x in numeric]
    out["number_recall"] = [x[1] for x in numeric]
    out["number_f1"] = [x[2] for x in numeric]

    out["hallucinated_number_rate"] = [
        hallucinated_number_rate(t, p)
        for t, p in zip(out["table"], out["prediction"])
    ]
    out["table_content_coverage"] = [
        table_content_coverage(t, p)
        for t, p in zip(out["table"], out["prediction"])
    ]
    out["repetition_rate"] = out["prediction"].map(repetition_rate)
    out["prompt_leakage"] = out["prediction"].map(prompt_leakage_score)
    out["length_ratio"] = [
        length_ratio(r, p)
        for r, p in zip(out["reference"], out["prediction"])
    ]
    out["length_penalty"] = out["length_ratio"].map(length_penalty)
    out["empty_output"] = (
        out["prediction"].map(lambda x: len(tokenize(x)) == 0).astype(float)
    )

    return out


def add_generation_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Score de qualité : combinaison interprétable, comprise approximativement entre 0 et 1.
    out["quality_score"] = (
        0.22 * out["semantic_similarity"].clip(0, 1)
        + 0.14 * out["rouge_l_f1"]
        + 0.10 * out["lexical_f1"]
        + 0.12 * out["number_f1"]
        + 0.10 * out["table_content_coverage"]
        + 0.24 * out["nli_entailment"]
        - 0.18 * out["nli_contradiction"]
        - 0.08 * out["hallucinated_number_rate"]
        - 0.08 * out["prompt_leakage"]
        - 0.05 * out["repetition_rate"]
        - 0.05 * out["length_penalty"]
        - 0.30 * out["empty_output"]
    )

    # Score d'échec : contradiction, omission, chiffres inventés, fuite du prompt, etc.
    out["failure_score"] = (
        0.34 * out["nli_contradiction"]
        + 0.14 * out["nli_neutral"]
        + 0.14 * (1.0 - out["table_content_coverage"])
        + 0.12 * (1.0 - out["number_recall"])
        + 0.10 * out["hallucinated_number_rate"]
        + 0.08 * out["prompt_leakage"]
        + 0.04 * out["repetition_rate"]
        + 0.04 * out["length_penalty"]
        + 0.30 * out["empty_output"]
    )

    return out


# ============================================================
# 7. AGRÉGATION PAR EXEMPLE
# ============================================================

def fixed_best_shot_view(scored: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for model, shot in BEST_SHOTS.items():
        selected = scored[
            (scored["model"] == model) &
            (scored["shot"] == shot)
        ].copy()
        parts.append(selected)

    view = pd.concat(parts, ignore_index=True)

    # Garder uniquement les exemples présents pour les cinq modèles.
    counts = view.groupby("example_id")["model"].nunique()
    common_ids = counts[counts == len(BEST_SHOTS)].index
    return view[view["example_id"].isin(common_ids)].copy()


def build_example_ranking(scored: pd.DataFrame) -> pd.DataFrame:
    view = fixed_best_shot_view(scored)

    score_pivot = view.pivot(
        index="example_id",
        columns="model",
        values="quality_score",
    )
    failure_pivot = view.pivot(
        index="example_id",
        columns="model",
        values="failure_score",
    )
    entailment_pivot = view.pivot(
        index="example_id",
        columns="model",
        values="nli_entailment",
    )
    contradiction_pivot = view.pivot(
        index="example_id",
        columns="model",
        values="nli_contradiction",
    )

    ranking = pd.DataFrame(index=score_pivot.index)

    # Table 3 : progression visible vers LLaDA / QLoRA.
    baselines = score_pivot[["GPT-2", "GPT-Neo", "MDLM"]].mean(axis=1)
    diffusion = score_pivot[["LLaDA", "LLaDA + QLoRA"]].mean(axis=1)

    ranking["progression_score"] = (
        0.42 * (diffusion - baselines)
        + 0.22 * score_pivot["LLaDA + QLoRA"]
        + 0.16 * score_pivot["LLaDA"]
        + 0.10 * (
            score_pivot["LLaDA + QLoRA"] - score_pivot["GPT-2"]
        )
        + 0.10 * entailment_pivot["LLaDA + QLoRA"]
    )

    # Bonus si l'ordre général attendu est visible.
    ranking["progression_score"] += (
        (score_pivot["LLaDA"] > score_pivot["MDLM"]).astype(float) * 0.05
        + (
            score_pivot["LLaDA + QLoRA"] >
            score_pivot["GPT-Neo"]
        ).astype(float) * 0.05
    )

    # Table 4 : vrai cas difficile, particulièrement intéressant si
    # LLaDA/QLoRA échouent aussi et si les modèles font des erreurs variées.
    mean_failure = failure_pivot.mean(axis=1)
    strong_model_failure = failure_pivot[
        ["LLaDA", "LLaDA + QLoRA"]
    ].mean(axis=1)

    label_diversity = view.groupby("example_id")["nli_label"].nunique()
    score_spread = score_pivot.max(axis=1) - score_pivot.min(axis=1)

    ranking["failure_case_score"] = (
        0.36 * mean_failure
        + 0.34 * strong_model_failure
        + 0.14 * contradiction_pivot["LLaDA + QLoRA"]
        + 0.08 * (label_diversity / 3.0)
        + 0.08 * score_spread
    )

    metadata = (
        view.sort_values(["example_id", "model"])
        .drop_duplicates("example_id")
        .set_index("example_id")[["table", "reference"]]
    )
    ranking = ranking.join(metadata)

    return ranking.reset_index()


# ============================================================
# 8. EXPORTS
# ============================================================

def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in clean_text(text))


def compact_prediction(text: str) -> str:
    # On conserve le texte exact sur le fond, mais les espaces/retours sont normalisés
    # pour éviter de casser la table LaTeX.
    return clean_text(text)


def build_latex_table(
    example_id: int,
    view: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    rows = view[view["example_id"] == example_id].copy()
    rows["model_order"] = rows["model"].map({
        model: i for i, model in enumerate(MODEL_ORDER)
    })
    rows = rows.sort_values("model_order")

    if rows.empty:
        raise ValueError(f"Exemple {example_id} absent de la vue.")

    table_text = latex_escape(rows.iloc[0]["table"])
    reference_text = latex_escape(rows.iloc[0]["reference"])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabular}{p{0.17\linewidth} p{0.77\linewidth}}",
        r"\toprule",
        r"\textbf{Input Table} & " + table_text + r" \\",
        r"\midrule",
        r"\textbf{Reference} & " + reference_text + r" \\",
    ]

    for _, row in rows.iterrows():
        model_name = latex_escape(
            f"{row['model']} ({int(row['shot'])}-shot)"
        )
        prediction = latex_escape(compact_prediction(row["prediction"]))
        nli = latex_escape(
            f"[NLI: {row['nli_label']}; "
            f"E={row['nli_entailment']:.2f}, "
            f"C={row['nli_contradiction']:.2f}]"
        )

        lines.extend([
            r"\midrule",
            rf"\textbf{{{model_name}}} & {prediction} "
            rf"\textit{{\footnotesize {nli}}} \\",
        ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def export_candidate_details(
    ranking: pd.DataFrame,
    view: pd.DataFrame,
    score_column: str,
    filename: str,
    top_k: int,
) -> pd.DataFrame:
    top_ids = ranking.nlargest(top_k, score_column)["example_id"].tolist()
    detail = view[view["example_id"].isin(top_ids)].copy()

    rank_map = {example_id: rank + 1 for rank, example_id in enumerate(top_ids)}
    detail["candidate_rank"] = detail["example_id"].map(rank_map)
    detail = detail.sort_values(
        ["candidate_rank", "example_id", "model"]
    )
    detail.to_csv(OUTPUT_DIR / filename, index=False)
    return detail


def export_latex_candidates(
    ranking: pd.DataFrame,
    view: pd.DataFrame,
) -> None:
    best_id = int(
        ranking.nlargest(1, "progression_score").iloc[0]["example_id"]
    )
    failure_id = int(
        ranking.nlargest(1, "failure_case_score").iloc[0]["example_id"]
    )

    table3 = build_latex_table(
        best_id,
        view,
        caption=(
            "Qualitative comparison on a representative ToTTo example. "
            "All outputs are reproduced verbatim from the generation files."
        ),
        label="tab:qualitative",
    )
    table4 = build_latex_table(
        failure_id,
        view,
        caption=(
            "Failure case illustrating common generation errors across models. "
            "All outputs are reproduced verbatim from the generation files."
        ),
        label="tab:failure",
    )

    (OUTPUT_DIR / "table3_candidate.tex").write_text(
        table3, encoding="utf-8"
    )
    (OUTPUT_DIR / "table4_candidate.tex").write_text(
        table4, encoding="utf-8"
    )

    print(f"Meilleur candidat Table 3 : example_id={best_id}")
    print(f"Meilleur candidat Table 4 : example_id={failure_id}")


def export_shot_sensitivity(scored: pd.DataFrame) -> None:
    """
    Bonus : identifie les exemples où augmenter le nombre de shots dégrade
    fortement la qualité. Utile pour une analyse ICL complémentaire.
    """
    pivot = scored.pivot_table(
        index=["model", "example_id"],
        columns="shot",
        values="quality_score",
        aggfunc="first",
    )

    available_shots = [shot for shot in SHOTS if shot in pivot.columns]
    if 0 not in available_shots or 10 not in available_shots:
        return

    pivot["best_quality"] = pivot[available_shots].max(axis=1)
    pivot["worst_quality"] = pivot[available_shots].min(axis=1)
    pivot["shot_sensitivity"] = pivot["best_quality"] - pivot["worst_quality"]
    pivot["degradation_0_to_10"] = pivot[0] - pivot[10]

    pivot.sort_values(
        "shot_sensitivity", ascending=False
    ).reset_index().to_csv(
        OUTPUT_DIR / "shot_sensitivity.csv",
        index=False,
    )


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    print("=" * 72)
    print("Sélection qualitative V2")
    print(f"Device : {DEVICE}")
    print("=" * 72)

    generations = load_all_generations()
    print(
        f"{len(generations)} générations chargées, "
        f"{generations['example_id'].nunique()} IDs distincts."
    )

    generations = add_surface_metrics(generations)
    generations = add_semantic_similarity(generations)
    generations = add_nli_scores(generations)
    scored = add_generation_scores(generations)

    scored.to_csv(
        OUTPUT_DIR / "all_generations_scored.csv",
        index=False,
    )

    ranking = build_example_ranking(scored)
    ranking.sort_values(
        "progression_score", ascending=False
    ).to_csv(
        OUTPUT_DIR / "example_ranking.csv",
        index=False,
    )

    fixed_view = fixed_best_shot_view(scored)

    export_candidate_details(
        ranking,
        fixed_view,
        "progression_score",
        "table3_top_candidates.csv",
        TOP_K,
    )
    export_candidate_details(
        ranking,
        fixed_view,
        "failure_case_score",
        "table4_top_candidates.csv",
        TOP_K,
    )

    export_latex_candidates(ranking, fixed_view)
    export_shot_sensitivity(scored)

    print("\nFichiers créés dans :", OUTPUT_DIR.resolve())
    print(" - all_generations_scored.csv")
    print(" - example_ranking.csv")
    print(" - table3_top_candidates.csv")
    print(" - table4_top_candidates.csv")
    print(" - table3_candidate.tex")
    print(" - table4_candidate.tex")
    print(" - shot_sensitivity.csv")
    print("\nVérifie manuellement le candidat final avant publication,")
    print("notamment les annotations rouge/bleu/vert.")


if __name__ == "__main__":
    main()
