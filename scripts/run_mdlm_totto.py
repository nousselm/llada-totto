#!/usr/bin/env python3
# ============================================================
# MDLM + ToTTo Few-Shot Evaluation Pipeline
# Pretrained model only — no fine-tuning
# Compatible with the local dllm repository on Windows.
# ============================================================

import gc
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
RETRIEVAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train_subset.json"
EVAL_PATH = DATA_DIR / "eval_subset.json"

# Local custom implementation validated in run_mdlm_v3.py.
LOCAL_MODELING_FILE = (
    PROJECT_DIR
    / "dllm"
    / "dllm"
    / "pipelines"
    / "a2d"
    / "models"
    / "qwen3"
    / "modeling_qwen3.py"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "mdlm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
USE_FIXED_SEED = True

SHOT_SETS = [0, 1, 3, 5, 10]
MAX_RETRIEVAL_SHOTS = max(SHOT_SETS)

# Parameters from the supplied notebook.
GEN_STEPS = 64
MAX_NEW_TOKENS = 128
BLOCK_SIZE = 64
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"

# Long few-shot prompts are truncated from the left so that the target
# table, located at the end of the prompt, is preserved.
MAX_PROMPT_LENGTH = 2048

# First run: True -> only 5 evaluation samples.
# Complete experiment: False -> all evaluation samples.
DEBUG = False
DEBUG_SAMPLES = 5

# Compute BLEU, ROUGE, BERTScore and Self-BLEU after generation.
# Requires: pip install evaluate sacrebleu rouge_score bert_score nltk
COMPUTE_METRICS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Le script attend data/train_subset.json et "
            "data/eval_subset.json."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_local_mdlm_registration() -> None:
    """
    Charge directement modeling_qwen3.py sans importer le package dllm
    complet. Cela évite les dépendances d'entraînement telles que lm_eval
    et deepspeed, tout en exécutant les enregistrements AutoConfig /
    AutoModelForMaskedLM présents dans le fichier.
    """
    if not LOCAL_MODELING_FILE.exists():
        raise FileNotFoundError(
            "Le fichier custom MDLM est introuvable :\n"
            f"{LOCAL_MODELING_FILE}\n\n"
            "Vérifie que le dépôt dllm se trouve bien dans le dossier "
            "racine du projet."
        )

    module_name = "modeling_qwen3"

    spec = importlib.util.spec_from_file_location(
        module_name,
        LOCAL_MODELING_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Impossible de créer le module depuis {LOCAL_MODELING_FILE}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "A2DQwen3Config"):
        raise ImportError(
            "modeling_qwen3.py a été chargé, mais A2DQwen3Config "
            "est introuvable."
        )

    print("✓ A2DQwen3Config enregistré")


def patch_attention_type(model: torch.nn.Module) -> int:
    """
    Compatibilité entre l'implémentation A2D et les versions récentes
    de Transformers, où Qwen3DecoderLayer n'expose plus attention_type.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)

    if layers is None:
        raise AttributeError(
            "Impossible de trouver model.model.layers pour appliquer "
            "le correctif attention_type."
        )

    patched = 0

    for layer in layers:
        if not hasattr(layer, "attention_type"):
            layer.attention_type = "full_attention"
            patched += 1

    return patched


# ============================================================
# DATA
# ============================================================

print("=" * 80)
print("MDLM + ToTTo pretrained few-shot evaluation")
print("=" * 80)
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"CUDA   : {torch.version.cuda}")

train_subset = load_json(TRAIN_PATH)
eval_subset = load_json(EVAL_PATH)

if USE_FIXED_SEED:
    random.Random(SEED).shuffle(train_subset)

N_SAMPLES = (
    min(DEBUG_SAMPLES, len(eval_subset))
    if DEBUG
    else len(eval_subset)
)

print(f"Train examples : {len(train_subset)}")
print(f"Eval examples  : {N_SAMPLES}")
print(f"Shots          : {SHOT_SETS}")


# ============================================================
# RETRIEVAL
# ============================================================

def format_table(example: dict) -> str:
    return f"Table: {example['serialized_table']}\n"


print("\nChargement du modèle de retrieval...")

retrieval_tokenizer = AutoTokenizer.from_pretrained(
    RETRIEVAL_MODEL_NAME
)
retrieval_model = AutoModel.from_pretrained(
    RETRIEVAL_MODEL_NAME
).to(DEVICE)
retrieval_model.eval()

print("✓ Retrieval chargé")


@torch.no_grad()
def encode_texts(
    texts: list[str],
    batch_size: int = 32,
) -> torch.Tensor:
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = retrieval_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = retrieval_model(**encoded, return_dict=True)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]

        expanded_mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        sentence_embeddings = (
            (token_embeddings * expanded_mask).sum(dim=1)
            / expanded_mask.sum(dim=1).clamp(min=1e-9)
        )

        embeddings.append(
            F.normalize(sentence_embeddings, p=2, dim=1)
        )

    return torch.cat(embeddings, dim=0)


train_texts = [
    format_table(example)
    for example in train_subset
]

print("Calcul des embeddings du train...")
train_embeddings = encode_texts(train_texts)
print(f"Train embeddings shape: {tuple(train_embeddings.shape)}")


@torch.no_grad()
def retrieve(
    query: str,
    top_k: int = MAX_RETRIEVAL_SHOTS,
) -> list[int]:
    query_embedding = encode_texts([query])[0]
    similarities = torch.matmul(
        train_embeddings,
        query_embedding,
    )
    indices = torch.argsort(
        similarities,
        descending=True,
    )

    return indices[:top_k].cpu().tolist()


print("Construction des ensembles few-shot imbriqués...")

nested_retrieval: dict[int, dict[int, list[int]]] = {}

for index, example in enumerate(
    tqdm(eval_subset[:N_SAMPLES], desc="Retrieval")
):
    top_indices = retrieve(
        format_table(example),
        MAX_RETRIEVAL_SHOTS,
    )

    nested_retrieval[index] = {
        shot: top_indices[:shot] if shot > 0 else []
        for shot in SHOT_SETS
    }

save_json(
    nested_retrieval,
    OUTPUT_DIR / "retrieval_ids.json",
)

# The retrieval model is no longer needed. Releasing it prevents
# competition for VRAM with the MDLM.
del retrieval_model
del retrieval_tokenizer
del train_embeddings

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✓ Retrieval terminé et mémoire libérée")


# ============================================================
# LOAD PRETRAINED MDLM
# ============================================================

print("\nChargement des classes MDLM locales...")
load_local_mdlm_registration()

print("Chargement du tokenizer MDLM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Le tokenizer ne définit ni pad_token_id ni eos_token_id."
        )
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.truncation_side = "left"

if DEVICE.type == "cuda":
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    compute_dtype = torch.float32

print(f"Chargement du modèle MDLM en {compute_dtype}...")

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

patched_layers = patch_attention_type(model)

PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id

if MASK_ID is None:
    raise ValueError(
        "Le tokenizer MDLM ne définit pas mask_token_id."
    )

print("✓ Modèle MDLM chargé")
print(f"✓ attention_type ajouté à {patched_layers} couches")
print(f"mask_token_id={MASK_ID} | pad_token_id={PAD_ID}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM allouée: {allocated:.2f} Gio")


# ============================================================
# MDLM GENERATION FUNCTIONS
# Logic supplied by the user from the Hugging Face notebook.
# ============================================================

def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(
        logits,
        dtype=torch.float64,
    )
    gumbel_noise = (-torch.log(noise)) ** temperature

    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        )
        + base
    )

    for batch_index in range(mask_num.size(0)):
        remainder_count = int(
            remainder[batch_index].item()
        )
        num_transfer_tokens[
            batch_index,
            :remainder_count,
        ] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_mdlm(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
    steps: int = GEN_STEPS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    block_size: int = BLOCK_SIZE,
    temperature: float = TEMPERATURE,
    cfg_scale: float = CFG_SCALE,
    remasking: str = REMASKING,
) -> torch.Tensor:
    batch_size = prompt.size(0)
    total_length = (
        int(prompt_lens.max().item())
        + max_new_tokens
    )

    generated = torch.full(
        (batch_size, total_length),
        pad_id,
        dtype=torch.long,
        device=DEVICE,
    )

    for batch_index, prompt_length in enumerate(
        prompt_lens.tolist()
    ):
        generated[
            batch_index,
            :prompt_length,
        ] = prompt[
            batch_index,
            :prompt_length,
        ]

        generated[
            batch_index,
            prompt_length:prompt_length + max_new_tokens,
        ] = MASK_ID

    prompt_index = (
        torch.arange(
            total_length,
            device=DEVICE,
        ).unsqueeze(0)
        < prompt_lens.unsqueeze(1)
    )

    positions = torch.arange(
        total_length,
        device=DEVICE,
    )

    if max_new_tokens % block_size != 0:
        raise ValueError(
            f"MAX_NEW_TOKENS ({max_new_tokens}) doit être "
            f"divisible par BLOCK_SIZE ({block_size})."
        )

    num_blocks = max_new_tokens // block_size

    if steps % num_blocks != 0:
        raise ValueError(
            f"GEN_STEPS ({steps}) doit être divisible par "
            f"le nombre de blocs ({num_blocks})."
        )

    steps_per_block = steps // num_blocks

    for block_number in range(num_blocks):
        block_start = (
            prompt_lens
            + block_number * block_size
        )
        block_end = block_start + block_size

        initial_block_mask = (
            (
                positions.unsqueeze(0)
                >= block_start.unsqueeze(1)
            )
            & (
                positions.unsqueeze(0)
                < block_end.unsqueeze(1)
            )
            & (generated == MASK_ID)
        )

        transfer_counts = get_num_transfer_tokens(
            initial_block_mask,
            steps_per_block,
        )

        for step_index in range(steps_per_block):
            block_mask = (
                (
                    positions.unsqueeze(0)
                    >= block_start.unsqueeze(1)
                )
                & (
                    positions.unsqueeze(0)
                    < block_end.unsqueeze(1)
                )
                & (generated == MASK_ID)
            )

            if cfg_scale > 0.0:
                unconditional = generated.clone()
                unconditional[prompt_index] = MASK_ID

                combined = torch.cat(
                    [generated, unconditional],
                    dim=0,
                )

                logits = model(combined).logits

                (
                    conditional_logits,
                    unconditional_logits,
                ) = torch.chunk(
                    logits,
                    2,
                    dim=0,
                )

                logits = (
                    unconditional_logits
                    + (cfg_scale + 1.0)
                    * (
                        conditional_logits
                        - unconditional_logits
                    )
                )
            else:
                logits = model(generated).logits

            noisy_logits = add_gumbel_noise(
                logits,
                temperature,
            )

            predicted_tokens = torch.argmax(
                noisy_logits,
                dim=-1,
            )

            if remasking == "low_confidence":
                probabilities = F.softmax(
                    logits,
                    dim=-1,
                )
                token_confidence = torch.gather(
                    probabilities,
                    dim=-1,
                    index=predicted_tokens.unsqueeze(-1),
                ).squeeze(-1)

            elif remasking == "random":
                token_confidence = torch.rand_like(
                    predicted_tokens,
                    dtype=torch.float,
                )

            else:
                raise NotImplementedError(remasking)

            confidence = torch.full_like(
                token_confidence,
                -np.inf,
            )

            confidence = torch.where(
                block_mask,
                token_confidence,
                confidence,
            )

            predicted_tokens = torch.where(
                block_mask,
                predicted_tokens,
                generated,
            )

            transfer_index = torch.zeros_like(
                predicted_tokens,
                dtype=torch.bool,
            )

            for batch_index in range(
                confidence.shape[0]
            ):
                number_to_transfer = int(
                    transfer_counts[
                        batch_index,
                        step_index,
                    ].item()
                )

                if number_to_transfer == 0:
                    continue

                _, selected_indices = torch.topk(
                    confidence[batch_index],
                    k=number_to_transfer,
                )

                transfer_index[
                    batch_index,
                    selected_indices,
                ] = True

            generated[transfer_index] = (
                predicted_tokens[transfer_index]
            )

    return generated


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert at generating factual natural language "
    "descriptions from structured tables."
)


def build_conversation(
    example: dict,
    retrieved_examples: list[dict] | None = None,
) -> list[dict]:
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if retrieved_examples:
        for retrieved_example in retrieved_examples:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Table:\n"
                        f"{retrieved_example['serialized_table']}\n\n"
                        "Description:"
                    ),
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": retrieved_example["reference"],
                }
            )

    conversation.append(
        {
            "role": "user",
            "content": (
                "Generate one faithful description for the following "
                "table.\n\n"
                "Table:\n"
                f"{example['serialized_table']}\n\n"
                "Description:"
            ),
        }
    )

    return conversation


def tokenize_conversation(
    conversation: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        # Fallback for tokenizers without a chat template.
        parts = []

        for message in conversation:
            role = message["role"].upper()
            parts.append(
                f"{role}: {message['content']}"
            )

        parts.append("ASSISTANT:")
        prompt_text = "\n\n".join(parts)

    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
        return_tensors="pt",
    )

    prompt_ids = encoded["input_ids"].to(DEVICE)
    prompt_lens = (
        encoded["attention_mask"]
        .sum(dim=1)
        .to(DEVICE)
    )

    return prompt_ids, prompt_lens


def clean_prediction(text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = (
        text
        .replace("<think>", "")
        .replace("</think>", "")
    )

    return " ".join(text.strip().split())


@torch.no_grad()
def generate_description(
    prompt_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> str:
    output = generate_mdlm(
        model=model,
        prompt=prompt_ids,
        prompt_lens=prompt_lens,
        pad_id=PAD_ID,
    )

    prompt_length = int(
        prompt_lens[0].item()
    )

    new_tokens = output[
        0,
        prompt_length:prompt_length + MAX_NEW_TOKENS,
    ]

    prediction = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return clean_prediction(prediction)


def predict(
    example_index: int,
    shot: int,
) -> str:
    example = eval_subset[example_index]
    retrieved_examples = []

    if shot > 0:
        retrieved_indices = (
            nested_retrieval[example_index][shot]
        )

        retrieved_examples = [
            train_subset[index]
            for index in retrieved_indices
        ]

    conversation = build_conversation(
        example,
        retrieved_examples,
    )

    prompt_ids, prompt_lens = (
        tokenize_conversation(conversation)
    )

    if example_index == 0:
        prompt_preview = tokenizer.decode(
            prompt_ids[0],
            skip_special_tokens=False,
        )

        print("\n" + "=" * 80)
        print(f"{shot}-shot prompt preview")
        print("=" * 80)
        print(prompt_preview[:1500])
        print("=" * 80)

    prediction = generate_description(
        prompt_ids,
        prompt_lens,
    )

    if example_index == 0:
        print(f"Prediction: {prediction}\n")

    return prediction


# ============================================================
# GENERATION
# ============================================================

predictions: dict[int, list[str]] = {
    shot: []
    for shot in SHOT_SETS
}

predictions_path = OUTPUT_DIR / "predictions.json"

for shot in SHOT_SETS:
    print(f"\nGénération {shot}-shot...")

    for example_index in tqdm(
        range(N_SAMPLES),
        desc=f"{shot}-shot",
    ):
        try:
            prediction = predict(
                example_index,
                shot,
            )

        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise RuntimeError(
                f"CUDA out of memory pendant la génération "
                f"{shot}-shot, exemple {example_index}. "
                "Réduis MAX_PROMPT_LENGTH, MAX_NEW_TOKENS "
                "ou le nombre de shots."
            ) from error

        predictions[shot].append(prediction)

        # Checkpoint after every prediction.
        save_json(predictions, predictions_path)

print("\n✓ Génération terminée")


# ============================================================
# EXPORT PREDICTIONS
# ============================================================

rows = []

for shot in SHOT_SETS:
    for example_index, prediction in enumerate(
        predictions[shot]
    ):
        rows.append(
            {
                "shot": shot,
                "example_id": example_index,
                "table": eval_subset[
                    example_index
                ]["serialized_table"],
                "reference": eval_subset[
                    example_index
                ]["reference"],
                "prediction": prediction,
                "retrieval_ids": json.dumps(
                    nested_retrieval[
                        example_index
                    ][shot]
                ),
            }
        )

analysis = pd.DataFrame(rows)
analysis_path = (
    OUTPUT_DIR
    / "generation_analysis.csv"
)
analysis.to_csv(
    analysis_path,
    index=False,
    encoding="utf-8",
)


# ============================================================
# METRICS
# ============================================================

def compute_self_bleu(
    generated_sentences: list[str],
) -> float:
    if len(generated_sentences) < 2:
        return 0.0

    try:
        from nltk.translate.bleu_score import (
            SmoothingFunction,
            sentence_bleu,
        )
    except ImportError as error:
        raise ImportError(
            "NLTK est nécessaire pour Self-BLEU. "
            "Installe-le avec : pip install nltk"
        ) from error

    smoothing = SmoothingFunction().method1
    scores = []

    for index, prediction in enumerate(
        generated_sentences
    ):
        references = (
            generated_sentences[:index]
            + generated_sentences[index + 1:]
        )

        references_tokens = [
            reference.split()
            for reference in references
            if reference.strip()
        ]
        prediction_tokens = prediction.split()

        if (
            not references_tokens
            or not prediction_tokens
        ):
            scores.append(0.0)
            continue

        score = sentence_bleu(
            references_tokens,
            prediction_tokens,
            smoothing_function=smoothing,
        )
        scores.append(score)

    return float(np.mean(scores))


def evaluate_predictions(
    all_predictions: dict[int, list[str]],
) -> dict[int, dict[str, float]]:
    try:
        import evaluate
    except ImportError as error:
        raise ImportError(
            "Le package evaluate est nécessaire. "
            "Installe les métriques avec :\n"
            "pip install evaluate sacrebleu rouge_score "
            "bert_score nltk"
        ) from error

    print("\nChargement des métriques...")

    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    references = [
        example["reference"]
        for example in eval_subset[:N_SAMPLES]
    ]

    results = {}

    for shot in SHOT_SETS:
        shot_predictions = all_predictions[shot]

        print(f"Calcul des métriques {shot}-shot...")

        bleu_result = bleu_metric.compute(
            predictions=shot_predictions,
            references=[
                [reference]
                for reference in references
            ],
        )

        rouge_result = rouge_metric.compute(
            predictions=shot_predictions,
            references=references,
        )

        bertscore_result = bertscore_metric.compute(
            predictions=shot_predictions,
            references=references,
            lang="en",
        )

        results[shot] = {
            "BLEU": float(bleu_result["score"]),
            "ROUGE1": float(rouge_result["rouge1"]),
            "ROUGE2": float(rouge_result["rouge2"]),
            "ROUGEL": float(rouge_result["rougeL"]),
            "BERTScore": float(
                np.mean(bertscore_result["f1"])
            ),
            "SelfBLEU": compute_self_bleu(
                shot_predictions
            ),
        }

    return results


results = {}

if COMPUTE_METRICS:
    try:
        results = evaluate_predictions(predictions)

        save_json(
            results,
            OUTPUT_DIR / "results.json",
        )

        results_table = (
            pd.DataFrame(results)
            .T
        )
        results_table.index = [
            f"{shot}-shot"
            for shot in SHOT_SETS
        ]
        results_table.index.name = "setting"

        results_path = (
            OUTPUT_DIR
            / "results_mdlm.csv"
        )
        results_table.to_csv(
            results_path,
            encoding="utf-8",
        )

        print("\nRésultats :")
        print(results_table)

    except (ImportError, ModuleNotFoundError) as error:
        print("\n⚠ Métriques non calculées :")
        print(error)
        print(
            "Les prédictions ont tout de même été sauvegardées."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PIPELINE TERMINÉ")
print("=" * 80)
print(f"Predictions   : {predictions_path}")
print(
    f"Retrieval IDs : "
    f"{OUTPUT_DIR / 'retrieval_ids.json'}"
)
print(f"Analysis CSV : {analysis_path}")

if COMPUTE_METRICS and results:
    print(
        f"Metrics JSON : "
        f"{OUTPUT_DIR / 'results.json'}"
    )
    print(
        f"Metrics CSV  : "
        f"{OUTPUT_DIR / 'results_mdlm.csv'}"
    )
