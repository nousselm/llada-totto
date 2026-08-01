#!/usr/bin/env python3
"""
QLoRA fine-tuning of LLaDA-8B-Instruct on ToTTo table-to-text pairs.

LLaDA SFT objective:
- prompt/table tokens stay unchanged;
- only response tokens may be masked;
- the model predicts original tokens at masked response positions;
- masked-token CE is weighted by 1 / masking probability.

First test:
python train_llada_totto_qlora.py --max-train-samples 100 --epochs 1 --max-length 256

Dependencies:
pip install torch transformers==4.38.2 tokenizers==0.15.2
pip install bitsandbytes accelerate peft tqdm
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:
    raise ImportError("Install PEFT with: pip install peft") from exc


MASK_ID = 126336
SYSTEM_PROMPT = (
    "You are an expert at generating factual natural language "
    "descriptions from structured tables."
)


@dataclass
class TrainConfig:
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
    train_file: str = "data/train_subset.json"
    output_dir: str = "outputs/llada_totto_qlora"
    max_length: int = 384
    max_train_samples: int | None = None
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    log_every: int = 10
    save_every: int = 0
    min_mask_probability: float = 1e-3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    trust_remote_code: bool = True


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning of LLaDA on ToTTo.")
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--train-file", default=TrainConfig.train_file)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--min-mask-probability", type=float, default=TrainConfig.min_mask_probability)
    parser.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=TrainConfig.lora_dropout)
    return TrainConfig(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_user_content(example: dict[str, Any]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Generate one faithful description for the following table.\n\n"
        f"Table:\n{example['serialized_table']}\n\n"
        "Description:"
    )


class ToTToSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": build_user_content(example)}],
            add_generation_prompt=True,
            tokenize=False,
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(
            str(example["reference"]).strip(),
            add_special_tokens=False,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            response_ids.append(eos_id)

        if len(response_ids) >= self.max_length:
            response_ids = response_ids[: self.max_length - 1]
            if eos_id is not None:
                response_ids.append(eos_id)

        max_prompt_length = max(1, self.max_length - len(response_ids))
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]

        input_ids = prompt_ids + response_ids
        response_mask = [False] * len(prompt_ids) + [True] * len(response_ids)

        if not any(response_mask):
            raise ValueError(f"Example {index} has no response tokens.")

        return {"input_ids": input_ids, "response_mask": response_mask}


class LLaDADiffusionCollator:
    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int = MASK_ID,
        min_mask_probability: float = 1e-3,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.min_mask_probability = min_mask_probability

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        max_length = max(len(item["input_ids"]) for item in features)

        clean_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        response_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

        for row, item in enumerate(features):
            length = len(item["input_ids"])
            clean_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
            response_mask[row, :length] = torch.tensor(item["response_mask"], dtype=torch.bool)

        mask_probability = torch.rand(batch_size).clamp_min(self.min_mask_probability)
        masked_positions = (
            (torch.rand(batch_size, max_length) < mask_probability.unsqueeze(1))
            & response_mask
            & attention_mask.bool()
        )

        for row in range(batch_size):
            if not masked_positions[row].any():
                candidates = torch.where(response_mask[row])[0]
                selected = candidates[torch.randint(0, len(candidates), size=(1,))]
                masked_positions[row, selected] = True

        noisy_ids = clean_ids.clone()
        noisy_ids[masked_positions] = self.mask_token_id

        return {
            "input_ids": noisy_ids,
            "clean_ids": clean_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "masked_positions": masked_positions,
            "mask_probability": mask_probability,
        }


def diffusion_sft_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    masked_positions: torch.Tensor,
    response_mask: torch.Tensor,
    mask_probability: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        clean_ids.reshape(-1),
        reduction="none",
    ).reshape_as(clean_ids)

    weights = masked_positions.float() / mask_probability.unsqueeze(1)
    response_lengths = response_mask.sum(dim=1).clamp_min(1).float()
    per_example_loss = (token_loss * weights).sum(dim=1) / response_lengths
    return per_example_loss.mean()


def load_examples(path: Path, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        examples = json.load(file)

    valid_examples = [
        item
        for item in examples
        if {"serialized_table", "reference"}.issubset(item)
        and str(item["serialized_table"]).strip()
        and str(item["reference"]).strip()
    ]

    random.Random(seed).shuffle(valid_examples)
    if max_samples is not None:
        valid_examples = valid_examples[:max_samples]

    if not valid_examples:
        raise ValueError(f"No valid examples found in {path}.")
    return valid_examples


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    config: TrainConfig,
    global_step: int,
    history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "training_state.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"global_step": global_step, "config": asdict(config), "history": history},
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA script.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id == MASK_ID:
        raise ValueError("pad_token_id must differ from LLaDA MASK_ID.")

    examples = load_examples(Path(config.train_file), config.max_train_samples, config.seed)
    print(f"Training examples: {len(examples)}")

    dataset = ToTToSFTDataset(examples, tokenizer, config.max_length)
    collator = LLaDADiffusionCollator(
        tokenizer.pad_token_id,
        MASK_ID,
        config.min_mask_probability,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading LLaDA in 4-bit...")
    model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )
    print("Device du modèle :", next(model.parameters()).device)
    print(
    f"VRAM utilisée après chargement : "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} Go"
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters were created.")

    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    updates_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    total_update_steps = max(1, updates_per_epoch * config.epochs)
    warmup_steps = int(total_update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    device = torch.device("cuda:0")
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    update_step = 0
    history: list[dict[str, float]] = []
    running_loss = 0.0

    print(
        f"Training: epochs={config.epochs}, batch={config.batch_size}, "
        f"grad_accum={config.gradient_accumulation_steps}, max_length={config.max_length}"
    )

    for epoch in range(config.epochs):
        progress = tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{config.epochs}",
        )

        for micro_step, batch in progress:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            clean_ids = batch["clean_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            response_mask = batch["response_mask"].to(device, non_blocking=True)
            masked_positions = batch["masked_positions"].to(device, non_blocking=True)
            mask_probability = batch["mask_probability"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            loss = diffusion_sft_loss(
                outputs.logits,
                clean_ids,
                masked_positions,
                response_mask,
                mask_probability,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss}")

            (loss / config.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            global_step += 1

            should_update = (
                micro_step % config.gradient_accumulation_steps == 0
                or micro_step == len(dataloader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if config.save_every > 0 and update_step % config.save_every == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        output_dir / f"checkpoint-{update_step}",
                        config,
                        global_step,
                        history,
                    )

            if global_step % config.log_every == 0:
                average_loss = running_loss / config.log_every
                lr = scheduler.get_last_lr()[0]
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "global_step": float(global_step),
                        "update_step": float(update_step),
                        "loss": float(average_loss),
                        "learning_rate": float(lr),
                        "vram_gb": float(vram_gb),
                    }
                )
                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    lr=f"{lr:.2e}",
                    vram=f"{vram_gb:.1f}GB",
                )
                running_loss = 0.0

        save_checkpoint(
            model,
            tokenizer,
            output_dir / f"epoch-{epoch + 1}",
            config,
            global_step,
            history,
        )
        print(f"Saved: {output_dir / f'epoch-{epoch + 1}'}")

    final_dir = output_dir / "final"
    save_checkpoint(model, tokenizer, final_dir, config, global_step, history)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Training complete. Adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()