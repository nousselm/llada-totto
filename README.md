
# A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation

**Research Code and Experimental Materials**

**Authors:** Zaynab Merimi*, Noussayba El Marrakchi*, Arthur Satouf

*Equal contribution.*

[![arXiv](https://img.shields.io/badge/arXiv-2610.29602-b31b1b.svg)](https://arxiv.org/abs/2610.29602)

**Paper:** [arXiv:2610.29602](https://arxiv.org/abs/2610.29602) | [PDF](https://arxiv.org/pdf/2610.29602)

---

## Overview

This repository accompanies our research paper, **"A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation."**

We investigate factual faithfulness in table-to-text generation, comparing masked diffusion language models with autoregressive baselines on the **ToTTo benchmark**.

Our central research question is:

> Does improved text generation quality necessarily imply better factual faithfulness to structured source data?

We evaluate models along two complementary dimensions:

- **Reference similarity:** How closely does generated text match the reference description?
- **Source faithfulness:** How reliably does generated text remain grounded in the source table?

### Key Finding

**Higher reference similarity does not necessarily imply better factual grounding.**

Within LLaDA, increasing the number of few-shot demonstrations substantially improves BLEU scores while decreasing NLI entailment against the source table.

QLoRA adaptation achieves the highest BLEU score in our experiments, but does not consistently improve factual faithfulness.

---

## Research Objectives

This study addresses four questions:

1. How faithfully do diffusion language models generate natural-language descriptions from structured tables?
2. How does the number of in-context demonstrations affect generation quality and factual grounding?
3. Does QLoRA adaptation improve the performance and faithfulness of LLaDA?
4. What types of factual errors characterize diffusion and autoregressive language models?

---

## Models

We evaluate five model configurations:

| Model | Architecture | Parameters | Setting |
|---|---|---|---|
| GPT-2 | Autoregressive | 117M | Pretrained |
| GPT-Neo | Autoregressive | 1.3B | Pretrained |
| MDLM | Masked Diffusion | 0.6B | Pretrained |
| LLaDA | Masked Diffusion | 8B | Pretrained |
| LLaDA + QLoRA | Masked Diffusion | 8B | Fine-tuned |

### Autoregressive Baselines

**GPT-2 and GPT-Neo** generate text sequentially, predicting each token conditioned on previously generated tokens.

They provide autoregressive reference points for our diagnostic comparison.

### Masked Diffusion Models

**MDLM and LLaDA** generate text through iterative denoising rather than strictly left-to-right decoding.

Their generation mechanisms allow us to investigate whether different decoding paradigms exhibit distinct factual error patterns.

### QLoRA Adaptation

We additionally investigate a parameter-efficient adaptation of LLaDA using **Quantized Low-Rank Adaptation (QLoRA)**.

The objective is to examine whether task-specific adaptation improves reference similarity and factual consistency.

**Important:** The evaluated models differ in size and pretraining. Our experiments therefore do not establish causal advantages of one model architecture over another.

---

## Dataset

We use **ToTTo (Table-to-Text)**, a benchmark for generating natural-language descriptions grounded in structured Wikipedia tables.

Dataset: https://github.com/google-research-datasets/ToTTo

ToTTo provides:

- Structured source tables
- Highlighted table cells
- Natural-language reference descriptions

The highlighted cells help define which information should be verbalized and provide a basis for examining factual grounding.

### Experimental Protocol

We evaluate generation using retrieval-based few-shot prompting.

The reported shot configurations are:

- 0-shot
- 1-shot
- 3-shot
- 5-shot
- 10-shot

The experiments investigate how additional demonstrations influence generation quality and faithfulness.

---

## Evaluation Framework

Our evaluation separates reference similarity from source-grounded factual consistency.

### Reference-Based Metrics

| Metric | Purpose |
|---|---|
| BLEU | Lexical overlap with reference descriptions |
| ROUGE-L | Longest-common-subsequence similarity |
| BERTScore | Semantic similarity to reference descriptions |
| Self-BLEU | Similarity among generated outputs; diversity proxy |

### Faithfulness Evaluation

We use Natural Language Inference (NLI) to assess whether generated statements are supported by the source table.

**NLI direction:**

- **Premise:** Serialized source table
- **Hypothesis:** Generated description

We examine entailment and contradiction scores as automated indicators of grounding.

NLI is an imperfect proxy for factual faithfulness and may be sensitive to generation length and information coverage.

### Manual Evaluation

We complement automatic evaluation with manual annotation of 30 examples per model at the 3-shot setting.

The analysis examines error patterns including:

- Faithful generation
- Entity substitution
- Topic drift
- Other unsupported content

Manual evaluation provides qualitative insight into errors that reference-based metrics may fail to capture.

---

## Main Results

### Best BLEU Scores

| Model | Best BLEU | Shot Setting |
|---|---:|---:|
| GPT-2 | 4.64 | 1 |
| GPT-Neo | 5.74 | 1 |
| MDLM | 7.93 | 10 |
| LLaDA | 21.96 | 5 |
| LLaDA + QLoRA | **24.15** | 1 |

These values are reported in our research paper.

### Reference Similarity vs. Faithfulness

For pretrained LLaDA:

| Setting | BLEU | NLI Entailment |
|---|---:|---:|
| 0-shot | 11.25 | 0.781 |
| 5-shot | 21.96 | 0.516 |

Increasing demonstrations improves reference similarity while reducing measured source-table entailment.

This illustrates the central distinction investigated in our study:

**A generation can resemble its reference more closely without becoming more faithful to its source.**

### Effect of QLoRA

QLoRA-adapted LLaDA achieves the highest BLEU score in the study.

However:

- Reference similarity improvements are not uniformly statistically significant.
- Higher BLEU does not establish improved source faithfulness.
- Task adaptation may improve generation quality without resolving factual inconsistencies.

For complete experimental results, statistical tests, and qualitative examples, consult the paper.

---

## Repository Structure

```text
llada-totto/
├── data/
│   ├── train_subset.json
│   └── eval_subset.json
│
├── dataset_adapters/
│   ├── __init__.py
│   └── totto.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_llada_totto_qlora.py
│   ├── run_gpt2_totto_pretrained.py
│   ├── run_gptneo_totto_pretrained.py
│   ├── run_mdlm_totto.py
│   ├── run_llada_totto.py
│   ├── run_llada_totto_pretrained.py
│   ├── run_llada_totto_qlora.py
│   ├── evaluate_predictions.py
│   └── select_qualitative_examples.py
│
├── outputs/
├── results/
├── requirements.txt
└── README.md
```

The repository contains prepared dataset subsets, model scripts, evaluation utilities, result summaries, and experimental artifacts.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nousselm/llada-totto.git
cd llada-totto
```

### 2. Create a Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The repository declares dependencies including:

- PyTorch
- Hugging Face Transformers
- Accelerate
- PEFT
- bitsandbytes
- Sentence Transformers
- Hugging Face Datasets
- SacreBLEU
- ROUGE Score
- BERTScore
- NumPy, Pandas, SciPy and scikit-learn

**Note:** Running the larger models requires suitable accelerator hardware, model checkpoint access, and compatible library versions.

The installation procedure has not been validated across all operating systems and hardware configurations.

---

## Running Experiments

Run the following commands from the repository root.

### 1. Prepare the Dataset

```bash
python scripts/prepare_dataset.py
```

This script creates dataset subsets in:

```text
data/train_subset.json
data/eval_subset.json
```

Consult the script for the sampling and preprocessing configuration.

### 2. Run Autoregressive Baselines

GPT-2:

```bash
python scripts/run_gpt2_totto_pretrained.py
```

GPT-Neo:

```bash
python scripts/run_gptneo_totto_pretrained.py
```

### 3. Run Diffusion Models

MDLM:

```bash
python scripts/run_mdlm_totto.py
```

Pretrained LLaDA:

```bash
python scripts/run_llada_totto_pretrained.py
```

The repository also contains:

```text
scripts/run_llada_totto.py
```

Some scripts require checkpoint access, accelerator resources, or configuration of paths and generation parameters.

### 4. Fine-Tune LLaDA with QLoRA

The repository includes a QLoRA training implementation:

```bash
python scripts/train_llada_totto_qlora.py --help
```

The training script exposes configuration options including:

- Model checkpoint
- Training dataset
- Output directory
- Number of epochs
- Batch size
- Learning rate
- Gradient accumulation
- LoRA rank and scaling parameters
- Random seed

Trained adapters and checkpoints are saved to the configured output directory.

### 5. Run QLoRA Inference

The repository provides:

```text
scripts/run_llada_totto_qlora.py
```

Configure the adapter or checkpoint path before running inference.

### 6. Evaluate Predictions

To evaluate an existing predictions file:

```bash
python scripts/evaluate_predictions.py \
    --input outputs/qlora/predictions.json
```

The specified predictions file must already exist.

Additional options include:

```text
--eval-file
--nli-model
--skip-nli
--nli-batch-size
--bertscore-batch-size
```

The evaluation script generates summary metrics and analysis files alongside the prediction data.

### 7. Qualitative Analysis

The repository also provides:

```text
scripts/select_qualitative_examples.py
```

This script supports selecting and examining qualitative generation examples.

Review its configuration before execution.

---

## Reproducibility

The repository contains materials supporting experimental analysis, including:

- Prepared training and evaluation subsets
- Retrieval identifiers
- Generation scripts
- QLoRA training code
- Automatic evaluation utilities
- Result summaries
- Qualitative analysis scripts

For faithful replication, the following settings should be preserved:

- Model and tokenizer revisions
- Dataset subsets and preprocessing
- Retrieval configuration
- Few-shot demonstrations
- Generation hyperparameters
- Random seeds
- NLI evaluator and evaluation settings

Some checkpoints and experimental artifacts may require separate configuration or access.

---

## Limitations

Several limitations should be considered when interpreting the findings.

**Model comparability:** The evaluated models differ in parameter count and pretraining data, preventing strong causal conclusions about architecture.

**NLI evaluation:** NLI-based entailment is an imperfect measure of source faithfulness and may be affected by generation length or incomplete source coverage.

**Manual evaluation:** Human annotations cover a limited sample and do not exhaustively quantify omissions of relevant table information.

**Generalizability:** The findings are specific to the evaluated models, checkpoints, and ToTTo experimental setting.

These limitations motivate further work on controlled architectural comparisons and more comprehensive grounding evaluation.

---

## Citation

If you use this repository or build upon our research, please cite:

```bibtex
@misc{merimi2026diagnostic,
  title = {A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation},
  author = {Merimi, Zaynab and El Marrakchi, Noussayba and Satouf, Arthur},
  year = {2026},
  eprint = {2610.29602},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2610.29602}
}
```

**Preprint:** https://arxiv.org/abs/2610.29602

---

## Authors

**Zaynab Merimi**  
CY Tech, CY Cergy Paris Université, France

**Noussayba El Marrakchi**  
CY Tech, CY Cergy Paris Université, France

**Arthur Satouf**  
Mila – Quebec AI Institute  
Université Paris-Saclay / CentraleSupélec

*Zaynab Merimi and Noussayba El Marrakchi contributed equally.*

---

## License

No license was identified in the repository root when this README was prepared.

Public availability does not automatically grant permission to redistribute or modify the code.

Please contact the authors regarding reuse permissions unless a license is added.

---

## Project Status

**Research preprint and accompanying experimental code.**

This repository documents the experiments associated with the paper. Additional documentation, reproducibility improvements, and experimental extensions may be released in future updates.
