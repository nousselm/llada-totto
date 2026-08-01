# LLaDA-ToTTo-Diffusion

Implementation and evaluation of diffusion-based language models for the ToTTo table-to-text benchmark.

## Models
- LLaDA (Pretrained)
- LLaDA + QLoRA
- MDLM (Pretrained)

## Project Structure

```text
data/
manual_evaluation/
scripts/
README.md
requirements.txt
.gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Main Scripts

- train_llada_totto_qlora.py
- run_llada_totto_pretrained.py
- run_llada_totto_qlora.py
- run_mdlm_totto_pretrained.py
- evaluate_predictions.py

Generated outputs are written to `outputs/`, which is ignored by Git.