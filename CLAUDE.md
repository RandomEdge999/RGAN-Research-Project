# CLAUDE.md — Project Context

## What This Project Is
RGAN (Regression-GAN): LSTM-based Generative Adversarial Network for noise-resilient time-series forecasting. This is an active research project working toward a publication.

## Important: document.md
There is a `document.md` file at the project root. It is a sequential research journal logging every important decision, rationale, and result. When making decisions that affect the research (model architecture, training strategy, dataset choices, evaluation methodology), **always append an entry to document.md** with the date and reasoning. Never remove or rewrite previous entries — it is append-only, like a lab notebook.

## Project Structure
```
src/rgan/              — All library code (models, training, data, metrics, plots)
src/rgan/scripts/      — CLI scripts (run_training, run_augmentation, build_paper, etc.)
data/                  — All datasets (binance/, m5/, tslib/) — gitignored
results/               — Training outputs — gitignored
papers/                — LaTeX manuscript template
docs/                  — Reference documentation
notebooks/             — Colab notebook
cloud/                 — AWS SageMaker infrastructure (future)
```

## Setup
```bash
pip install -e .          # installs rgan package + CLI entry points
pip install -e ".[cloud]" # also installs sagemaker + boto3
```

## How to Run
```bash
# Via entry points (after pip install -e .):
rgan-train --csv data/binance/Binance_Data.csv --target index_value --epochs 80
rgan-augment --csv data/binance/Binance_Data.csv
rgan-benchmark --datasets ETTh1 --pred_lens 96

# Or via python -m:
python -m rgan.scripts.run_training --csv data/binance/Binance_Data.csv --target index_value --epochs 80
```

## Key Technical Decisions
- PyTorch >= 2.2 with AMP support
- GAN variants: standard BCE, WGAN, WGAN-GP
- Checkpoint/resume support via --checkpoint_dir and --resume_from
- All data paths are CLI args, never hardcoded
- Metrics reported on standardized scale for TSLib benchmarks (matches literature)
- Bootstrap uncertainty estimation (300 samples default)

## AWS / Cloud
- SageMaker quota: 8 (us-east-1)
- EC2 quota: 8
- Cloud integration planned in cloud/ directory
- Checkpoint infrastructure already in place for spot instance recovery
