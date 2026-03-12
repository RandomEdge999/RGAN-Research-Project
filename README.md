# RGAN — Noise-Resilient Time-Series Forecasting

LSTM-based Generative Adversarial Network for time-series forecasting, with a focus on robustness under real-world noise. Active research project working toward publication.

---

## What This Is

RGAN trains an LSTM generator and LSTM discriminator adversarially to forecast time-series data. The key research question: does the GAN training signal make forecasts more robust to noise than supervised baselines?

**Models included:**

| Model | Type |
|-------|------|
| RGAN, RGAN-WGAN, RGAN-WGAN-GP | GAN variants (ours) |
| LSTM Supervised | Neural baseline |
| DLinear, NLinear | Linear baselines |
| FITS | Frequency-domain baseline |
| PatchTST, iTransformer | Transformer baselines |
| TimeGAN | GAN competitor |
| Naive, ARIMA, ARMA, Tree Ensemble | Classical baselines |

---

## Setup

```bash
pip install -e .
```

For AWS/SageMaker cloud training:
```bash
pip install -e ".[cloud]"
```

---

## Running Experiments

```bash
# Train on Binance data (all models, 80 epochs)
rgan-train --csv data/binance/Binance_Data.csv --target index_value --epochs 80

# With checkpointing (for resumable runs)
rgan-train --csv data/binance/Binance_Data.csv --target index_value --epochs 80 \
  --checkpoint_dir checkpoints/ --checkpoint_every 10

# Resume from checkpoint
rgan-train --csv data/binance/Binance_Data.csv --target index_value --epochs 80 \
  --checkpoint_dir checkpoints/ --resume_from checkpoints/checkpoint_latest.pt

# Noise robustness sweep
rgan-train --csv data/binance/Binance_Data.csv --target index_value \
  --noise_levels 0,0.01,0.05,0.1,0.2

# Data augmentation experiment
rgan-augment --csv data/binance/Binance_Data.csv

# TSLib benchmarks
rgan-benchmark --datasets ETTh1 --pred_lens 96,192,336,720
```

Alternatively via `python -m`:
```bash
python -m rgan.scripts.run_training --csv data/binance/Binance_Data.csv --target index_value --epochs 80
```

---

## Key CLI Flags

| Flag | Purpose |
|------|---------|
| `--csv` | Path to input CSV |
| `--target` | Target column name |
| `--epochs` | Number of training epochs |
| `--gan_variant` | `standard`, `wgan`, or `wgan-gp` |
| `--L`, `--H` | Input window length and forecast horizon |
| `--units_g`, `--units_d` | Generator/discriminator hidden size |
| `--noise_levels` | Comma-separated σ values for robustness testing |
| `--checkpoint_dir` | Directory to save checkpoints |
| `--checkpoint_every` | Save checkpoint every N epochs |
| `--resume_from` | Path to checkpoint file to resume from |
| `--skip_classical` | Skip ARIMA/ARMA/tree baselines |
| `--results_dir` | Output directory (default: `./results`) |

---

## Datasets

Real-world noisy datasets used for evaluation:

| Dataset | Rows | Target | Noise Type |
|---------|------|--------|------------|
| Binance crypto | 86K | `index_value` | Market microstructure noise |
| Household Power | 2M | `Global_active_power` | Appliance load spikes |
| Beijing Air Quality | 420K | `PM2.5` | Outdoor sensor drift |
| MetroPT-3 Air Compressor | 1.5M | `TP2` | Mechanical wear noise |
| Gas Sensor Home | 929K | `CO_sensor` | Chemical sensor drift |
| Micro Gas Turbine | 71K | `el_power` | Mechanical + electrical noise |

Data files are not tracked in git. See `data/*/README.md` for download instructions.

---

## Outputs

Results are saved to `--results_dir` (default: `results/`):

- `metrics.json` — all metrics, architecture summaries, artifact paths
- `noise_robustness_table.csv` — RMSE per model per noise level
- `augmentation_results.csv` — downstream forecasting with synthetic data
- PNG/HTML plots for all figures

---

## Cloud Training (AWS SageMaker)

See `cloud/` for SageMaker infrastructure. Requires `cloud/config.yaml` (not tracked — contains AWS credentials).

```bash
python -m cloud.launch --csv data/binance/Binance_Data.csv --epochs 80
```
