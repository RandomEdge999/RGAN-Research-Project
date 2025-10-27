# RGAN: Noise-Resilient Time-Series Forecasting with an LSTM Regression-GAN

This repository implements the end-to-end pipeline described in the accompanying research draft for **Regression-GAN (R-GAN)** time-series forecasting. The workflow faithfully mirrors the instructor's original R prototype while extending it with:

- **Generator/Discriminator inspection** – Experiment runs capture layer-by-layer descriptions and key hyperparameters.
- **Expanded metrics** – RMSE, MSE, MAE, and bias for R-GAN, an LSTM baseline, and a naïve persistence model.
- **Professor-required figures** – Matching plots for each single model, bar charts for model comparison, the original naïve baseline graphic, classical ETS/ARIMA curves, and learning curves as the sample size grows.
- **Noise robustness** – Automatic evaluation with Gaussian noise injected into the test windows.
- **Camera-ready paper builder** – Injects every figure, the R-GAN architecture, and an error table into the LaTeX template.

The tooling handles data loading, resampling, interpolation, feature standardisation, sliding-window generation (with optional covariates), GAN/LSTM training with early stopping, automatic tuning, and report generation.

---

## Quick Start

### 1. Create an environment and install dependencies

<details>
<summary><strong>Windows (PowerShell / PyCharm)</strong></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>macOS / Linux (bash / zsh)</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

### 2. Run an experiment

```bash
python run_experiment.py \
  --csv ./path/to/YourData.csv \
  --target auto --time_col auto \
  --L 24 --H 12 --epochs 40 --batch_size 64 \
  --tune true --results_dir ./results
```

Key command-line options:

| Flag | Description |
| ---- | ----------- |
| `--tune [true|false]` | Grid-search generator/discriminator units and λ. Use `--tune_csv` to tune on a different dataset. |
| `--g_layers`, `--d_layers` | Number of stacked LSTM layers in the generator/discriminator. |
| `--g_activation`, `--g_recurrent_activation`, `--g_dense_activation` | Generator LSTM/Dense activations. |
| `--d_activation`, `--d_recurrent_activation` | Discriminator LSTM activations. |
| `--curve_steps`, `--curve_min_frac`, `--curve_epochs` | Control learning-curve generation across increasing sample sizes. |
| `--train_ratio`, `--resample`, `--agg` | Temporal split and optional resampling strategy. |
| `--results_dir` | Output directory for metrics, models, and figures (default: `./results`). |

Learning-curve analysis (sample-size sweeps) is enabled by default. Disable it by setting `--curve_steps 0` if you need a minimal run.

### 3. Build the paper

```bash
python papers/build_paper.py --metrics ./results/metrics.json --out ./results/research_paper.tex
```

The generated TeX document embeds every figure (naïve baseline, neural models, bar charts, classical baselines, learning curves) and summarises the R-GAN architecture plus the RMSE/MSE/Bias table requested by the professor.

---

## How it works

1. **Data preparation** – `src/mit_rgan/data.py` automatically locates the target, converts recognised time columns, optionally resamples, fills gaps via interpolation, and standardises numeric fields (with covariates kept for the GAN input).
2. **Windowing** – Sliding windows are produced with or without covariates (`make_windows_univariate` / `make_windows_with_covariates`).
3. **Model training** –
   - `build_generator` / `build_discriminator` construct configurable LSTM stacks.
   - `train_rgan_keras` optimises the adversarial + regression objectives with label smoothing, gradient clipping, and validation-based early stopping.
   - `train_lstm_supervised` serves as a direct supervised baseline.
   - `naive_baseline` provides the persistence comparison and powers the naïve figure.
4. **Diagnostics** – `run_experiment.py` computes clean train/test statistics (RMSE, MSE, MAE, Bias) for every model, evaluates robustness on noisy inputs, plots comparison bar charts, generates learning curves, and records ETS/ARIMA baselines via `classical_curves_vs_samples` when `statsmodels` is available.
5. **Reporting** – `metrics.json` aggregates every metric, figure path, and architecture description so the LaTeX builder can inject them directly. `papers/template.tex` references these fields to create the final paper.

---

## Outputs

Running `run_experiment.py` populates the chosen `--results_dir` with:

- `metrics.json` – central log containing dataset metadata, train/test/noisy metrics, architecture summaries, learning-curve values, and classical baseline scores.
- `tuning_results.csv` – grid-search results (present when `--tune true`).
- Figures required by the professor:
  - `rgan_train_test_rmse_vs_epochs.png`
  - `lstm_train_test_rmse_vs_epochs.png`
  - `naive_train_test_rmse_vs_epochs.png`
  - `models_test_error.png` / `models_train_error.png`
  - `classical_error_vs_samples.png` (when ETS/ARIMA fits succeed)
  - `ml_error_vs_samples.png` (learning curves for R-GAN, LSTM, Naïve)
- `research_paper.tex` – complete LaTeX manuscript produced by `papers/build_paper.py`.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** – ensure you execute `run_experiment.py` from the repository root so that `src/` is importable (or prepend the repo path to `sys.path`).
- **Statsmodels not installed** – `classical_curves_vs_samples` silently skips ETS/ARIMA when `statsmodels` is unavailable. Install the dependency via `pip install statsmodels` to obtain the classical plots.
- **Long pause before epochs print** – the first log appears after epoch 1. Reduce `--epochs`, shrink `--L/--H`, or disable tuning/learning curves for faster iteration.
- **Insufficient data for windows** – ensure your CSV has at least `L + H` observations after cleaning; otherwise, adjust the window sizes.

For reproducibility and ethics considerations, refer to `papers/` and the notes embedded in the generated LaTeX document.
