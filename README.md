# RGAN Research Project

Noise-resilient time-series forecasting with an LSTM Regression-GAN (R-GAN) and a fully instrumented reporting toolchain. This repository reproduces the instructor's original prototype while extending it with modern PyTorch training, reproducibility controls, a publication-ready paper builder, and a React + Plotly executive dashboard.

---

## Feature Highlights

- **PyTorch Regression-GAN pipeline** with deterministic seeding, AMP, early stopping, gradient clipping, and configurable generator/discriminator stacks.
- **Baseline coverage** for supervised LSTM, naive persistence, ARIMA, and ARMA, plus optional ETS/ARIMA classical references.
- **Quantitative diagnostics** including RMSE/MSE/MAE/Bias with bootstrap confidence intervals in both scaled and original units, and automated noise-robustness sweeps.
- **Learning-curve analysis** across increasing training-set sizes to quantify sample efficiency.
- **RGAN Analytics Terminal** (`web_dashboard/`) built with React + Vite + Recharts that ingests `results/metrics.json` to deliver executive-friendly storytelling with a professional trading terminal aesthetic.
- **Camera-ready LaTeX builder** (`papers/build_paper.py`) that injects every figure, metric table, and architecture summary into the manuscript template.

---

## Quick Start

### 0. Prerequisites
- Python 3.10+ (project tested with Python 3.11.4)
- pip
- (Optional) CUDA-capable GPU + PyTorch build with CUDA for faster training
- Node.js and npm (for the dashboard)

### 1. Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash or zsh)**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** If you need a specific PyTorch build (e.g., with CUDA 12.1), install it before running the command above, then rerun `pip install -r requirements.txt` to pick up the remaining packages.

### 3. Run an experiment

Run the training script to generate results. This will create a `metrics.json` file in your specified results directory.

```bash
python run_experiment.py \
  --csv src/rgan/Binance_Data.csv \
  --target index_value \
  --time_col calc_time \
  --results_dir results_auto \
  --epochs 50 \
  --gan_variant wgan-gp \
  --use_logits True \
  --d_activation linear
```

### 4. View Results in Dashboard

The dashboard is a standalone application. You can run it once and use it to view results from any experiment by uploading the generated `metrics.json` file.

1. Navigate to the dashboard directory:
   ```bash
   cd web_dashboard
   ```
2. Install dependencies (first time only):
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open the URL shown in the terminal (usually `http://localhost:5173`).
5. Drag and drop the `metrics.json` file from your results directory (e.g., `results_auto/metrics.json`) into the dashboard.

### 5. Build the LaTeX paper

```bash
python papers/build_paper.py \
  --metrics results/metrics.json \
  --out results/research_paper.tex
```
Compile the resulting `.tex` file with your LaTeX toolchain (e.g., `pdflatex` or `latexmk`).

---

## CLI Reference

| Flag | Purpose |
| ---- | ------- |
| `--csv` | Path to the input CSV. Required. |
| `--target`, `--time_col` | Column names (use `auto` to infer). |
| `--L`, `--H` | Input window length and forecast horizon. |
| `--epochs`, `--batch_size` | Training schedule knobs for both GAN and LSTM. |
| `--units_g`, `--units_d`, `--g_layers`, `--d_layers` | Generator/discriminator architecture. |
| `--lambda_reg`, `--dropout`, `--label_smooth`, `--grad_clip` | Regularisation controls. |
| `--amp`, `--eval_batch_size`, `--num_workers`, `--persistent_workers`, `--pin_memory` | PyTorch runtime tuning. |
| `--tune`, `--tune_csv`, `--tune_eval_frac` | Enable and configure the hyperparameter sweep. |
| `--curve_steps`, `--curve_min_frac`, `--curve_epochs`, `--curve_repeats` | Generate learning curves over increasing sample sizes. |
| `--noise_levels` | Comma-separated Gaussian noise standard deviations for robustness testing (`0` automatically included). |
| `--results_dir` | Destination for artifacts (defaults to `./results`). |

All flags are documented in `run_experiment.py` (`python run_experiment.py --help`).

---

## RGAN Analytics Terminal (`web_dashboard/`)

- **Stack:** React 18, Vite, Recharts, Vanilla CSS (Premium "Trading Terminal" Theme).
- **Data Source:** User-uploaded `metrics.json`.
- **Features:**
  - **Professional Aesthetic:** High-density, dark-mode interface designed for financial analysis.
  - **Precision Data:** All metrics displayed with 8 decimal places.
  - **Comprehensive Analysis:**
    - **Performance Summary:** Key KPIs and improvement metrics.
    - **Detailed Metrics Table:** RMSE, MAE, sMAPE, MASE for all models (RGAN, LSTM, ARIMA, ARMA, Naive).
    - **Confidence Intervals:** 95% CI bounds for robust statistical comparison.
    - **Interactive Charts:** Zoomable training dynamics and noise robustness curves.

---

## Outputs

Running `run_experiment.py` populates the chosen `--results_dir` with:

- `metrics.json` – master record of dataset metadata, evaluation metrics (train/test/noise), learning-curve values, architecture summaries, tuning results, and artifact paths.
- Plot artifacts in both static (PNG) and interactive HTML formats for every figure referenced in the manuscript and dashboard.
- `tuning_results.csv` – only when `--tune` is supplied; contains per-trial metrics and seeds for reproducibility.
- Serialized models/checkpoints emitted by the training routines (see `results/` subdirectories).
- Optional LaTeX manuscript generated via `papers/build_paper.py` (`research_paper.tex`).

Key static figures (PNG):
- `rgan_train_test_rmse_vs_epochs.png`
- `lstm_train_test_rmse_vs_epochs.png`
- `naive_train_test_rmse_vs_epochs.png`
- `arima_train_test_rmse_vs_epochs.png`
- `arma_train_test_rmse_vs_epochs.png`
- `models_test_error.png` and `models_train_error.png`
- `naive_baseline_vs_naive_bayes.png`
- `classical_error_vs_samples.png` (when ETS/ARIMA available)
- `ml_error_vs_samples.png` (learning curves)

Interactive counterparts share the same filenames suffixed by `_interactive.html` when Plotly export succeeds.

---

## Troubleshooting & Tips

- **Rich styling errors:** The console now degrades gracefully if `rich` lacks gradient support, but ensure `rich>=13` for the full experience (`pip install --upgrade rich`).
- **Slow CPU training:** Reduce `--epochs`, disable learning curves (`--curve_steps 0`), and set DataLoader flags for single-process loading (`--num_workers 0 --persistent_workers false --pin_memory false`).
- **GPU not detected:** Install the appropriate PyTorch wheel for your CUDA version (see https://pytorch.org/get-started/locally/). Reinstall requirements afterwards.
- **Missing classical plots:** Install `statsmodels` (`pip install statsmodels`) to enable ETS/ARIMA baselines.
- **No metrics generated:** Ensure the run completed without interruption. Partial runs may leave stale PNGs but no `metrics.json`.
- **Dashboard shows blank sections:** Verify the metrics file path in the browser console and confirm CORS is not blocking the fetch (serve the dashboard via HTTP rather than opening `index.html` directly from disk).

For deeper customisation (new baselines, alternative logging, etc.), browse the modules under `src/rgan/` and `dashboard/` for inline documentation.
