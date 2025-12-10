# RGAN Research Project

Noise-resilient time-series forecasting with an LSTM Regression-GAN (R-GAN) and a fully instrumented reporting toolchain. This repository reproduces the instructor's original prototype while extending it with modern PyTorch training, reproducibility controls, a publication-ready paper builder, and a React + Plotly executive dashboard.

---

## Feature Highlights

- **PyTorch Regression-GAN pipeline** with deterministic seeding, AMP, early stopping, gradient clipping, and configurable generator/discriminator stacks.
- **Baseline coverage** for supervised LSTM, naive persistence, ARIMA, and ARMA, plus optional ETS/ARIMA classical references.
- **Quantitative diagnostics** including RMSE/MSE/MAE/Bias with bootstrap confidence intervals in both scaled and original units, and automated noise-robustness sweeps.
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

**Want a lean smoke test?** You can keep the full pipeline available while avoiding the slowest extras:

```bash
python run_experiment.py \
  --csv src/rgan/Binance_Data.csv \
  --target index_value \
  --time_col calc_time \
  --results_dir results_quick \
  --epochs 5 \
  --gan_variant wgan-gp \
  --skip_classical \
  --noise_levels 0 \
  --bootstrap_samples 0
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
| `--gan_variant` | Adversarial loss style (`standard`, `wgan`, or `wgan-gp`). |
| `--d_steps`, `--g_steps` | Critic/Generator updates per batch (Wasserstein defaults to 5 D steps when unset). |
| `--wgan_clip_value`, `--wgan_gp_lambda` | Wasserstein regularisers (weight clipping vs. gradient penalty strength). |
| `--lambda_reg`, `--dropout`, `--label_smooth`, `--grad_clip` | Regularisation controls. |
| `--amp`, `--eval_batch_size`, `--num_workers`, `--persistent_workers`, `--pin_memory` | PyTorch runtime tuning. |
| `--prefetch_factor` | Background prefetch for multi-worker DataLoaders. |
| `--gpu_id` | ID of the GPU to use (default: 0). |
| `--require_cuda` | Strict mode: Fail if CUDA is not available or the specified GPU ID is invalid. |
| `--skip_classical` | Skip ARIMA/ARMA/tree baselines for faster smoke tests. |
| `--tune`, `--tune_csv`, `--tune_eval_frac` | Enable and configure the hyperparameter sweep. |
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

- `metrics.json` – master record of dataset metadata, evaluation metrics (train/test/noise), architecture summaries, tuning results, and artifact paths.
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

Interactive counterparts share the same filenames suffixed by `_interactive.html` when Plotly export succeeds.

---

## Troubleshooting & Tips

**Resource hotspots (optional features you can turn off when speed matters):**

- Classical baselines (ARIMA/ARMA/tree ensemble) can dominate startup on long series. Disable with `--skip_classical`.
- Hyperparameter sweeps via `--tune` launch multiple full trainings. Leave disabled unless exploring the search space.
 - Noise robustness (`--noise_levels` with multiple values) repeats evaluation per noise level. Use a single value (e.g., `0`).
 - Bootstrap confidence intervals (`--bootstrap_samples 300` by default) repeatedly resample metrics. Set `--bootstrap_samples 0` to skip.

- **Rich styling errors:** The console now degrades gracefully if `rich` lacks gradient support, but ensure `rich>=13` for the full experience (`pip install --upgrade rich`).
- **Slow CPU training:** Reduce `--epochs` and set DataLoader flags for single-process loading (`--num_workers 0 --persistent_workers false --pin_memory false`).
- **GPU not detected:** Install the appropriate PyTorch wheel for your CUDA version (see https://pytorch.org/get-started/locally/). Reinstall requirements afterwards.
- **Enforce GPU usage:** Pass `--require_cuda --gpu_id <id>` to fail fast if the requested GPU is unavailable instead of silently falling back to CPU.
- **Missing classical plots:** Install `statsmodels` (`pip install statsmodels`) to enable ETS/ARIMA baselines.
- **No metrics generated:** Ensure the run completed without interruption. Partial runs may leave stale PNGs but no `metrics.json`.
- **Dashboard shows blank sections:** Verify the metrics file path in the browser console and confirm CORS is not blocking the fetch (serve the dashboard via HTTP rather than opening `index.html` directly from disk).
- **Windows DataLoader hangs:** If training freezes on Windows, try adding `--num_workers 0 --persistent_workers false` to disable multi-process data loading, which can deadlock on Windows/WSL.

For deeper customisation (new baselines, alternative logging, etc.), browse the modules under `src/rgan/` and `dashboard/` for inline documentation.
