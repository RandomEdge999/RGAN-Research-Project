# RGAN Research Project

Noise-resilient time-series forecasting with an LSTM Regression-GAN (R-GAN) and a fully instrumented reporting toolchain. This repository reproduces the instructor's original prototype while extending it with modern PyTorch training, reproducibility controls, a publication-ready paper builder, and a React + Plotly executive dashboard.

---

## Feature Highlights

- **PyTorch Regression-GAN pipeline** with deterministic seeding, AMP, early stopping, gradient clipping, and configurable generator/discriminator stacks.
- **Baseline coverage** for supervised LSTM, naïve persistence, and Gaussian naïve Bayes, plus optional ETS/ARIMA classical references.
- **Quantitative diagnostics** including RMSE/MSE/MAE/Bias with bootstrap confidence intervals in both scaled and original units, and automated noise-robustness sweeps.
- **Learning-curve analysis** across increasing training-set sizes to quantify sample efficiency.
- **Interactive dashboard** (`dashboard/`) built with React and Plotly that ingests `results/metrics.json` to deliver executive-friendly storytelling.
- **Camera-ready LaTeX builder** (`papers/build_paper.py`) that injects every figure, metric table, and architecture summary into the manuscript template.

---

## Quick Start

### 0. Prerequisites
- Python 3.10+ (project tested with Python 3.11.4)
- pip
- (Optional) CUDA-capable GPU + PyTorch build with CUDA for faster training

### 1. Create and activate a virtual environment

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
```
</details>

<details>
<summary><strong>macOS / Linux (bash or zsh)</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```
</details>

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** If you need a specific PyTorch build (e.g., with CUDA 12.1), install it before running the command above, then rerun `pip install -r requirements.txt` to pick up the remaining packages.

### 3. Run an experiment

#### Fast CPU-friendly demo (recommended on laptops)
```powershell
python run_experiment.py ^
  --csv src/rgan/Binance_Data.csv ^
  --target index_value ^
  --time_col calc_time ^
  --results_dir results ^
  --epochs 5 ^
  --curve_steps 0 ^
  --num_workers 0 ^
  --persistent_workers false ^
  --pin_memory false
```
*(Replace `^` with `\` on macOS/Linux.)*

This configuration keeps training short, prevents PyTorch's DataLoader from forking extra processes, and still produces a valid `results/metrics.json` plus all dashboard artifacts.

#### Full experiment (longer, more thorough)
```bash
python run_experiment.py \
  --csv ./data/your_timeseries.csv \
  --target Close \
  --time_col Timestamp \
  --L 24 --H 12 \
  --epochs 80 --batch_size 64 \
  --curve_steps 4 --curve_epochs 40 \
  --tune --tune_eval_frac 0.1 \
  --results_dir ./results
```
Add GPU-specific flags such as `--amp false` if your accelerator lacks AMP support. Supply `--tune_csv path/to/validation.csv` to run the sweep on a different dataset.

### 4. Launch the interactive dashboard

1. Ensure the previous step produced `results/metrics.json` (and corresponding figures).
2. Serve the dashboard folder to avoid browser CORS restrictions:
  ```bash
  cd dashboard
  python serve.py --port 8000
  ```
  This binds explicitly to `127.0.0.1` and opens your default browser automatically. To skip the auto-launch, add `--no-browser`; to change the metrics file in the convenience link, pass `--metrics path/to/metrics.json`.
3. Alternatively, use the standard module (`python -m http.server 8000 --bind 127.0.0.1`). Some terminals print `http://0.0.0.0:8000/`; replace `0.0.0.0` with `localhost` or `127.0.0.1` before clicking.
4. Browse to `http://localhost:8000/`. The app auto-loads `../results/metrics.json`. To point at a different metrics file, append `?metrics=/absolute/or/relative/path.json` to the URL.

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
| `--amp`, `--eval_batch_size`, `--num_workers`, `--persistent_workers`, `--pin_memory` | PyTorch runtime tuning. Set `--num_workers 0 --persistent_workers false --pin_memory false` for constrained CPU environments. |
| `--tune`, `--tune_csv`, `--tune_eval_frac` | Enable and configure the hyperparameter sweep. |
| `--curve_steps`, `--curve_min_frac`, `--curve_epochs`, `--curve_repeats` | Generate learning curves over increasing sample sizes. |
| `--noise_levels` | Comma-separated Gaussian noise standard deviations for robustness testing (`0` automatically included). |
| `--results_dir` | Destination for artifacts (defaults to `./results`). |

All flags are documented in `run_experiment.py` (`python run_experiment.py --help`).

---

## Interactive Dashboard (`dashboard/`)

- **Stack:** React 18 via CDN, Plotly 2.27+, modern CSS with glassmorphism accents.
- **Data Source:** Defaults to `../results/metrics.json`. Provide `?metrics=path/to/metrics.json` to override.
- **Sections:**
  - Experiment overview (dataset, reproducibility metadata, environment packages)
  - Executive highlights (best performer, generalisation gap, noise resilience, classical baseline)
  - Model snapshot cards with train/test RMSE & MAE
  - Learning trajectories (epoch curves) and comparative bar charts
  - Learning curves with uncertainty bands
  - Noise robustness analysis
  - Full precision table with bootstrap confidence intervals
  - Configuration & tuning summaries, architecture inventories, and artifact download links
- **Metrics selector:** Use the form in the hero banner to point the dashboard at any `metrics.json`; the page refreshes without a reload. Query parameters (`?metrics=...`) still work for deep links.
- **Remote metrics helper:** `serve.py` proxies the requested JSON, so relative paths that escape `dashboard/` (e.g. `../results_wsl_run/metrics.json`) resolve automatically. For headless environments (WSL, servers), pass `--no-browser` to avoid `xdg-open` warnings.
- **Serving:** Any static file server works (`python -m http.server`, `npx serve`, nginx, etc.). Hosting behind GitHub Pages is possible by copying `results/metrics.json` into the published directory and updating the query parameter.

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
- `naive_bayes_train_test_rmse_vs_epochs.png`
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
