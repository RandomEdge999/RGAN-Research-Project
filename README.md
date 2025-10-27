# RGAN: Noise-Resilient Time-Series Forecasting with an LSTM Regression-GAN

This package implements a **Regression-GAN (R-GAN)** for time-series forecasting using **LSTMs** in both the generator and the discriminator,
**faithfully following the provided R code and draft paper**, and **meeting the professor’s plotting requirements**:

- **Y-axis**: Accuracy or Error (we report **RMSE** for regression).
- **X-axis**: Epochs (neural models) or **Number of Samples** (classical baselines).
- **Single model plots**: Train and Test errors **on the same plot** (per model).
- **Multiple model comparison**: Two separate plots — **Test error by model** and **Train error by model**.

The pipeline includes: robust data loading, automatic target detection, resampling and imputation, standardization, sliding-window creation,
R-GAN training (with label smoothing, gradient clipping, early stopping), supervised LSTM baseline, classical baselines (Naive, ETS, ARIMA),
noise-robustness tests, and a LaTeX paper builder that embeds all figures and results.

## Quick Start

### Windows (PyCharm/PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run (no tuner; quick start)
python run_experiment.py --csv ".\path\to\YourData.csv" --target auto --time_col auto --L 24 --H 12 --epochs 20 --batch_size 64 --tune false --results_dir ".\results"

# Build paper
python papers\build_paper.py --metrics .\results\metrics.json --out .\results\research_paper.tex
```

### macOS / Linux (bash / zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (no tuner; quick start)
python run_experiment.py --csv "./path/to/YourData.csv" --target auto --time_col auto --L 24 --H 12 --epochs 20 --batch_size 64 --tune false --results_dir "./results"

# Build paper
python papers/build_paper.py --metrics ./results/metrics.json --out ./results/research_paper.tex
```

## Works with Any CSV

- **Target selection**: `--target auto` picks `index_value` if present, otherwise the **last numeric column**.
  You can override: `--target price`.
- **Time column**: `--time_col auto` tries common names (`date`, `datetime`, `time`, `timestamp`). If not found, the order of rows is used.
  You can specify: `--time_col calc_time`.
- **Resampling**: `--resample ""` (none) by default. Optionally `--resample D` (day), `--resample H` (hour), with `--agg last|mean`.
- **Covariates**: All numeric columns besides the target become optional covariates for the generator input (target forecast remains univariate).

## Outputs
- `results/metrics.json` — all metrics, histories, and file paths to figures
- Curves (per professor):
  - `rgan_train_test_rmse_vs_epochs.png`
  - `lstm_train_test_rmse_vs_epochs.png`
  - `models_test_error.png` and `models_train_error.png`
  - `classical_error_vs_samples.png`
- `tuning_results.csv` (if tuning enabled)
- `research_paper.tex` — camera-ready LaTeX with figures embedded

See **REPRODUCIBILITY.md** and **ETHICS.md** for best practices.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**  
  Add these lines at the very top of `run_experiment.py`:
  ```python
  import sys, pathlib
  sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
  ```

- **Long pause before epochs appear**  
  The first progress print happens after epoch 1. For immediate feedback, reduce `--L/--H`, increase `--batch_size`, and set `--tune false`.
