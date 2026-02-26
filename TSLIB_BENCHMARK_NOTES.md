# TSLib Benchmark Integration Notes

## Goal
Run our models (RGAN, LSTM, Naive, ARIMA, Tree Ensemble) on standard time-series benchmark datasets used by the research community so we can compare our results against published numbers from papers like PatchTST, iTransformer, DLinear, TimesNet etc.

## Approach: Option A - Their Datasets, Our Models
We chose to evaluate our models on standard TSLib benchmark datasets rather than running their models on our data. This makes our results directly comparable to published tables in the literature.

## Datasets Selected
We picked the "Core 4" datasets commonly reported in papers:

- **ETTh1** - Electricity Transformer Temperature, hourly, 17420 rows, 7 features, target=OT. Split: 8640/2880/2880 (train/val/test, fixed row counts = 12/4/4 months)
- **ETTh2** - Same format, different transformer. Same split.
- **Weather** - 52696 rows, 21 meteorological indicators, 10-min frequency. Split: 70/10/20 ratio.
- **Exchange** - 7588 rows, 8 currency exchange rates, daily. Split: 70/10/20 ratio.

Started with ETTh1 and ETTh2 first before running all 4.

## Data Source
Datasets downloaded from HuggingFace (`thuml/Time-Series-Library`) using the `datasets` library. Initially tried direct URL download but the HF repo only has parquet files, not raw CSVs. So we use `datasets.load_dataset()` and convert to CSV, cached locally in `data/tslib/`.

## Pipeline Setup

### New files created:
1. `src/rgan/tslib_data.py` - Dataset download, loading, and TSLib-standard train/val/test splitting with z-score normalization
2. `run_tslib_benchmark.py` - Main benchmark runner that loops over datasets x prediction horizons

### How it works:
- Downloads dataset CSV from HuggingFace (cached after first download)
- Applies TSLib-standard splits (ETT uses fixed row counts, Weather/Exchange use ratio-based)
- Z-score normalizes using train set statistics only (same as TSLib does)
- Creates sliding windows using our existing `make_windows_univariate()` - univariate mode, predict target (OT) from its own history
- Trains each model on train windows, evaluates on test windows
- Reports MSE and MAE on standardized scale (this is what TSLib papers report, NOT original scale)
- Standard settings: seq_len=96, pred_len in {96, 192, 336, 720}

### Output:
- `results/tslib_benchmark/benchmark_results.csv` - long format
- `results/tslib_benchmark/benchmark_pivot.csv` - paper format pivot table
- `results/tslib_benchmark/benchmark_results.tex` - LaTeX table
- Per-run `metrics.json` files in subdirectories

## RGAN Training Instability on TSLib Data

### The Problem
RGAN was originally tuned for Binance crypto data with L=24, H=12 (small windows). On TSLib data with L=96 the GAN training collapsed immediately:
- Discriminator saturated to D=100.0 (perfectly distinguishing real from fake)
- Generator loss went to NaN
- Early stopping triggered at epoch 10 with no valid predictions

The Binance config used: lambda_reg=0.1, lr_g=1e-3, lr_d=1e-3, no grad clipping, L=24, H=12.

### Why It Failed
With longer sequences (L=96), gradients are larger and the discriminator overpowers the generator almost immediately. The low lambda_reg=0.1 meant the supervised signal was too weak to keep the generator stable. Once D saturates, the adversarial gradient for G becomes uninformative (vanishing gradients through saturated sigmoid), and the generator diverges.

### Fix Attempts

**Attempt 1 - Conservative hyperparameters (still failed):**
- lambda_reg=1.0, lr_d halved to 5e-4, grad_clip=1.0, dropout=0.1, amp=False
- Result: still NaN. The discriminator was too fast even with these changes.

**Attempt 2 - Supervised warmup + reduced adversarial weight (worked):**
- lambda_reg=5.0 (50x the original - heavily favor supervised loss)
- adv_weight=0.1 (adversarial loss contributes only 10%)
- supervised_warmup_epochs=5 (train generator on pure MSE for 5 epochs before introducing discriminator)
- grad_clip=1.0, lr_d=5e-4, dropout=0.1
- Result: RGAN trains successfully. MSE=0.160 on ETTh1 pred_len=96 (comparable to LSTM's 0.166)

The key insight: the supervised warmup gives the generator a reasonable initialization before the discriminator is introduced. Without it, the generator starts from random weights and the discriminator can trivially distinguish real from fake, causing immediate mode collapse.

### Implication for the Paper
This is actually an interesting finding worth discussing: RGAN's GAN component requires careful tuning per-dataset, and the supervised warmup is critical for stability on longer sequence lengths. The adversarial training acts more as a regularizer when lambda_reg >> adv_weight, which is essentially what makes it work on these benchmarks.

## Final RGAN Config for TSLib Benchmarks
```python
TrainConfig(
    lambda_reg=5.0,          # strong supervised signal
    adv_weight=0.1,          # weak adversarial signal
    grad_clip=1.0,           # prevent gradient explosion
    supervised_warmup_epochs=5,  # pure MSE before GAN kicks in
    lr_g=1e-3, lr_d=5e-4,   # slower discriminator
    dropout=0.1,             # regularization
    amp=False,               # no mixed precision on CPU
    units_g=64, units_d=64,  # same architecture as Binance
    num_layers=1,            # single LSTM layer
)
```

## Running the Benchmark
```bash
# 2 datasets first
/opt/anaconda3/bin/python3 run_tslib_benchmark.py --datasets ETTh1,ETTh2 --epochs 80

# all 4 datasets
/opt/anaconda3/bin/python3 run_tslib_benchmark.py --epochs 80

# quick test
/opt/anaconda3/bin/python3 run_tslib_benchmark.py --datasets ETTh1 --pred_lens 96 --epochs 10 --skip_classical
```

Note: using conda python (`/opt/anaconda3/bin/python3`) because it has torch and the `datasets` library. Homebrew python3.14 doesn't have these packages.

## Repo Cleanup (done before benchmark work)
- Consolidated all results into single `results/` directory
- Kept only latest results: `results/binance_rgan/`, `results/binance_augmentation/`, `results/m5_combined/`
- Deleted old runs: `results_auto/`, `results_torch_80ep/`, `results_wsl_run/`, `results_supervisor_tables/`, `demo_charts/`
- The 80-epoch Binance run was in `results_torch_80ep/` which got deleted (was gitignored, not recoverable). The kept `results/binance_rgan/` has a 40-epoch run. Need to re-run 80 epochs later.
- Updated script defaults: `run_experiment.py` defaults to `results/experiment`, `run_augmentation_experiment.py` defaults to `results/augmentation`
- Updated `.gitignore`: `results_*/` changed to `results/`, added `data/`
