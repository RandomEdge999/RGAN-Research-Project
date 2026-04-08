#!/usr/bin/env python3
"""Aggregate 15 training runs (3 datasets × 5 seeds) into paper-quality tables and charts.

Steps 2-3 of the paper pack plan:
  - Step 2: Clean forecasting mean ± std tables + boxplots + ranking stability
  - Step 3: Robustness mean ± std across noise levels + degradation table
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add src to path so we can import rgan.plots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rgan.plots import (
    _NOISE_KEY_MAP,
    plot_seed_boxplots,
    create_mean_std_table,
    plot_ranking_stability,
    plot_noise_robustness_multi_dataset,
    plot_clean_vs_noisy_rankings,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "cloud"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "paper_pack"

DATASET_RUNS = {
    "Binance BTCBVOL": [
        f"rgan-binance-s{s}-eval4" for s in range(42, 47)
    ],
    "NASA POWER Denver": [
        f"rgan-nasa-s{s}-e100-v2" for s in range(42, 47)
    ],
    "Wind Turbine SCADA": [
        f"rgan-wind-s{s}-e100-v2" for s in range(42, 47)
    ],
}

# Display name mapping (metrics.json key → paper name)
KEY_TO_NAME = dict(_NOISE_KEY_MAP)

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]


def load_metrics(run_dir: str) -> dict:
    path = RESULTS_ROOT / run_dir / "results" / "metrics.json"
    with open(path) as f:
        return json.load(f)


def extract_clean_metrics(metrics: dict) -> dict:
    """Extract sd=0.0 metrics (clean data) from noise_robustness array."""
    for entry in metrics["noise_robustness"]:
        if abs(entry["sd"]) < 1e-9:
            return entry
    raise ValueError("No sd=0.0 entry found")


# ---------------------------------------------------------------------------
# Step 2: Clean Forecasting Aggregation
# ---------------------------------------------------------------------------
def aggregate_clean_forecasting():
    """Aggregate RMSE and MAE across seeds for each dataset."""
    print("=" * 70)
    print("STEP 2: Aggregate Clean Forecasting Results")
    print("=" * 70)

    # seed_metrics: {dataset: {model_display_name: [rmse_per_seed]}}
    rmse_by_seed = {}
    mae_by_seed = {}

    for ds_name, runs in DATASET_RUNS.items():
        rmse_by_seed[ds_name] = defaultdict(list)
        mae_by_seed[ds_name] = defaultdict(list)

        for run_dir in runs:
            metrics = load_metrics(run_dir)
            clean = extract_clean_metrics(metrics)

            for key, display_name in KEY_TO_NAME.items():
                stats = clean.get(key, {})
                if not stats:
                    continue
                rmse_val = stats.get("rmse_orig")
                mae_val = stats.get("mae_orig")
                if rmse_val is not None:
                    rmse_by_seed[ds_name][display_name].append(rmse_val)
                if mae_val is not None:
                    mae_by_seed[ds_name][display_name].append(mae_val)

        # Print per-dataset summary
        print(f"\n--- {ds_name} ({len(runs)} seeds) ---")
        print(f"{'Model':<18} {'RMSE mean':>12} {'± std':>12}   {'MAE mean':>12} {'± std':>12}")
        print("-" * 70)
        # Sort by mean RMSE
        models_sorted = sorted(
            rmse_by_seed[ds_name].keys(),
            key=lambda m: np.mean(rmse_by_seed[ds_name][m])
        )
        for i, model in enumerate(models_sorted, 1):
            r = rmse_by_seed[ds_name][model]
            m_vals = mae_by_seed[ds_name][model]
            marker = " <-- RGAN" if model == "RGAN" else ""
            print(f"  {i:2d}. {model:<14} {np.mean(r):12.6f} {np.std(r, ddof=1):12.6f}   "
                  f"{np.mean(m_vals):12.6f} {np.std(m_vals, ddof=1):12.6f}{marker}")

    # Convert defaultdicts to regular dicts for plotting functions
    rmse_seed = {ds: dict(v) for ds, v in rmse_by_seed.items()}
    mae_seed = {ds: dict(v) for ds, v in mae_by_seed.items()}

    # --- Produce outputs ---
    out = OUTPUT_DIR / "step2_clean_forecasting"
    out.mkdir(parents=True, exist_ok=True)

    # 1) Mean ± std table (RMSE)
    print("\n\n--- RMSE Mean ± Std Table ---")
    table = create_mean_std_table(rmse_seed, "RMSE", str(out / "rmse_mean_std.csv"))
    print(table)

    # 2) Mean ± std table (MAE)
    print("\n\n--- MAE Mean ± Std Table ---")
    table_mae = create_mean_std_table(mae_seed, "MAE", str(out / "mae_mean_std.csv"))
    print(table_mae)

    # 3) Seed variance boxplot (RMSE)
    bp_path = plot_seed_boxplots(rmse_seed, "RMSE (original scale)", str(out / "seed_boxplots_rmse.png"))
    print(f"\nSaved boxplot: {bp_path}")

    # 4) Ranking stability bump chart
    rs_path = plot_ranking_stability(rmse_seed, str(out / "ranking_stability.png"))
    print(f"Saved ranking stability: {rs_path}")

    return rmse_seed, mae_seed


# ---------------------------------------------------------------------------
# Step 3: Robustness Aggregation
# ---------------------------------------------------------------------------
def aggregate_robustness():
    """Aggregate noise robustness across seeds for each dataset."""
    print("\n\n" + "=" * 70)
    print("STEP 3: Aggregate Robustness Results")
    print("=" * 70)

    # For multi-dataset robustness plot: average across seeds per noise level
    # all_noise_avg: {dataset: [{sd, model1: {rmse_orig: mean}, ...}]}
    all_noise_avg = {}

    # For degradation table: {dataset: {model: {sd: [rmse_per_seed]}}}
    degradation_data = {}

    for ds_name, runs in DATASET_RUNS.items():
        # Collect: {model_key: {sd: [rmse_values_across_seeds]}}
        model_sd_rmse = defaultdict(lambda: defaultdict(list))

        for run_dir in runs:
            metrics = load_metrics(run_dir)
            for entry in metrics["noise_robustness"]:
                sd = entry["sd"]
                for key in KEY_TO_NAME:
                    stats = entry.get(key, {})
                    if stats:
                        rmse = stats.get("rmse_orig", stats.get("rmse"))
                        if rmse is not None:
                            model_sd_rmse[key][sd].append(rmse)

        # Build averaged noise results for plotting
        averaged_entries = []
        for sd in NOISE_LEVELS:
            entry = {"sd": sd}
            for key, display_name in KEY_TO_NAME.items():
                vals = model_sd_rmse[key].get(sd, [])
                if vals:
                    entry[key] = {
                        "rmse_orig": np.mean(vals),
                        "rmse_orig_std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                    }
            averaged_entries.append(entry)
        all_noise_avg[ds_name] = averaged_entries

        # Print degradation table
        print(f"\n--- {ds_name}: Robustness Degradation ---")
        print(f"{'Model':<18}", end="")
        for sd in NOISE_LEVELS:
            print(f"  sd={sd:<5}", end="")
        print(f"  {'Δ(0→0.2)':>10}  {'Δ%':>8}")
        print("-" * 100)

        models_sorted = sorted(
            model_sd_rmse.keys(),
            key=lambda k: np.mean(model_sd_rmse[k].get(0.0, [float("inf")]))
        )
        for key in models_sorted:
            display = KEY_TO_NAME.get(key, key)
            print(f"  {display:<16}", end="")
            clean_rmse = np.mean(model_sd_rmse[key].get(0.0, [float("nan")]))
            noisy_rmse = np.mean(model_sd_rmse[key].get(0.2, [float("nan")]))
            for sd in NOISE_LEVELS:
                vals = model_sd_rmse[key].get(sd, [])
                if vals:
                    print(f"  {np.mean(vals):.6f}", end="")
                else:
                    print(f"  {'N/A':>8}", end="")
            delta = noisy_rmse - clean_rmse
            pct = (delta / clean_rmse * 100) if clean_rmse > 0 else float("nan")
            print(f"  {delta:>+10.6f}  {pct:>+7.1f}%")

        degradation_data[ds_name] = model_sd_rmse

    # --- Produce outputs ---
    out = OUTPUT_DIR / "step3_robustness"
    out.mkdir(parents=True, exist_ok=True)

    # 1) Multi-dataset robustness line plot (averaged across seeds)
    rob_path = plot_noise_robustness_multi_dataset(all_noise_avg, str(out / "robustness_multi_dataset.png"))
    print(f"\nSaved robustness plot: {rob_path}")

    # 2) Clean vs noisy rankings
    # For this we need per-seed data averaged, pass the averaged entries
    cvn_path = plot_clean_vs_noisy_rankings(all_noise_avg, str(out / "clean_vs_noisy_rankings.png"))
    print(f"Saved clean vs noisy: {cvn_path}")

    # 3) Save degradation data as CSV
    _save_degradation_csv(degradation_data, out / "degradation_table.csv")
    print(f"Saved degradation CSV: {out / 'degradation_table.csv'}")

    return all_noise_avg, degradation_data


def _save_degradation_csv(degradation_data: dict, path: Path):
    """Save cross-dataset degradation summary as CSV."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "RMSE_sd0.0", "RMSE_sd0.2", "Delta", "Delta_pct"])
        for ds_name, model_sd_rmse in degradation_data.items():
            for key in sorted(model_sd_rmse.keys()):
                display = KEY_TO_NAME.get(key, key)
                clean = np.mean(model_sd_rmse[key].get(0.0, [float("nan")]))
                noisy = np.mean(model_sd_rmse[key].get(0.2, [float("nan")]))
                delta = noisy - clean
                pct = (delta / clean * 100) if clean > 0 else float("nan")
                writer.writerow([ds_name, display, f"{clean:.6f}", f"{noisy:.6f}",
                                 f"{delta:+.6f}", f"{pct:+.1f}%"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}")

    # Verify all runs exist
    missing = []
    for ds, runs in DATASET_RUNS.items():
        for r in runs:
            p = RESULTS_ROOT / r / "results" / "metrics.json"
            if not p.exists():
                missing.append(f"{ds}: {r}")
    if missing:
        print("MISSING RUNS:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    print("All 15 runs verified.\n")

    rmse_seed, mae_seed = aggregate_clean_forecasting()
    all_noise_avg, degradation_data = aggregate_robustness()

    print("\n\n" + "=" * 70)
    print("DONE — Steps 2 & 3 complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)
