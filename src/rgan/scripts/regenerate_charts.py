#!/usr/bin/env python
"""
Regenerate Charts from Saved Augmentation Results

Reads metrics_augmentation.json (produced by run_augmentation.py) and
regenerates all tables and visualizations without re-running any models.

Usage:
    # Single run
    python -m rgan.scripts.regenerate_charts --results_dir results/augmentation

    # Multi-dataset aggregation (cross-dataset charts)
    python -m rgan.scripts.regenerate_charts \
        --multi results/cloud/rgan-binance-s42/augmentation \
               results/cloud/rgan-nasa-s42/augmentation \
               results/cloud/rgan-wind-s42/augmentation \
        --dataset_names "Binance BTCBVOL" "NASA POWER Denver" "Wind Turbine SCADA"

    # Multi-seed aggregation (mean ± std table)
    python -m rgan.scripts.regenerate_charts \
        --multi_seeds results/cloud/rgan-binance-s42/augmentation \
                      results/cloud/rgan-binance-s43/augmentation \
                      results/cloud/rgan-binance-s44/augmentation \
                      results/cloud/rgan-binance-s45/augmentation \
                      results/cloud/rgan-binance-s46/augmentation \
        --dataset_name "Binance BTCBVOL"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_metrics(results_dir: Path) -> dict:
    """Load metrics_augmentation.json from a results directory."""
    metrics_path = results_dir / "metrics_augmentation.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics_augmentation.json in {results_dir}")
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def regenerate_single(results_dir: Path, output_dir: Optional[Path] = None):
    """Regenerate all charts/tables for a single augmentation run."""
    metrics = _load_metrics(results_dir)
    out = output_dir or results_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded metrics from: {results_dir}")
    print(f"  Dataset: {metrics.get('dataset', '?')}")
    print(f"  Seed: {metrics.get('seed', '?')}")
    print(f"  Models in augmentation: {list(metrics.get('data_augmentation', {}).keys())}")

    # --- Table 1: Supervisor Baseline Classification ---
    baseline = metrics.get("supervisor_baseline_classification", {})
    if baseline:
        rows = []
        for model, m in baseline.items():
            rows.append({
                "Model": model,
                "Accuracy": f"{m['Accuracy']*100:.2f}%",
                "F1": f"{m['F1']*100:.2f}%",
                "Precision": f"{m['Precision']*100:.2f}%",
                "Recall": f"{m['Recall']*100:.2f}%",
            })
        df = pd.DataFrame(rows)
        path = out / "supervisor_baseline_classification_table.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    # --- Table 2: Synthetic Quality (FD, Var diff, Discrimination Score) ---
    syn_quality = metrics.get("synthetic_quality", {})
    if syn_quality:
        rows = []
        for method, q in syn_quality.items():
            disc = q.get("all_discrimination", {}).get("Random Forest", {})
            rows.append({
                "Method": method,
                "FD": f"{q['frechet_distance']:.4f}",
                "Variance diff.": f"{q['variance_difference']['abs_diff']:.4f}",
                "Discrimination Score": f"{disc.get('accuracy', float('nan')):.4f}",
            })
        df = pd.DataFrame(rows)
        path = out / "supervisor_gan_quality_table.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    # --- Table 3: Augmentation Classification (full metrics per scenario) ---
    aug_acc = metrics.get("supervisor_augmentation_accuracy", {})
    if aug_acc:
        # Accuracy-only table (matches screenshot format)
        rows_acc = []
        # Full metrics table (all metrics for any future chart)
        rows_full = []
        for model, scenarios in aug_acc.items():
            row_acc = {"Model": model}
            for scen, val in scenarios.items():
                # val is either a dict {Accuracy, F1, ...} or a float (legacy)
                if isinstance(val, dict):
                    row_acc[scen] = f"{val['Accuracy']*100:.2f}%"
                    for metric_name, metric_val in val.items():
                        rows_full.append({
                            "Model": model, "Scenario": scen,
                            "Metric": metric_name, "Value": metric_val,
                        })
                else:
                    row_acc[scen] = f"{val*100:.2f}%"
            rows_acc.append(row_acc)

        df = pd.DataFrame(rows_acc)
        path = out / "supervisor_augmentation_table.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

        if rows_full:
            df_full = pd.DataFrame(rows_full)
            path_full = out / "supervisor_augmentation_full_metrics.csv"
            df_full.to_csv(path_full, index=False)
            print(f"  Saved: {path_full.name}")

    # --- Table 4: Regression Augmentation (RMSE comparison) ---
    aug_results = metrics.get("data_augmentation", {})
    if aug_results:
        from rgan.synthetic_analysis import create_data_augmentation_table
        path = out / "data_augmentation_table.csv"
        create_data_augmentation_table(aug_results, path)
        print(f"  Saved: {path.name}")

    # --- Visualization: Real vs Synthetic sequences ---
    comp = metrics.get("comparison_data", {})
    if comp.get("real_sequences") and comp.get("synthetic_sequences"):
        real = np.array(comp["real_sequences"])
        syn = np.array(comp["synthetic_sequences"])
        n_seq = min(5, len(real), len(syn))
        from rgan.synthetic_analysis import plot_real_vs_synthetic_sequences, plot_real_vs_synthetic_kde
        viz1 = plot_real_vs_synthetic_sequences(real[:n_seq], syn[:n_seq], str(out / "real_vs_synthetic_sequences"), n_samples=n_seq)
        print(f"  Saved: {Path(viz1['static']).name}")
        viz2 = plot_real_vs_synthetic_kde(real, syn, str(out / "real_vs_synthetic_kde"), seed=metrics.get("seed", 42))
        print(f"  Saved: {Path(viz2['static']).name}")

    # --- Visualization: Synthetic quality heatmap ---
    if syn_quality:
        from rgan.synthetic_analysis import plot_synthetic_quality_heatmap
        viz = plot_synthetic_quality_heatmap(syn_quality, str(out / "synthetic_quality_heatmap"))
        print(f"  Saved: {Path(viz['static']).name}")

    # --- Visualization: Augmentation comparison bar chart ---
    if aug_results:
        from rgan.synthetic_analysis import plot_data_augmentation_comparison
        viz = plot_data_augmentation_comparison(aug_results, str(out / "data_augmentation_comparison"))
        print(f"  Saved: {Path(viz['static']).name}")

    print(f"\nDone — all charts regenerated in {out}")
    return metrics


def regenerate_multi_dataset(
    results_dirs: List[Path],
    dataset_names: List[str],
    output_dir: Path,
):
    """Generate cross-dataset comparison charts from multiple augmentation runs."""
    from rgan.plots import (
        plot_noise_robustness_multi_dataset,
        plot_ranking_stability,
        plot_clean_vs_noisy_rankings,
        plot_seed_boxplots,
        create_mean_std_table,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for rd, name in zip(results_dirs, dataset_names):
        all_metrics[name] = _load_metrics(rd)
        print(f"Loaded: {name} from {rd}")

    # --- Cross-dataset augmentation comparison table ---
    print("\n--- Cross-Dataset Regression Augmentation ---")
    rows = []
    for ds_name, m in all_metrics.items():
        aug = m.get("data_augmentation", {})
        for model, results in aug.items():
            real_rmse = results.get("real_only", {}).get("rmse", None)
            mixed_rmse = results.get("real_plus_synthetic", {}).get("rmse", None)
            if real_rmse is not None and mixed_rmse is not None:
                delta = (mixed_rmse - real_rmse) / real_rmse * 100
                rows.append({
                    "Dataset": ds_name,
                    "Model": model,
                    "Real Only RMSE": f"{real_rmse:.6f}",
                    "Augmented RMSE": f"{mixed_rmse:.6f}",
                    "Change (%)": f"{delta:+.2f}%",
                })
    if rows:
        df = pd.DataFrame(rows)
        path = output_dir / "cross_dataset_augmentation_table.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    # --- Cross-dataset synthetic quality comparison ---
    print("\n--- Cross-Dataset Synthetic Quality ---")
    rows = []
    for ds_name, m in all_metrics.items():
        for method, q in m.get("synthetic_quality", {}).items():
            disc = q.get("all_discrimination", {}).get("Random Forest", {})
            rows.append({
                "Dataset": ds_name,
                "Method": method,
                "FD": f"{q['frechet_distance']:.4f}",
                "Var Diff (%)": f"{q['variance_difference']['rel_diff']:.2f}%",
                "Disc Score": f"{disc.get('accuracy', float('nan')):.4f}",
            })
    if rows:
        df = pd.DataFrame(rows)
        path = output_dir / "cross_dataset_synthetic_quality.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    # --- Cross-dataset supervisor augmentation accuracy ---
    print("\n--- Cross-Dataset Classification Augmentation ---")
    rows = []
    for ds_name, m in all_metrics.items():
        aug_acc = m.get("supervisor_augmentation_accuracy", {})
        for model, scenarios in aug_acc.items():
            row = {"Dataset": ds_name, "Model": model}
            for scen, val in scenarios.items():
                acc = val["Accuracy"] if isinstance(val, dict) else val
                row[scen] = f"{acc*100:.2f}%"
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        path = output_dir / "cross_dataset_classification_augmentation.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    print(f"\nDone — cross-dataset charts saved to {output_dir}")


def regenerate_multi_seed(
    results_dirs: List[Path],
    dataset_name: str,
    output_dir: Path,
):
    """Generate mean ± std tables from multiple seed runs of the same dataset."""
    from rgan.plots import create_mean_std_table, plot_seed_boxplots

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for rd in results_dirs:
        m = _load_metrics(rd)
        all_metrics.append(m)
        print(f"Loaded seed={m.get('seed', '?')} from {rd}")

    # --- Regression: mean ± std RMSE across seeds ---
    print(f"\n--- {dataset_name}: Mean ± Std RMSE (Regression Augmentation) ---")

    # Collect per-model RMSE across seeds
    seed_rmse = {"Real Only": {}, "Augmented": {}}
    for m in all_metrics:
        aug = m.get("data_augmentation", {})
        for model, results in aug.items():
            real_rmse = results.get("real_only", {}).get("rmse")
            mixed_rmse = results.get("real_plus_synthetic", {}).get("rmse")
            if real_rmse is not None:
                seed_rmse["Real Only"].setdefault(model, []).append(real_rmse)
            if mixed_rmse is not None:
                seed_rmse["Augmented"].setdefault(model, []).append(mixed_rmse)

    rows = []
    all_models = sorted(set(
        list(seed_rmse["Real Only"].keys()) + list(seed_rmse["Augmented"].keys())
    ))
    for model in all_models:
        row = {"Model": model}
        for condition in ["Real Only", "Augmented"]:
            vals = seed_rmse[condition].get(model, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[condition] = f"{mean:.6f} ± {std:.6f}"
            else:
                row[condition] = "—"
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        path = output_dir / f"mean_std_rmse_{dataset_name.lower().replace(' ', '_')}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

        # Print formatted table
        print(f"\n  {dataset_name} — Regression RMSE (mean ± std over {len(all_metrics)} seeds):")
        print(df.to_string(index=False))

    # --- Classification: mean ± std accuracy across seeds ---
    print(f"\n--- {dataset_name}: Mean ± Std Accuracy (Classification Augmentation) ---")
    seed_acc = {}
    for m in all_metrics:
        aug_acc = m.get("supervisor_augmentation_accuracy", {})
        for model, scenarios in aug_acc.items():
            for scen, val in scenarios.items():
                acc = val["Accuracy"] if isinstance(val, dict) else val
                key = (model, scen)
                seed_acc.setdefault(key, []).append(acc)

    if seed_acc:
        # Pivot into table
        models = sorted(set(k[0] for k in seed_acc))
        scenarios = sorted(set(k[1] for k in seed_acc))
        rows = []
        for model in models:
            row = {"Model": model}
            for scen in scenarios:
                vals = seed_acc.get((model, scen), [])
                if vals:
                    mean = np.mean(vals) * 100
                    std = np.std(vals, ddof=1) * 100 if len(vals) > 1 else 0.0
                    row[scen] = f"{mean:.2f} ± {std:.2f}%"
                else:
                    row[scen] = "—"
            rows.append(row)
        df = pd.DataFrame(rows)
        path = output_dir / f"mean_std_accuracy_{dataset_name.lower().replace(' ', '_')}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")
        print(df.to_string(index=False))

    # --- Synthetic quality: mean ± std across seeds ---
    print(f"\n--- {dataset_name}: Mean ± Std Synthetic Quality ---")
    seed_quality = {}
    for m in all_metrics:
        for method, q in m.get("synthetic_quality", {}).items():
            disc = q.get("all_discrimination", {}).get("Random Forest", {})
            seed_quality.setdefault(method, {"fd": [], "var": [], "disc": []})
            seed_quality[method]["fd"].append(q["frechet_distance"])
            seed_quality[method]["var"].append(q["variance_difference"]["rel_diff"])
            seed_quality[method]["disc"].append(disc.get("accuracy", float("nan")))

    if seed_quality:
        rows = []
        for method, vals in seed_quality.items():
            fd_m, fd_s = np.mean(vals["fd"]), np.std(vals["fd"], ddof=1) if len(vals["fd"]) > 1 else 0
            var_m, var_s = np.mean(vals["var"]), np.std(vals["var"], ddof=1) if len(vals["var"]) > 1 else 0
            disc_m, disc_s = np.nanmean(vals["disc"]), np.nanstd(vals["disc"], ddof=1) if len(vals["disc"]) > 1 else 0
            rows.append({
                "Method": method,
                "FD": f"{fd_m:.4f} ± {fd_s:.4f}",
                "Var Diff (%)": f"{var_m:.2f} ± {var_s:.2f}%",
                "Disc Score": f"{disc_m:.4f} ± {disc_s:.4f}",
            })
        df = pd.DataFrame(rows)
        path = output_dir / f"mean_std_quality_{dataset_name.lower().replace(' ', '_')}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")
        print(df.to_string(index=False))

    # --- Box plot: RMSE variance across seeds ---
    if seed_rmse["Real Only"]:
        box_data = {dataset_name: seed_rmse["Real Only"]}
        path = output_dir / f"seed_boxplot_{dataset_name.lower().replace(' ', '_')}.png"
        plot_seed_boxplots(box_data, "RMSE", str(path))
        print(f"\n  Saved: {path.name}")

    print(f"\nDone — multi-seed analysis saved to {output_dir}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Regenerate charts from saved augmentation metrics (no retraining needed)"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Single augmentation results directory containing metrics_augmentation.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for regenerated charts (defaults to results_dir)"
    )
    parser.add_argument(
        "--multi", nargs="+", default=None,
        help="Multiple results dirs for cross-dataset comparison"
    )
    parser.add_argument(
        "--dataset_names", nargs="+", default=None,
        help="Dataset names corresponding to --multi dirs"
    )
    parser.add_argument(
        "--multi_seeds", nargs="+", default=None,
        help="Multiple results dirs from different seeds of the SAME dataset"
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="Dataset name for --multi_seeds"
    )
    return parser


def cli_main():
    args = _build_parser().parse_args()

    if args.results_dir:
        rd = Path(args.results_dir)
        out = Path(args.output_dir) if args.output_dir else rd
        regenerate_single(rd, out)

    elif args.multi:
        if not args.dataset_names or len(args.dataset_names) != len(args.multi):
            print("ERROR: --dataset_names must match --multi (one name per dir)")
            sys.exit(1)
        out = Path(args.output_dir) if args.output_dir else Path("results/cross_dataset")
        regenerate_multi_dataset(
            [Path(d) for d in args.multi],
            args.dataset_names,
            out,
        )

    elif args.multi_seeds:
        if not args.dataset_name:
            print("ERROR: --dataset_name required with --multi_seeds")
            sys.exit(1)
        out = Path(args.output_dir) if args.output_dir else Path(f"results/multi_seed_{args.dataset_name.lower().replace(' ', '_')}")
        regenerate_multi_seed(
            [Path(d) for d in args.multi_seeds],
            args.dataset_name,
            out,
        )

    else:
        print("ERROR: Provide --results_dir, --multi, or --multi_seeds")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
