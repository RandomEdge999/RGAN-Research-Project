#!/usr/bin/env python
"""
Data Augmentation Experiment Runner

Standalone script that:
1. Loads existing trained RGAN model
2. Generates synthetic data
3. Trains classical models on real vs mixed data
4. Computes synthetic quality metrics
5. Creates comparison visualizations

Can be deleted entirely without affecting the main pipeline.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from rgan.data import load_csv_series, interpolate_and_standardize, make_windows_univariate
from rgan.baselines import arima_forecast, arma_forecast, tree_ensemble_forecast, naive_baseline
from rgan.lstm_supervised_torch import train_lstm_supervised_torch
from rgan.config import TrainConfig
from rgan.metrics import error_stats, summarise_with_uncertainty
from rgan.fits import train_fits
from rgan.patchtst import train_patchtst
from rgan.itransformer import train_itransformer
from rgan.timegan import train_timegan
from rgan.synthetic_analysis import (
    generate_synthetic_sequences,
    frechet_distance,
    variance_difference,
    discrimination_score,
    evaluate_discriminators,
    create_classification_metrics_table,
    create_synthetic_quality_table,
    create_data_augmentation_table,
    plot_real_vs_synthetic_sequences,
    plot_real_vs_synthetic_distributions,
    plot_data_augmentation_comparison
)

# New imports for Supervisor Tables
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_trend_labels(X, Y):
    """
    Convert forecasting task to binary trend classification.
    Class 1: Average of Future > Last Observed Value (Increase)
    Class 0: Average of Future <= Last Observed Value (Decrease/Stable)
    """
    # X: (N, L, 1), Y: (N, H, 1)
    last_obs = X[:, -1, 0]
    future_mean = np.mean(Y, axis=1)[:, 0]
    return (future_mean > last_obs).astype(int)


def train_evaluate_classifiers(X_train, y_train, X_test, y_test):
    """Train RF, SVM, MLP on flattened time series and return metrics."""
    # Flatten timeseries for standard classifiers
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', random_state=42)),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    }

    results = {}
    for name, clf in classifiers.items():
        # print(f"    Training {name}...") # Too verbose for main script?
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0)
        }
    return results


def main(args):
    """Run augmentation experiment."""
    print("="*80)
    print("SYNTHETIC DATA AUGMENTATION EXPERIMENT")
    print("="*80)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # ========================================================================
    # STEP 1: Load and prepare data
    # ========================================================================
    print("\n[STEP 1] Loading and preparing data...")
    data, target_col, time_col = load_csv_series(args.csv, target=args.target_col, time_col=args.time_col)
    print(f"  Loaded {len(data)} time series samples from {args.csv}")

    data_split = interpolate_and_standardize(data, target_col, train_ratio=args.train_split)
    data_train = data_split['scaled_train']
    data_test = data_split['scaled_test']

    print(f"  Train size: {len(data_train)}, Test size: {len(data_test)}")

    # ========================================================================
    # STEP 2: Create windows
    # ========================================================================
    print(f"\n[STEP 2] Creating time windows (L={args.L}, H={args.H})...")
    X_train, Y_train = make_windows_univariate(data_train, target_col, args.L, args.H)
    X_test, Y_test = make_windows_univariate(data_test, target_col, args.L, args.H)

    print(f"  Train windows: {X_train.shape}, Test windows: {X_test.shape}")

    # ========================================================================
    # STEP 3: Load RGAN model (if exists in results directory)
    # ========================================================================
    print("\n[STEP 3] Loading RGAN model...")

    # Try to find RGAN checkpoint from existing experiments
    rgan_model_path = None
    rgan_config_path = None
    if args.rgan_model:
        rgan_model_path = Path(args.rgan_model)
    else:
        # Search for RGAN model in results directory
        results_root = Path('results')
        if results_root.exists():
            for potential_model in results_root.rglob('rgan_model.pt'):
                rgan_model_path = potential_model
                break

    # Also look for the metrics.json next to the checkpoint to get architecture params
    if rgan_model_path and rgan_model_path.exists():
        rgan_config_path = rgan_model_path.parent / 'metrics.json'

    has_rgan = False
    G = None
    if rgan_model_path and rgan_model_path.exists():
        print(f"  Loading RGAN from: {rgan_model_path}")
        try:
            rgan_checkpoint = torch.load(rgan_model_path, map_location='cpu', weights_only=False)

            # Try to read architecture params from metrics.json
            g_units = 64
            g_layers = 1
            g_layer_norm = True
            if rgan_config_path and rgan_config_path.exists():
                with open(rgan_config_path) as f:
                    saved_metrics = json.load(f)
                cfg = saved_metrics.get("rgan", {}).get("config", {})
                g_units = cfg.get("units_g", g_units)
                g_layers = cfg.get("g_layers", g_layers)
                print(f"  Architecture from metrics.json: units={g_units}, layers={g_layers}")

            from rgan.models_torch import build_generator
            G = build_generator(args.L, args.H, n_in=1, units=g_units,
                                num_layers=g_layers, layer_norm=g_layer_norm)
            G.load_state_dict(rgan_checkpoint)
            G.eval()
            has_rgan = True
            print("  RGAN Generator loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load RGAN model: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  Warning: RGAN model not found.")

    # ========================================================================
    # STEP 4: Generate synthetic data (RGAN + TimeGAN)
    # ========================================================================
    print("\n[STEP 4] Generating synthetic data...")

    # --- 4a: RGAN synthetic data (with stochastic diversity) ---
    rng = np.random.default_rng(42)
    if has_rgan and G is not None:
        print("  Generating RGAN synthetic data (with dropout noise for diversity)...")
        n_synth = len(X_train)
        # Use multiple forward passes with dropout noise to create diverse outputs
        # This injects stochasticity into a conditional generator G(X) -> Y
        G.train()  # Enable dropout for stochastic generation
        Y_rgan_runs = []
        n_runs = 5  # Multiple forward passes for diversity
        with torch.no_grad():
            X_torch = torch.from_numpy(X_train).float()
            for run_i in range(n_runs):
                # Add small input perturbation for extra diversity
                noise = torch.randn_like(X_torch) * 0.02
                Y_run = G(X_torch + noise).numpy()
                Y_rgan_runs.append(Y_run)
        G.eval()

        # Pick one run per sample (round-robin) for maximum diversity
        Y_synthetic = np.empty_like(Y_rgan_runs[0])
        for i in range(n_synth):
            Y_synthetic[i] = Y_rgan_runs[i % n_runs][i]

        # Verify scale consistency: synthetic should have similar stats to real
        real_mean, real_std = Y_train.mean(), Y_train.std()
        syn_mean, syn_std = Y_synthetic.mean(), Y_synthetic.std()
        print(f"  Scale check — Real: mean={real_mean:.4f} std={real_std:.4f}")
        print(f"  Scale check — Syn:  mean={syn_mean:.4f} std={syn_std:.4f}")

        # If scale is way off, re-normalize synthetic to match real distribution
        if abs(syn_mean - real_mean) > 2 * real_std or syn_std > 3 * real_std or syn_std < 0.1 * real_std:
            print("  WARNING: Scale mismatch detected! Re-normalizing synthetic data.")
            Y_synthetic = (Y_synthetic - syn_mean) / (syn_std + 1e-8) * real_std + real_mean

        print(f"  Generated {len(Y_synthetic)} RGAN synthetic sequences")
    else:
        # No RGAN: use Gaussian perturbation as baseline
        print("  No RGAN model — using Gaussian perturbation as synthetic baseline")
        Y_synthetic = Y_train + rng.normal(0, 0.15 * Y_train.std(), size=Y_train.shape).astype(Y_train.dtype)
        print(f"  Generated {len(Y_synthetic)} perturbed sequences")

    # --- 4b: TimeGAN synthetic data ---
    print("  Training TimeGAN for synthetic data comparison...")
    try:
        # Reshape X_train for TimeGAN: (n_samples, seq_len, 1)
        # TimeGAN generates full sequences, so we use X_train windows
        timegan_result = train_timegan(
            X_train,  # (n_samples, L, 1)
            hidden_dim=24, latent_dim=24, n_layers=1,
            epochs_ae=30, epochs_sup=30, epochs_joint=30,
            batch_size=min(128, len(X_train)),
            lr=1e-3, device='auto',
        )
        # TimeGAN generates sequences of the same shape as input
        # For augmentation, we need (X_syn, Y_syn) pairs
        # Generate synthetic X, then use the RGAN generator (if available) to predict Y
        X_timegan = timegan_result["synthetic_data"].astype(np.float32)
        if has_rgan and G is not None:
            G.eval()
            with torch.no_grad():
                Y_timegan = G(torch.from_numpy(X_timegan).float()).numpy()
        else:
            # Without RGAN, can't generate Y from TimeGAN X
            # Use a simple shifted window approach instead
            Y_timegan = None
        has_timegan = True
        print(f"  TimeGAN: generated {len(X_timegan)} synthetic sequences")
    except Exception as e:
        print(f"  TimeGAN training failed: {e}")
        import traceback; traceback.print_exc()
        has_timegan = False
        X_timegan = None
        Y_timegan = None

    # ========================================================================
    # STEP 5: Create mixed dataset
    # ========================================================================
    print("\n[STEP 5] Creating mixed real + synthetic dataset...")

    X_mixed = np.concatenate([X_train, X_train], axis=0)
    Y_mixed = np.concatenate([Y_train, Y_synthetic], axis=0)

    print(f"  Mixed dataset: X shape {X_mixed.shape}, Y shape {Y_mixed.shape}")

    # ========================================================================
    # STEP 6: Train classical models on both datasets
    # ========================================================================
    print("\n[STEP 6] Training models...")

    augmentation_results = {}

    # 1. Naive Baseline
    print("  Evaluating Naive Baseline...")
    try:
        # Naive doesn't "train", but we evaluate on Real vs Mixed (Mixed doesn't change prediction for Naive)
        # Actually naive just predicts last value. So "Training" on mixed is irrelevant.
        # But we report it for completeness on the Test set.
        naive_stats, _ = naive_baseline(X_test, Y_test)
        augmentation_results['Naive'] = {
            'real_only': naive_stats,
            'real_plus_synthetic': naive_stats # Same
        }
        print(f"    Naive RMSE: {naive_stats['rmse']:.6f}")
    except Exception as e:
        print(f"    Naive failed: {e}")

    # 2. ARIMA
    print("  Training ARIMA...")
    try:
        arima_real_stats, arima_real_preds = arima_forecast(X_train, Y_train, X_eval=X_test, Y_eval=Y_test)
        arima_mixed_stats, arima_mixed_preds = arima_forecast(X_mixed, Y_mixed, X_eval=X_test, Y_eval=Y_test)

        arima_real_metrics = error_stats(Y_test, arima_real_preds)
        arima_mixed_metrics = error_stats(Y_test, arima_mixed_preds)

        augmentation_results['ARIMA'] = {
            'real_only': arima_real_metrics,
            'real_plus_synthetic': arima_mixed_metrics
        }
        print(f"    ARIMA RMSE - Real only: {arima_real_metrics['rmse']:.6f}, Mixed: {arima_mixed_metrics['rmse']:.6f}")
    except Exception as e:
        print(f"    ARIMA failed: {e}")

    # 3. ARMA
    print("  Training ARMA...")
    try:
        arma_real_stats, arma_real_preds = arma_forecast(X_train, Y_train, X_eval=X_test, Y_eval=Y_test)
        arma_mixed_stats, arma_mixed_preds = arma_forecast(X_mixed, Y_mixed, X_eval=X_test, Y_eval=Y_test)

        arma_real_metrics = error_stats(Y_test, arma_real_preds)
        arma_mixed_metrics = error_stats(Y_test, arma_mixed_preds)

        augmentation_results['ARMA'] = {
            'real_only': arma_real_metrics,
            'real_plus_synthetic': arma_mixed_metrics
        }
        print(f"    ARMA RMSE - Real only: {arma_real_metrics['rmse']:.6f}, Mixed: {arma_mixed_metrics['rmse']:.6f}")
    except Exception as e:
        print(f"    ARMA failed: {e}")

    # 4. Tree Ensemble
    print("  Training Tree Ensemble...")
    try:
        tree_real_stats, tree_real_preds = tree_ensemble_forecast(X_train, Y_train, X_eval=X_test, Y_eval=Y_test)
        tree_mixed_stats, tree_mixed_preds = tree_ensemble_forecast(X_mixed, Y_mixed, X_eval=X_test, Y_eval=Y_test)

        tree_real_metrics = error_stats(Y_test, tree_real_preds)
        tree_mixed_metrics = error_stats(Y_test, tree_mixed_preds)

        augmentation_results['Tree_Ensemble'] = {
            'real_only': tree_real_metrics,
            'real_plus_synthetic': tree_mixed_metrics
        }
        print(f"    Tree Ensemble RMSE - Real only: {tree_real_metrics['rmse']:.6f}, Mixed: {tree_mixed_metrics['rmse']:.6f}")
    except Exception as e:
        print(f"    Tree Ensemble failed: {e}")

    # 5. Neural models (LSTM, FITS, PatchTST, iTransformer)
    # These are the models that actually benefit from more training data
    print("  Training neural models (real vs augmented)...")

    # Create validation splits
    val_idx = int(len(X_train) * 0.9)
    Xtr_sub, Ytr_sub = X_train[:val_idx], Y_train[:val_idx]
    Xval_sub, Yval_sub = X_train[val_idx:], Y_train[val_idx:]

    val_idx_mixed = int(len(X_mixed) * 0.9)
    Xtr_mixed_sub, Ytr_mixed_sub = X_mixed[:val_idx_mixed], Y_mixed[:val_idx_mixed]
    Xval_mixed_sub, Yval_mixed_sub = X_mixed[val_idx_mixed:], Y_mixed[val_idx_mixed:]

    nn_config = TrainConfig(
        L=args.L, H=args.H,
        epochs=30,
        batch_size=64,
        units_g=64,
        device='auto',
        patience=8,
    )

    real_splits = {"Xtr": Xtr_sub, "Ytr": Ytr_sub, "Xval": Xval_sub, "Yval": Yval_sub,
                   "Xte": X_test, "Yte": Y_test}
    mixed_splits = {"Xtr": Xtr_mixed_sub, "Ytr": Ytr_mixed_sub,
                    "Xval": Xval_mixed_sub, "Yval": Yval_mixed_sub,
                    "Xte": X_test, "Yte": Y_test}

    # Helper to run a model on real vs mixed and record results
    def _run_augmentation_test(name, train_fn, real_sp, mixed_sp, **kwargs):
        try:
            print(f"    {name} on Real Data...")
            real_out = train_fn(nn_config, real_sp, str(results_dir), tag=f"{name}_real", **kwargs)
            print(f"    {name} on Mixed Data...")
            mixed_out = train_fn(nn_config, mixed_sp, str(results_dir), tag=f"{name}_mixed", **kwargs)
            augmentation_results[name] = {
                'real_only': real_out['test_stats'],
                'real_plus_synthetic': mixed_out['test_stats'],
            }
            r_rmse = real_out['test_stats']['rmse']
            m_rmse = mixed_out['test_stats']['rmse']
            delta = (m_rmse - r_rmse) / r_rmse * 100
            print(f"    {name} RMSE — Real: {r_rmse:.6f}, Mixed: {m_rmse:.6f} ({delta:+.1f}%)")
        except Exception as e:
            print(f"    {name} failed: {e}")
            import traceback; traceback.print_exc()

    _run_augmentation_test("LSTM", train_lstm_supervised_torch, real_splits, mixed_splits)
    _run_augmentation_test("FITS", train_fits, real_splits, mixed_splits)
    _run_augmentation_test("PatchTST", train_patchtst, real_splits, mixed_splits)
    _run_augmentation_test("iTransformer", train_itransformer, real_splits, mixed_splits)

    # ========================================================================
    # STEP 7: Compute synthetic quality metrics (RGAN vs TimeGAN)
    # ========================================================================
    print("\n[STEP 7] Computing synthetic quality metrics...")

    def _compute_quality(name, Y_real, Y_syn):
        """Compute FD, variance diff, and discrimination score for a generator."""
        print(f"  [{name}] Fréchet Distance...")
        fd = frechet_distance(Y_real, Y_syn)
        print(f"    FD: {fd:.6f}")
        print(f"  [{name}] Variance Difference...")
        var_diff = variance_difference(Y_real, Y_syn)
        print(f"    Var Diff (rel): {var_diff['rel_diff']:.2f}%")
        print(f"  [{name}] Discrimination Score (RF/SVM/MLP)...")
        disc = evaluate_discriminators(Y_real, Y_syn, n_folds=3)
        for m, s in disc.items():
            print(f"    {m} Accuracy: {s['accuracy']:.4f}")
        return {
            'frechet_distance': fd,
            'variance_difference': var_diff,
            'discrimination_score': disc.get('Random Forest'),
            'all_discrimination': disc,
        }

    synthetic_quality = {}
    synthetic_quality['RGAN'] = _compute_quality("RGAN", Y_train, Y_synthetic)
    disc_scores = synthetic_quality['RGAN']['all_discrimination']

    if has_timegan and X_timegan is not None:
        synthetic_quality['TimeGAN'] = _compute_quality("TimeGAN", X_train, X_timegan)

    # ========================================================================
    # STEP 8: Create classification metrics from synthetic quality metrics
    # ========================================================================
    print("\n[STEP 8] Classification metrics ready from synthetic quality analysis...")

    # Classification metrics are the discrimination scores from synthetic quality
    # This shows how well a classifier can distinguish real from synthetic data
    classification_results = disc_scores

    # ========================================================================
    # STEP 9: Create tables
    # ========================================================================
    print("\n[STEP 9] Creating comparison tables...")

    # Table 1: Classification metrics
    if classification_results:
        table1_path = results_dir / 'classification_metrics_table.csv'
        create_classification_metrics_table(classification_results, table1_path)
        print(f"  Saved: {table1_path}")

    # Table 2: Synthetic quality
    table2_path = results_dir / 'synthetic_quality_table.csv'
    create_synthetic_quality_table(synthetic_quality, table2_path)
    print(f"  Saved: {table2_path}")

    # Table 3: Data augmentation
    if augmentation_results:
        table3_path = results_dir / 'data_augmentation_table.csv'
        create_data_augmentation_table(augmentation_results, table3_path)
        print(f"  Saved: {table3_path}")

    # ========================================================================
    # STEP 10: Create visualizations
    # ========================================================================
    print("\n[STEP 10] Creating visualizations...")

    # Real vs synthetic sequences
    n_seq = min(5, len(Y_train), len(Y_synthetic))
    viz1_path = results_dir / 'real_vs_synthetic_sequences'
    viz1 = plot_real_vs_synthetic_sequences(Y_train[:n_seq], Y_synthetic[:n_seq], str(viz1_path), n_samples=n_seq)
    print(f"  Saved: {viz1['static']}")
    if viz1['interactive']:
        print(f"  Saved: {viz1['interactive']}")

    # Distributions
    viz2_path = results_dir / 'real_vs_synthetic_distributions'
    viz2 = plot_real_vs_synthetic_distributions(Y_train, Y_synthetic, str(viz2_path))
    print(f"  Saved: {viz2['static']}")
    if viz2['interactive']:
        print(f"  Saved: {viz2['interactive']}")

    # Augmentation comparison
    if augmentation_results:
        viz3_path = results_dir / 'data_augmentation_comparison'
        viz3 = plot_data_augmentation_comparison(augmentation_results, str(viz3_path))
        print(f"  Saved: {viz3['static']}")
        if viz3['interactive']:
            print(f"  Saved: {viz3['interactive']}")

    # ========================================================================
    # STEP 11: Classification & Model Comparison (Supervisor Tables)
    # ========================================================================
    print("\n[STEP 11] Generating Classification & Model Comparison Tables...")

    # 1. Create Binary Labels from Real Data
    y_train_cls = get_trend_labels(X_train, Y_train)
    y_test_cls = get_trend_labels(X_test, Y_test)
    print(f"  Binary Classification Task: Predict Trend (Increase vs Decrease)")
    print(f"  Class Balance (Train): {np.mean(y_train_cls):.2f} positive")

    # 2. Table 1: Baseline Classification Metrics (Real Data)
    print("\n  Generating Table 1: Baseline Classification on Real Data...")
    baseline_metrics = train_evaluate_classifiers(X_train, y_train_cls, X_test, y_test_cls)
    
    table1_rows = []
    for model_name, metrics in baseline_metrics.items():
        row = {'Model': model_name}
        for k, v in metrics.items():
            row[k] = f"{v*100:.2f}%"
        table1_rows.append(row)
    
    df_table1 = pd.DataFrame(table1_rows)
    cols = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
    df_table1 = df_table1[cols]
    df_table1.to_csv(results_dir / "table1_baseline_classification.csv", index=False)
    print("    Saved: table1_baseline_classification.csv")

    # 3. Table 2 & 3: GAN vs WGAN Quality & Augmentation
    # We need the WGAN model for full comparison
    
    # Store synthetic datasets for classification augmentation
    syn_datasets = {}
    if 'RGAN' in synthetic_quality:
        syn_datasets['RGAN'] = Y_synthetic

    # Load WGAN if provided
    if args.wgan_model:
        print(f"\n  Loading WGAN model from {args.wgan_model}...")
        try:
            wgan_checkpoint = torch.load(args.wgan_model, map_location='cpu', weights_only=False)
            from rgan.models_torch import build_generator
            G_wgan = build_generator(args.L, args.H, n_in=1, units=g_units if has_rgan else 64,
                                     num_layers=g_layers if has_rgan else 1, layer_norm=True)
            G_wgan.load_state_dict(wgan_checkpoint)
            G_wgan.eval()
            with torch.no_grad():
                X_torch = torch.from_numpy(X_train).float()
                Y_wgan = G_wgan(X_torch).numpy()
            syn_datasets['WGAN-GP'] = Y_wgan
            print(f"  Generated {len(Y_wgan)} WGAN sequences.")
        except Exception as e:
            print(f"  Failed to load/run WGAN: {e}")

    if has_timegan and Y_timegan is not None:
        syn_datasets['TimeGAN'] = Y_timegan

    # Generate Table 2: Quantitative Quality (GAN vs WGAN)
    if syn_datasets:
        print("\n  Generating Table 2: GAN vs WGAN Quality Metrics...")
        quality_results = []
        for name, Y_syn in syn_datasets.items():
            print(f"    Computing metrics for {name}...")
            fd_val = frechet_distance(Y_train, Y_syn)
            var_val = variance_difference(Y_train, Y_syn)['abs_diff']
            # Discrimination Score
            disc_res = evaluate_discriminators(Y_train, Y_syn, n_folds=3)
            disc_acc = disc_res['Random Forest']['accuracy']
            
            quality_results.append({
                'Method': name,
                'FD': f"{fd_val:.4f}",
                'Variance diff.': f"{var_val:.4f}",
                'Discrimination Score': f"{disc_acc:.4f}"
            })
        
        df_table2 = pd.DataFrame(quality_results)
        df_table2.to_csv(results_dir / "table2_gan_quality.csv", index=False)
        print("    Saved: table2_gan_quality.csv")
    
    # Generate Table 3: Augmentation Performance
    print("\n  Generating Table 3: Augmentation Performance (Accuracy)...")
    
    # Dict: Model -> Scenario -> Accuracy
    # Initialized with Real Only results
    aug_table_data = {m: {'Real Only': baseline_metrics[m]['Accuracy']} for m in baseline_metrics}
    
    for syn_name, Y_syn in syn_datasets.items():
        print(f"    Training with {syn_name} Augmentation...")
        # Augment
        X_aug = np.concatenate([X_train, X_train], axis=0)
        
        # Labels for synthetic data (Consistency: derived from synthetic series)
        y_syn_cls = get_trend_labels(X_train, Y_syn)
        y_aug = np.concatenate([y_train_cls, y_syn_cls], axis=0)
        
        # Train & Evaluate
        aug_metrics = train_evaluate_classifiers(X_aug, y_aug, X_test, y_test_cls)
        
        col_name = f"+ {syn_name} Synthetic"
        for m, res in aug_metrics.items():
            aug_table_data[m][col_name] = res['Accuracy']

    # Format Table 3
    rows = []
    for model, scenarios in aug_table_data.items():
        row = {'Model': model}
        for scen, acc in scenarios.items():
            row[scen] = f"{acc*100:.2f}%"
        rows.append(row)
        
    df_table3 = pd.DataFrame(rows)
    # Order columns
    desired = ['Model', 'Real Only', '+ RGAN Synthetic', '+ WGAN-GP Synthetic', '+ TimeGAN Synthetic']
    final_cols = [c for c in desired if c in df_table3.columns]
    # Add any other columns found
    for c in df_table3.columns:
        if c not in final_cols:
            final_cols.append(c)
            
    df_table3 = df_table3[final_cols]
    df_table3.to_csv(results_dir / "table3_augmentation.csv", index=False)
    print("    Saved: table3_augmentation.csv")

    
    # ========================================================================
    # STEP 12: Create metrics JSON
    # ========================================================================
    print("\n[STEP 12] Creating metrics JSON...")

    metrics_json = {
        'dataset': str(args.csv),
        'L': args.L,
        'H': args.H,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'classification_metrics': classification_results,
        'synthetic_quality': synthetic_quality,
        'data_augmentation': augmentation_results,
        'comparison_data': {
            'real_sequences': Y_train[:min(100, len(Y_train))].tolist(),
            'synthetic_sequences': Y_synthetic[:min(100, len(Y_synthetic))].tolist()
        }
    }

    metrics_path = results_dir / 'metrics_augmentation.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nGenerated files:")
    print(f"  - classification_metrics_table.csv")
    print(f"  - synthetic_quality_table.csv")
    print(f"  - data_augmentation_table.csv")
    print(f"  - real_vs_synthetic_sequences.png/html")
    print(f"  - real_vs_synthetic_distributions.png/html")
    print(f"  - data_augmentation_comparison.png/html")
    print(f"  - metrics_augmentation.json")


def _build_parser():
    parser = argparse.ArgumentParser(description='Run synthetic data augmentation experiments')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--time_col', type=str, default=None,
                       help='Name of time column (auto-detected if not specified)')
    parser.add_argument('--target_col', type=str, default=None,
                       help='Name of target column (auto-detected if not specified)')
    parser.add_argument('--L', type=int, default=24,
                       help='Lookback window size')
    parser.add_argument('--H', type=int, default=12,
                       help='Forecast horizon')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/test split ratio')
    parser.add_argument('--results_dir', type=str, default='results/augmentation',
                       help='Directory to save results')
    parser.add_argument('--rgan_model', type=str, default=None,
                       help='Path to RGAN model checkpoint')
    parser.add_argument('--wgan_model', type=str, default=None,
                       help='Path to WGAN model checkpoint (for Supervisor Comparison)')
    return parser


def cli_main():
    args = _build_parser().parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
