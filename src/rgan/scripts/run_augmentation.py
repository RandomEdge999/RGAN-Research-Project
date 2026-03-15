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
import time
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


def _fmt_elapsed(seconds):
    """Format elapsed seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main(args):
    """Run augmentation experiment."""
    t_total_start = time.time()
    print("="*80)
    print("SYNTHETIC DATA AUGMENTATION EXPERIMENT")
    print("="*80)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # ========================================================================
    # STEP 1: Resolve training config (L/H/architecture) from prior results
    # ========================================================================
    print("\n[STEP 1] Resolving training configuration...")

    # Resolve the training run directory and model path
    rgan_model_path = None
    rgan_config_path = None
    run_dir = None  # the training results directory (contains metrics.json)

    if args.rgan_model:
        rgan_model_path = Path(args.rgan_model)
    elif args.results_from:
        run_dir = Path(args.results_from)
        ema_path = run_dir / 'models' / 'rgan_generator_ema.pt'
        gen_path = run_dir / 'models' / 'rgan_generator.pt'
        if ema_path.exists():
            rgan_model_path = ema_path
            print(f"  Using EMA generator (better quality): {ema_path.name}")
        elif gen_path.exists():
            rgan_model_path = gen_path
        else:
            print(f"  ERROR: No RGAN generator found in {run_dir / 'models'}")
    else:
        results_root = Path('results')
        if results_root.exists():
            for pattern in ['rgan_generator_ema.pt', 'rgan_generator.pt']:
                matches = list(results_root.rglob(pattern))
                if matches:
                    rgan_model_path = max(matches, key=lambda p: p.stat().st_mtime)
                    if 'ema' in pattern:
                        print(f"  Auto-found EMA generator: {rgan_model_path}")
                    break

    # Locate metrics.json
    if rgan_model_path and rgan_model_path.exists():
        if run_dir is None:
            run_dir = rgan_model_path.parent.parent
        rgan_config_path = run_dir / 'metrics.json'

    # Read L/H and architecture from metrics.json BEFORE creating windows
    g_units = 128
    g_layers = 2
    g_layer_norm = True
    config_loaded = False
    saved_metrics = None

    if rgan_config_path and rgan_config_path.exists():
        with open(rgan_config_path) as f:
            saved_metrics = json.load(f)

        # Override L/H to match training (cast to int for safe comparison)
        trained_L = int(saved_metrics["L"]) if saved_metrics.get("L") is not None else None
        trained_H = int(saved_metrics["H"]) if saved_metrics.get("H") is not None else None
        if trained_L and args.L != trained_L:
            print(f"  WARNING: Training used L={trained_L} but augmentation has L={args.L}")
            print(f"           Overriding L to {trained_L} to match trained model")
            args.L = trained_L
        if trained_H and args.H != trained_H:
            print(f"  WARNING: Training used H={trained_H} but augmentation has H={args.H}")
            print(f"           Overriding H to {trained_H} to match trained model")
            args.H = trained_H

        # Read architecture params
        cfg = saved_metrics.get("rgan", {}).get("config", {})
        if cfg:
            g_units = cfg.get("units_g", g_units)
            g_layers = cfg.get("g_layers", g_layers)
            config_loaded = True
            print(f"  Architecture from metrics.json: units={g_units}, layers={g_layers}")

    print(f"  Using L={args.L}, H={args.H}")

    # ========================================================================
    # STEP 2: Load and prepare data
    # ========================================================================
    t0 = time.time()
    print("\n[STEP 2] Loading and preparing data...")
    data, target_col, time_col = load_csv_series(args.csv, target=args.target_col, time_col=args.time_col)
    print(f"  Loaded {len(data)} time series samples from {args.csv}")

    data_split = interpolate_and_standardize(data, target_col, train_ratio=args.train_split)
    data_train = data_split['scaled_train']
    data_test = data_split['scaled_test']

    print(f"  Train size: {len(data_train)}, Test size: {len(data_test)}")
    print(f"  Step 2 took {_fmt_elapsed(time.time() - t0)}")

    # ========================================================================
    # STEP 3: Create windows
    # ========================================================================
    t0 = time.time()
    print(f"\n[STEP 3] Creating time windows (L={args.L}, H={args.H})...")
    X_train, Y_train = make_windows_univariate(data_train, target_col, args.L, args.H)
    X_test, Y_test = make_windows_univariate(data_test, target_col, args.L, args.H)

    print(f"  Train windows: {X_train.shape}, Test windows: {X_test.shape}")
    print(f"  Step 3 took {_fmt_elapsed(time.time() - t0)}")

    # ========================================================================
    # STEP 4: Load RGAN model
    # ========================================================================
    print("\n[STEP 4] Loading RGAN model...")

    has_rgan = False
    G = None

    if rgan_model_path and rgan_model_path.exists():
        print(f"  Loading RGAN from: {rgan_model_path}")
        try:
            rgan_checkpoint = torch.load(rgan_model_path, map_location='cpu', weights_only=False)

            # Unwrap if checkpoint is a dict containing 'state_dict' key
            if isinstance(rgan_checkpoint, dict) and 'state_dict' in rgan_checkpoint:
                print(f"  Unwrapping state_dict from checkpoint wrapper")
                rgan_checkpoint = rgan_checkpoint['state_dict']

            # Fallback: infer architecture from state_dict if metrics.json wasn't available
            if not config_loaded:
                print("  WARNING: Could not load config from metrics.json — inferring from state_dict")
                for key, tensor in rgan_checkpoint.items():
                    if 'lstm.weight_ih_l0' in key:
                        g_units = tensor.shape[0] // 4
                        print(f"  Inferred units={g_units} from state_dict")
                        break
                layer_keys = [k for k in rgan_checkpoint if 'lstm.weight_ih_l' in k]
                if layer_keys:
                    g_layers = len(layer_keys)
                    print(f"  Inferred layers={g_layers} from state_dict")

            from rgan.models_torch import build_generator
            G = build_generator(args.L, args.H, n_in=1, units=g_units,
                                num_layers=g_layers, layer_norm=g_layer_norm)
            G.load_state_dict(rgan_checkpoint)
            G.eval()
            has_rgan = True
            param_count = sum(p.numel() for p in G.parameters())
            print(f"  RGAN Generator loaded successfully ({param_count:,} params)")
        except Exception as e:
            print(f"  ERROR: Could not load RGAN model: {e}")
            import traceback; traceback.print_exc()
            print("  Falling back to Gaussian perturbation for synthetic data")
    else:
        search_hint = ""
        if rgan_model_path:
            search_hint = f" (looked for {rgan_model_path})"
        print(f"  WARNING: RGAN model not found{search_hint}")
        print("  Hint: Use --results_from <training_run_dir> or --rgan_model <path_to_generator.pt>")

    # ========================================================================
    # STEP 5: Generate synthetic data (RGAN + TimeGAN)
    # ========================================================================
    t0 = time.time()
    print("\n[STEP 5] Generating synthetic data...")

    # --- 4a: RGAN synthetic data (with stochastic diversity) ---
    rng = np.random.default_rng(42)
    if has_rgan and G is not None:
        print("  Generating RGAN synthetic data (with dropout noise for diversity)...")
        n_synth = len(X_train)
        n_runs = 5  # Multiple forward passes for diversity

        # Move generator to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        G.to(device)
        G.train()  # Enable dropout for stochastic generation

        # Process each run in GPU-friendly chunks to avoid OOM
        # Keep X on CPU, stream chunks to GPU
        X_np = X_train  # (N, L, 1) numpy
        chunk_size = 2048  # Safe for A10G 24GB with LSTM hidden states
        Y_rgan_runs = []

        with torch.no_grad(), torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            for run_i in range(n_runs):
                parts = []
                for start in range(0, n_synth, chunk_size):
                    end = min(start + chunk_size, n_synth)
                    xb = torch.from_numpy(X_np[start:end]).float().to(device)
                    noise = torch.randn_like(xb) * 0.02
                    parts.append(G(xb + noise).cpu().numpy())
                Y_rgan_runs.append(np.concatenate(parts, axis=0))

        G.eval()
        G.cpu()  # Free GPU memory for later training
        torch.cuda.empty_cache()

        # Pick one run per sample (round-robin) for maximum diversity — vectorized
        Y_synthetic = np.empty_like(Y_rgan_runs[0])
        run_indices = np.arange(n_synth) % n_runs
        for r in range(n_runs):
            mask = run_indices == r
            Y_synthetic[mask] = Y_rgan_runs[r][mask]

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
    t_tgan = time.time()
    print("  Training TimeGAN for synthetic data comparison...")
    if args.skip_timegan:
        print("  Skipping TimeGAN (--skip_timegan flag set)")
        has_timegan = False
        X_timegan = None
        Y_timegan = None
    else:
        try:
            # Scale epochs down for large datasets to keep runtime reasonable
            n_train = len(X_train)
            tgan_epochs = args.timegan_epochs
            tgan_batch = min(256, n_train)
            print(f"  TimeGAN config: epochs={tgan_epochs}/phase, batch_size={tgan_batch}, samples={n_train}")

            timegan_result = train_timegan(
                X_train,  # (n_samples, L, 1)
                hidden_dim=24, latent_dim=24, n_layers=1,
                epochs_ae=tgan_epochs, epochs_sup=tgan_epochs, epochs_joint=tgan_epochs,
                batch_size=tgan_batch,
                lr=1e-3, device='auto',
            )
            X_timegan = timegan_result["synthetic_data"].astype(np.float32)
            if has_rgan and G is not None:
                G.eval()
                with torch.no_grad():
                    Y_timegan = G(torch.from_numpy(X_timegan).float()).numpy()
            else:
                Y_timegan = None
            has_timegan = True
            print(f"  TimeGAN: generated {len(X_timegan)} synthetic sequences in {_fmt_elapsed(time.time() - t_tgan)}")
        except Exception as e:
            print(f"  TimeGAN training failed: {e}")
            import traceback; traceback.print_exc()
            has_timegan = False
            X_timegan = None
            Y_timegan = None

    print(f"  Step 5 took {_fmt_elapsed(time.time() - t0)}")

    # ========================================================================
    # STEP 6: Create mixed dataset
    # ========================================================================
    print("\n[STEP 6] Creating mixed real + synthetic dataset...")

    X_mixed = np.concatenate([X_train, X_train], axis=0)
    Y_mixed = np.concatenate([Y_train, Y_synthetic], axis=0)

    print(f"  Mixed dataset: X shape {X_mixed.shape}, Y shape {Y_mixed.shape}")

    # ========================================================================
    # STEP 7: Train classical models on both datasets
    # ========================================================================
    t0 = time.time()
    print("\n[STEP 7] Training models...")

    augmentation_results = {}

    # 1. Naive Baseline
    print("  Evaluating Naive Baseline...")
    try:
        naive_stats, _ = naive_baseline(X_test, Y_test)
        augmentation_results['Naive'] = {
            'real_only': naive_stats,
            'real_plus_synthetic': naive_stats  # Same
        }
        print(f"    Naive RMSE: {naive_stats['rmse']:.6f}")
    except Exception as e:
        print(f"    Naive failed: {e}")

    # 2-4. Classical models in parallel (CPU-bound, release GIL)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_classical(name, forecast_fn, X_tr, Y_tr, X_mix, Y_mix, X_te, Y_te):
        """Train classical model on real & mixed, return results dict."""
        real_stats, real_preds = forecast_fn(X_tr, Y_tr, X_eval=X_te, Y_eval=Y_te)
        mixed_stats, mixed_preds = forecast_fn(X_mix, Y_mix, X_eval=X_te, Y_eval=Y_te)
        real_metrics = error_stats(Y_te, real_preds)
        mixed_metrics = error_stats(Y_te, mixed_preds)
        return name, real_metrics, mixed_metrics

    print("  Training ARIMA, ARMA, Tree Ensemble in parallel...")
    classical_tasks = [
        ("ARIMA", arima_forecast),
        ("ARMA", arma_forecast),
        ("Tree_Ensemble", tree_ensemble_forecast),
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _run_classical, name, fn,
                X_train, Y_train, X_mixed, Y_mixed, X_test, Y_test
            ): name for name, fn in classical_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, real_m, mixed_m = future.result()
                augmentation_results[name] = {
                    'real_only': real_m,
                    'real_plus_synthetic': mixed_m,
                }
                print(f"    {name} RMSE - Real only: {real_m['rmse']:.6f}, Mixed: {mixed_m['rmse']:.6f}")
            except Exception as e:
                print(f"    {name} failed: {e}")

    # 5. Neural models (LSTM, FITS, PatchTST, iTransformer)
    # These are the models that actually benefit from more training data
    print("  Training neural models (real vs augmented)...")

    # Create validation splits
    # Use real-only data for validation in BOTH conditions so early stopping
    # is based on identical real-data signal — the only variable is training data.
    val_idx = int(len(X_train) * 0.9)
    Xtr_sub, Ytr_sub = X_train[:val_idx], Y_train[:val_idx]
    Xval_sub, Yval_sub = X_train[val_idx:], Y_train[val_idx:]

    # Mixed training: real train + synthetic train, but validate on real-only
    # Shuffle to prevent the model from seeing all real then all synthetic
    n_mixed = len(X_mixed)
    shuffle_idx = np.random.RandomState(42).permutation(n_mixed)
    X_mixed_shuffled = X_mixed[shuffle_idx]
    Y_mixed_shuffled = Y_mixed[shuffle_idx]
    # Use same real-only validation split for fair early stopping comparison
    Xtr_mixed_sub = X_mixed_shuffled
    Ytr_mixed_sub = Y_mixed_shuffled

    nn_config = TrainConfig(
        L=args.L, H=args.H,
        epochs=args.nn_epochs,
        batch_size=256,
        units_g=128,
        units_d=128,
        g_layers=2,
        d_layers=2,
        dropout=0.1,
        lr_g=5e-4,
        lr_d=5e-4,
        grad_clip=1.0,
        patience=args.nn_patience,
        device='auto',
        eval_every=5,
        amp=True,
    )

    real_splits = {"Xtr": Xtr_sub, "Ytr": Ytr_sub, "Xval": Xval_sub, "Yval": Yval_sub,
                   "Xte": X_test, "Yte": Y_test}
    mixed_splits = {"Xtr": Xtr_mixed_sub, "Ytr": Ytr_mixed_sub,
                    "Xval": Xval_sub, "Yval": Yval_sub,
                    "Xte": X_test, "Yte": Y_test}

    # Helper to run a model on real vs mixed and record results
    def _run_augmentation_test(name, train_fn, real_sp, mixed_sp, **kwargs):
        t_model = time.time()
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
            print(f"    {name} took {_fmt_elapsed(time.time() - t_model)}")
        except Exception as e:
            print(f"    {name} failed after {_fmt_elapsed(time.time() - t_model)}: {e}")
            import traceback; traceback.print_exc()

    _run_augmentation_test("LSTM", train_lstm_supervised_torch, real_splits, mixed_splits)
    _run_augmentation_test("FITS", train_fits, real_splits, mixed_splits)
    _run_augmentation_test("PatchTST", train_patchtst, real_splits, mixed_splits)
    _run_augmentation_test("iTransformer", train_itransformer, real_splits, mixed_splits)

    print(f"  Step 7 took {_fmt_elapsed(time.time() - t0)}")

    # ========================================================================
    # STEP 8: Compute synthetic quality metrics (RGAN vs TimeGAN)
    # ========================================================================
    t0 = time.time()
    print("\n[STEP 8] Computing synthetic quality metrics...")

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
        if Y_timegan is not None:
            # Compare in forecast (Y) space for consistency with RGAN metrics
            synthetic_quality['TimeGAN'] = _compute_quality("TimeGAN", Y_train, Y_timegan)
        else:
            # Fallback to input (X) space if no RGAN generator to produce Y_timegan
            synthetic_quality['TimeGAN'] = _compute_quality("TimeGAN (X-space)", X_train, X_timegan)

    # ========================================================================
    # STEP 9: Create classification metrics from synthetic quality metrics
    # ========================================================================
    print("\n[STEP 9] Classification metrics ready from synthetic quality analysis...")

    # Classification metrics are the discrimination scores from synthetic quality
    # This shows how well a classifier can distinguish real from synthetic data
    classification_results = disc_scores

    # ========================================================================
    # STEP 10: Create tables
    # ========================================================================
    print("\n[STEP 10] Creating comparison tables...")

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
    # STEP 11: Create visualizations
    # ========================================================================
    print("\n[STEP 11] Creating visualizations...")

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
    # STEP 12: Classification & Model Comparison (Supervisor Tables)
    # ========================================================================
    print("\n[STEP 12] Generating Classification & Model Comparison Tables...")

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
            if isinstance(wgan_checkpoint, dict) and 'state_dict' in wgan_checkpoint:
                wgan_checkpoint = wgan_checkpoint['state_dict']
            from rgan.models_torch import build_generator
            G_wgan = build_generator(args.L, args.H, n_in=1, units=g_units,
                                     num_layers=g_layers, layer_norm=True)
            wgan_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            G_wgan.load_state_dict(wgan_checkpoint)
            G_wgan.to(wgan_device).eval()
            with torch.no_grad(), torch.amp.autocast(wgan_device.type, enabled=(wgan_device.type == 'cuda')):
                X_torch = torch.from_numpy(X_train).float().to(wgan_device)
                Y_wgan = G_wgan(X_torch).float().cpu().numpy()
            G_wgan.cpu()
            syn_datasets['WGAN-GP'] = Y_wgan
            print(f"  Generated {len(Y_wgan)} WGAN sequences.")
        except Exception as e:
            print(f"  Failed to load/run WGAN: {e}")

    if has_timegan and Y_timegan is not None:
        syn_datasets['TimeGAN'] = Y_timegan

    # Generate Table 2: Quantitative Quality (GAN vs WGAN)
    # Reuse metrics already computed in Step 8 where possible
    if syn_datasets:
        print("\n  Generating Table 2: GAN vs WGAN Quality Metrics...")
        quality_results = []
        for name, Y_syn in syn_datasets.items():
            if name in synthetic_quality:
                # Reuse cached metrics from Step 8
                cached = synthetic_quality[name]
                fd_val = cached['frechet_distance']
                var_val = cached['variance_difference']['abs_diff']
                disc_acc = cached['all_discrimination']['Random Forest']['accuracy']
                print(f"    {name}: reusing cached metrics from Step 8")
            else:
                print(f"    Computing metrics for {name}...")
                fd_val = frechet_distance(Y_train, Y_syn)
                var_val = variance_difference(Y_train, Y_syn)['abs_diff']
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
    # STEP 13: Create metrics JSON
    # ========================================================================
    print("\n[STEP 13] Creating metrics JSON...")

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

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    metrics_path = results_dir / 'metrics_augmentation.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved: {metrics_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_elapsed = time.time() - t_total_start
    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETE — total time: {_fmt_elapsed(total_elapsed)}")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nGenerated files:")
    for f in sorted(results_dir.glob('*')):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.0f} KB)")


def _build_parser():
    parser = argparse.ArgumentParser(description='Run synthetic data augmentation experiments')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--time_col', type=str, default=None,
                       help='Name of time column (auto-detected if not specified)')
    parser.add_argument('--target_col', type=str, default=None,
                       help='Name of target column (auto-detected if not specified)')
    parser.add_argument('--L', type=int, default=60,
                       help='Lookback window size (auto-overridden from metrics.json if --results_from is set)')
    parser.add_argument('--H', type=int, default=12,
                       help='Forecast horizon (auto-overridden from metrics.json if --results_from is set)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/test split ratio')
    parser.add_argument('--results_dir', type=str, default='results/augmentation',
                       help='Directory to save results')
    parser.add_argument('--rgan_model', type=str, default=None,
                       help='Path to RGAN generator .pt file (auto-detected if not set)')
    parser.add_argument('--wgan_model', type=str, default=None,
                       help='Path to WGAN model checkpoint (for Supervisor Comparison)')
    parser.add_argument('--results_from', type=str, default=None,
                       help='Path to training results directory (e.g. results/cloud/results). '
                            'Auto-finds generator, metrics.json, and uses correct L/H/architecture.')
    parser.add_argument('--skip_timegan', action='store_true',
                       help='Skip TimeGAN training (saves significant time)')
    parser.add_argument('--timegan_epochs', type=int, default=100,
                       help='Epochs per TimeGAN training phase (3 phases: AE, supervised, joint)')
    parser.add_argument('--nn_epochs', type=int, default=200,
                       help='Max epochs for neural model augmentation training (early stopping applies)')
    parser.add_argument('--nn_patience', type=int, default=25,
                       help='Early stopping patience for neural models')
    return parser


def cli_main():
    args = _build_parser().parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
