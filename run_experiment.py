#!/usr/bin/env python3
import os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

from src.mit_rgan.data import load_csv_series, interpolate_and_standardize, make_windows_univariate, make_windows_with_covariates
from src.mit_rgan.models_keras import build_generator, build_discriminator
from src.mit_rgan.rgan_keras import train_rgan_keras, compute_rmse_mae
from src.mit_rgan.lstm_supervised import train_lstm_supervised
from src.mit_rgan.baselines import naive_baseline, classical_curves_vs_samples
from src.mit_rgan.plots import plot_single_train_test, plot_compare_models_bars, plot_classical_curves
from src.mit_rgan.tune import tune_rgan_keras

import tensorflow as tf
tf.keras.backend.set_floatx("float32")

def set_seed(seed=42):
    np.random.seed(seed); tf.random.set_seed(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="auto")
    ap.add_argument("--time_col", default="auto")
    ap.add_argument("--resample", default="")
    ap.add_argument("--agg", default="last")
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lambda_reg", type=float, default=0.1)
    ap.add_argument("--units_g", type=int, default=64)
    ap.add_argument("--units_d", type=int, default=64)
    ap.add_argument("--lrG", type=float, default=1e-3)
    ap.add_argument("--lrD", type=float, default=1e-3)
    ap.add_argument("--label_smooth", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--tune", default="true")
    ap.add_argument("--tune_csv", default="")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    df, target_col, time_used = load_csv_series(args.csv, args.target, args.time_col, args.resample, args.agg)
    prep = interpolate_and_standardize(df, target_col, train_ratio=args.train_ratio)

    if prep["covariates"]:
        Xfull_tr, Yfull_tr = make_windows_with_covariates(prep["scaled_train"], target_col, prep["covariates"], args.L, args.H)
        Xte, Yte = make_windows_with_covariates(prep["scaled_test"], target_col, prep["covariates"], args.L, args.H)
    else:
        Xfull_tr, Yfull_tr = make_windows_univariate(prep["scaled_train"], target_col, args.L, args.H)
        Xte, Yte = make_windows_univariate(prep["scaled_test"], target_col, args.L, args.H)

    n_tr = len(Xfull_tr); n_val = max(1, int(0.1*n_tr))
    Xtr, Ytr = Xfull_tr[:-n_val], Yfull_tr[:-n_val]
    Xval, Yval = Xfull_tr[-n_val:], Yfull_tr[-n_val:]

    base_config = dict(
        L=args.L, H=args.H, epochs=args.epochs, batch_size=args.batch_size,
        lambda_reg=args.lambda_reg, units_g=args.units_g, units_d=args.units_d,
        lrG=args.lrG, lrD=args.lrD, label_smooth=args.label_smooth, grad_clip=args.grad_clip,
        dropout=args.dropout, patience=args.patience
    )

    used_tune = ""
    if args.tune.lower() == "true":
        if args.tune_csv:
            df_tune, target_tune, _ = load_csv_series(args.tune_csv, args.target, args.time_col, args.resample, args.agg)
            prep_t = interpolate_and_standardize(df_tune, target_tune, train_ratio=args.train_ratio)
            if prep_t["covariates"]:
                Xtune, Ytune = make_windows_with_covariates(prep_t["scaled_train"], target_tune, prep_t["covariates"], args.L, args.H)
            else:
                Xtune, Ytune = make_windows_univariate(prep_t["scaled_train"], target_tune, args.L, args.H)
            used_tune = args.tune_csv
        else:
            Xtune, Ytune = Xfull_tr, Yfull_tr
            used_tune = args.csv
        hp_grid = {"units_g": [32, 64, 128], "units_d": [32, 64, 128], "lambda_reg": [0.05, 0.1, 0.2], "epochs_each": 30}
        best_hp, df_tune_res = tune_rgan_keras(hp_grid, base_config, {"Xtr_full": Xtune, "Ytr_full": Ytune}, str(results_dir))
        df_tune_res.to_csv(results_dir/"tuning_results.csv", index=False)
        base_config.update({k: v for k, v in best_hp.items() if k in ["units_g","units_d","lambda_reg"]})

    G = build_generator(args.L, args.H, n_in=Xtr.shape[-1], units=base_config["units_g"], dropout=base_config["dropout"])
    D = build_discriminator(args.L, args.H, units=base_config["units_d"], dropout=base_config["dropout"])
    rgan_out = train_rgan_keras(base_config, (G,D), {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte}, str(results_dir), tag="rgan")

    lstm_out = train_lstm_supervised(base_config, {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte}, str(results_dir), tag="lstm")

    plot_single_train_test(rgan_out["history"]["epoch"], rgan_out["history"]["train_rmse"], rgan_out["history"]["test_rmse"],
                           "R-GAN: Error vs Epochs", results_dir/"rgan_train_test_rmse_vs_epochs.png")
    plot_single_train_test(lstm_out["history"]["epoch"], lstm_out["history"]["train_rmse"], lstm_out["history"]["test_rmse"],
                           "LSTM (Supervised): Error vs Epochs", results_dir/"lstm_train_test_rmse_vs_epochs.png")

    naive_test_rmse, naive_test_mae = naive_baseline(Xte, Yte)
    naive_train_rmse, naive_train_mae = naive_baseline(Xtr, Ytr)

    sizes, ets_curve, arima_curve = classical_curves_vs_samples(prep["train_df"][target_col].values, prep["test_df"][target_col].values, min_frac=0.3, steps=6)
    class_curve_path = plot_classical_curves(sizes, ets_curve, arima_curve, results_dir/"classical_error_vs_samples.png") if sizes is not None else None
    ets_rmse_full = float(np.nan if ets_curve is None else ets_curve[-1])
    arima_rmse_full = float(np.nan if arima_curve is None else arima_curve[-1])

    # Noise robustness
    noise_sd = 0.05
    Xte_noisy = Xte + np.random.normal(0, noise_sd, size=Xte.shape).astype(Xte.dtype)
    rgan_noisy_rmse, rgan_noisy_mae, _ = compute_rmse_mae(rgan_out["G"], Xte_noisy, Yte)
    lstm_noisy_pred = lstm_out["model"].predict(Xte_noisy, verbose=0)
    lstm_noisy_rmse = float(np.sqrt(np.mean((lstm_noisy_pred.reshape(-1)-Yte.reshape(-1))**2)))
    lstm_noisy_mae  = float(np.mean(np.abs(lstm_noisy_pred.reshape(-1)-Yte.reshape(-1))))

    test_errors = {"R-GAN": rgan_out["test_rmse"], "LSTM": lstm_out["test_rmse"], "Naive": naive_test_rmse}
    train_errors = {"R-GAN": rgan_out["train_rmse"], "LSTM": lstm_out["train_rmse"], "Naive": naive_train_rmse}
    compare_test = results_dir/"models_test_error.png"; compare_train = results_dir/"models_train_error.png"
    plot_compare_models_bars(train_errors, test_errors, compare_test, compare_train)

    metrics = dict(
        dataset=args.csv, tuning_dataset=used_tune, time_col_used=time_used, target_col=target_col,
        L=args.L, H=args.H, train_size=int(prep["split"]), test_size=len(prep["test_df"]),
        num_train_windows=int(Xfull_tr.shape[0]), num_test_windows=int(Xte.shape[0]),
        rgan=dict(train_rmse=rgan_out["train_rmse"], test_rmse=rgan_out["test_rmse"],
                  train_mae=rgan_out["train_mae"], test_mae=rgan_out["test_mae"],
                  rmse_noisy=rgan_noisy_rmse, mae_noisy=rgan_noisy_mae,
                  curve=str(results_dir/"rgan_train_test_rmse_vs_epochs.png"), history=rgan_out["history"]),
        lstm=dict(train_rmse=lstm_out["train_rmse"], test_rmse=lstm_out["test_rmse"],
                  train_mae=lstm_out["train_mae"], test_mae=lstm_out["test_mae"],
                  rmse_noisy=lstm_noisy_rmse, mae_noisy=lstm_noisy_mae,
                  curve=str(results_dir/"lstm_train_test_rmse_vs_epochs.png"), history=lstm_out["history"]),
        naive=dict(train_rmse=naive_train_rmse, test_rmse=naive_test_rmse,
                   train_mae=naive_train_mae, test_mae=naive_test_mae),
        compare_plots=dict(test=str(compare_test), train=str(compare_train)),
        classical=dict(ets_rmse_full=ets_rmse_full, arima_rmse_full=arima_rmse_full, curves=str(class_curve_path) if class_curve_path else ""),
        created=datetime.utcnow().isoformat()
    )
    with open(results_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    # make src importable if run directly
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
    main()
