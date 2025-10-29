#!/usr/bin/env python3
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.mit_rgan.baselines import naive_baseline, naive_bayes_forecast, classical_curves_vs_samples
from src.mit_rgan.data import (
    load_csv_series,
    interpolate_and_standardize,
    make_windows_univariate,
    make_windows_with_covariates,
)
from src.mit_rgan.plots import (
    plot_single_train_test,
    plot_constant_train_test,
    plot_compare_models_bars,
    plot_classical_curves,
    plot_learning_curves,
    create_error_metrics_table,
    plot_naive_bayes_comparison,
)
from src.mit_rgan.tune import tune_rgan_keras


def set_seed(seed=42, backend="tf"):
    random.seed(seed)
    np.random.seed(seed)
    if backend == "tf":
        import tensorflow as tf  # Lazy import to avoid TF initialisation when unused

        tf.random.set_seed(seed)
    elif backend == "torch":
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def describe_model(model) -> list:
    """Return a human-readable description of the model layers."""
    description = []
    if hasattr(model, "layers"):
        for layer in model.layers:
            name = layer.__class__.__name__
            if name == "InputLayer":
                shape = getattr(layer, "input_shape", None)
                if shape is not None:
                    description.append(f"Input(shape={tuple(shape[1:])})")
                else:
                    description.append("Input")
            elif name == "LSTM":
                cfg = layer.get_config()
                description.append(
                    f"LSTM(units={cfg['units']}, return_sequences={cfg['return_sequences']}, "
                    f"activation={cfg['activation']}, recurrent_activation={cfg['recurrent_activation']})"
                )
            elif name == "Dense":
                cfg = layer.get_config()
                description.append(f"Dense(units={cfg['units']}, activation={cfg.get('activation', 'linear')})")
            elif name == "Dropout":
                cfg = layer.get_config()
                description.append(f"Dropout(rate={cfg['rate']})")
            elif name == "Reshape":
                cfg = layer.get_config()
                description.append(f"Reshape(target_shape={tuple(cfg['target_shape'])})")
            else:
                description.append(name)
    elif hasattr(model, "named_children"):
        for name, module in model.named_children():
            description.append(f"{name}: {module.__class__.__name__}")
        if not description:
            description.append(model.__class__.__name__)
    else:
        description.append(model.__class__.__name__)
    return description


def error_stats(y_true, y_pred):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}


def compute_learning_curves(args, base_config, Xfull_tr, Yfull_tr, Xte, Yte, n_features):
    if args.curve_steps <= 0:
        return [], {}

    total = len(Xfull_tr)
    if total < 2:
        return [], {}

    min_size = max(int(args.curve_min_frac * total), 5)
    if args.curve_steps == 1:
        sizes = np.array([total], dtype=int)
    else:
        sizes = np.linspace(min_size, total, args.curve_steps, dtype=int)
    sizes = np.clip(sizes, 2, total)
    sizes = np.unique(sizes)

    curves = {"R-GAN": [], "LSTM": [], "Naïve Baseline": [], "Naïve Bayes": []}
    used_sizes = []
    naive_test_stats, _ = naive_baseline(Xte, Yte)

    if args.backend == "torch":
        from src.mit_rgan.models_torch import build_generator as build_generator_backend, build_discriminator as build_discriminator_backend
        from src.mit_rgan.rgan_torch import train_rgan_torch as train_rgan_backend
        from src.mit_rgan.lstm_supervised_torch import train_lstm_supervised_torch as train_lstm_backend
    else:
        from src.mit_rgan.models_keras import build_generator as build_generator_backend, build_discriminator as build_discriminator_backend
        from src.mit_rgan.rgan_keras import train_rgan_keras as train_rgan_backend
        from src.mit_rgan.lstm_supervised import train_lstm_supervised as train_lstm_backend

    for size in sizes:
        if size < 2:
            continue
        val_size = max(1, int(0.1 * size))
        if size - val_size < 1:
            continue

        Xsubset = Xfull_tr[:size]
        Ysubset = Yfull_tr[:size]
        Xtr_sub = Xsubset[:-val_size]
        Ytr_sub = Ysubset[:-val_size]
        Xval_sub = Xsubset[-val_size:]
        Yval_sub = Ysubset[-val_size:]

        data_splits = {
            "Xtr": Xtr_sub,
            "Ytr": Ytr_sub,
            "Xval": Xval_sub,
            "Yval": Yval_sub,
            "Xte": Xte,
            "Yte": Yte,
        }

        curve_config = dict(base_config)
        curve_config["epochs"] = max(1, min(base_config["epochs"], args.curve_epochs))
        curve_config["patience"] = min(curve_config["patience"], curve_config["epochs"])

        set_seed(args.seed, backend=args.backend)
        gen_kwargs = dict(
            L=base_config["L"],
            H=base_config["H"],
            n_in=n_features,
            units=curve_config["units_g"],
            dropout=curve_config["dropout"],
            num_layers=curve_config.get("g_layers", base_config["g_layers"]),
        )
        disc_kwargs = dict(
            L=base_config["L"],
            H=base_config["H"],
            units=curve_config["units_d"],
            dropout=curve_config["dropout"],
            num_layers=curve_config.get("d_layers", base_config["d_layers"]),
        )
        if args.backend == "tf":
            gen_kwargs.update(
                activation=args.g_activation,
                recurrent_activation=args.g_recurrent_activation,
                dense_activation=args.g_dense_activation if args.g_dense_activation else None,
            )
            disc_kwargs.update(
                activation=args.d_activation,
                recurrent_activation=args.d_recurrent_activation,
            )

        G_curve = build_generator_backend(**gen_kwargs)
        D_curve = build_discriminator_backend(**disc_kwargs)
        rgan_curve_out = train_rgan_backend(curve_config, (G_curve, D_curve), data_splits, str(args.results_dir), tag="rgan_curve")
        lstm_curve_out = train_lstm_backend(curve_config, data_splits, str(args.results_dir), tag="lstm_curve")

        curves["R-GAN"].append(rgan_curve_out["test_stats"]["rmse"])
        curves["LSTM"].append(lstm_curve_out["test_stats"]["rmse"])

        # Naïve baseline does not require training and stays constant across sizes
        curves["Naïve Baseline"].append(naive_test_stats["rmse"])
        naive_bayes_stats, _ = naive_bayes_forecast(Xtr_sub, Ytr_sub, Xte, Yte)
        curves["Naïve Bayes"].append(naive_bayes_stats["rmse"])
        used_sizes.append(int(size))

    return used_sizes, curves

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
    ap.add_argument("--g_layers", type=int, default=1)
    ap.add_argument("--d_layers", type=int, default=1)
    ap.add_argument("--g_activation", default="tanh")
    ap.add_argument("--g_recurrent_activation", default="sigmoid")
    ap.add_argument("--g_dense_activation", default="")
    ap.add_argument("--d_activation", default="tanh")
    ap.add_argument("--d_recurrent_activation", default="sigmoid")
    ap.add_argument("--lrG", type=float, default=1e-3)
    ap.add_argument("--lrD", type=float, default=1e-3)
    ap.add_argument("--label_smooth", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--curve_steps", type=int, default=0)
    ap.add_argument("--curve_min_frac", type=float, default=0.4)
    ap.add_argument("--curve_epochs", type=int, default=40)
    # Torch-specific runtime knobs (safe to ignore on TF backend)
    ap.add_argument("--amp", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Enable automatic mixed precision for torch backend")
    ap.add_argument("--eval_batch_size", type=int, default=512,
                    help="Evaluation batch size for torch backend")
    ap.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers (torch backend)")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="DataLoader prefetch_factor (torch backend)")
    ap.add_argument("--persistent_workers", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Use persistent workers for DataLoader (torch backend)")
    ap.add_argument("--pin_memory", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Pin host memory for faster H2D copies (torch backend)")
    ap.add_argument("--backend", choices=["tf","torch"], default="tf")
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Run the R-GAN hyperparameter sweep (disabled by default).",
    )
    ap.add_argument("--tune_csv", default="")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.tune_csv and not args.tune:
        args.tune = True

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "torch":
        try:
            from src.mit_rgan.models_torch import (
                build_generator as build_generator_backend,
                build_discriminator as build_discriminator_backend,
            )
            from src.mit_rgan.rgan_torch import (
                train_rgan_torch as train_rgan_backend,
                compute_metrics as compute_metrics_backend,
            )
            from src.mit_rgan.lstm_supervised_torch import (
                train_lstm_supervised_torch as train_lstm_backend,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ModuleNotFoundError(
                    "Torch backend selected but PyTorch is not installed. Please install the 'torch' package to continue."
                ) from exc
            raise
    else:
        import tensorflow as tf

        tf.keras.backend.set_floatx("float32")
        from src.mit_rgan.models_keras import (
            build_generator as build_generator_backend,
            build_discriminator as build_discriminator_backend,
        )
        from src.mit_rgan.rgan_keras import (
            train_rgan_keras as train_rgan_backend,
            compute_metrics as compute_metrics_backend,
        )
        from src.mit_rgan.lstm_supervised import (
            train_lstm_supervised as train_lstm_backend,
        )

    set_seed(args.seed, backend=args.backend)

    df, target_col, time_used = load_csv_series(args.csv, args.target, args.time_col, args.resample, args.agg)
    prep = interpolate_and_standardize(df, target_col, train_ratio=args.train_ratio)

    try:
        if prep["covariates"]:
            Xfull_tr, Yfull_tr = make_windows_with_covariates(
                prep["scaled_train"], target_col, prep["covariates"], args.L, args.H
            )
            Xte, Yte = make_windows_with_covariates(
                prep["scaled_test"], target_col, prep["covariates"], args.L, args.H
            )
        else:
            Xfull_tr, Yfull_tr = make_windows_univariate(
                prep["scaled_train"], target_col, args.L, args.H
            )
            Xte, Yte = make_windows_univariate(
                prep["scaled_test"], target_col, args.L, args.H
            )
    except ValueError as exc:
        raise ValueError(
            "Unable to construct training/test windows with the provided L/H settings. "
            "Consider decreasing --L/--H or ensuring the dataset has sufficient rows."
        ) from exc

    n_tr = len(Xfull_tr); n_val = max(1, int(0.1*n_tr))
    Xtr, Ytr = Xfull_tr[:-n_val], Yfull_tr[:-n_val]
    Xval, Yval = Xfull_tr[-n_val:], Yfull_tr[-n_val:]

    base_config = dict(
        L=args.L,
        H=args.H,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_reg=args.lambda_reg,
        units_g=args.units_g,
        units_d=args.units_d,
        lrG=args.lrG,
        lrD=args.lrD,
        label_smooth=args.label_smooth,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        patience=args.patience,
        g_layers=args.g_layers,
        d_layers=args.d_layers,
        # Torch runtime knobs
        amp=args.amp,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
    )

    used_tune = ""
    if args.tune:
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
        best_hp, df_tune_res = tune_rgan_keras(
            hp_grid,
            base_config,
            {"Xtr_full": Xtune, "Ytr_full": Ytune},
            str(results_dir),
            backend=args.backend,
        )
        df_tune_res.to_csv(results_dir/"tuning_results.csv", index=False)
        base_config.update({k: v for k, v in best_hp.items() if k in ["units_g","units_d","lambda_reg"]})

    g_dense_act = args.g_dense_activation if args.g_dense_activation else None

    gen_kwargs = dict(
        L=args.L,
        H=args.H,
        n_in=Xtr.shape[-1],
        units=base_config["units_g"],
        dropout=base_config["dropout"],
        num_layers=base_config["g_layers"],
    )
    disc_kwargs = dict(
        L=args.L,
        H=args.H,
        units=base_config["units_d"],
        dropout=base_config["dropout"],
        num_layers=base_config["d_layers"],
    )
    if args.backend == "tf":
        gen_kwargs.update(
            activation=args.g_activation,
            recurrent_activation=args.g_recurrent_activation,
            dense_activation=g_dense_act,
        )
        disc_kwargs.update(
            activation=args.d_activation,
            recurrent_activation=args.d_recurrent_activation,
        )

    G = build_generator_backend(**gen_kwargs)
    D = build_discriminator_backend(**disc_kwargs)
    rgan_out = train_rgan_backend(base_config, (G,D), {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte}, str(results_dir), tag="rgan")

    lstm_out = train_lstm_backend(base_config, {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte}, str(results_dir), tag="lstm")

    plot_single_train_test(rgan_out["history"]["epoch"], rgan_out["history"]["train_rmse"], rgan_out["history"]["test_rmse"],
                           "R-GAN: Error vs Epochs", results_dir/"rgan_train_test_rmse_vs_epochs.png")
    plot_single_train_test(lstm_out["history"]["epoch"], lstm_out["history"]["train_rmse"], lstm_out["history"]["test_rmse"],
                           "LSTM (Supervised): Error vs Epochs", results_dir/"lstm_train_test_rmse_vs_epochs.png")

    naive_test_stats, _ = naive_baseline(Xte, Yte)
    naive_train_stats, _ = naive_baseline(Xtr, Ytr)
    naive_curve_path = results_dir/"naive_train_test_rmse_vs_epochs.png"
    plot_constant_train_test(naive_train_stats["rmse"], naive_test_stats["rmse"],
                             "Naïve Baseline: Error vs Epochs", naive_curve_path)

    # Naïve Bayes implementation (train on training windows, evaluate on both)
    naive_bayes_train_stats, _ = naive_bayes_forecast(Xtr, Ytr)
    naive_bayes_test_stats, _ = naive_bayes_forecast(Xtr, Ytr, Xte, Yte)
    naive_bayes_curve_path = results_dir/"naive_bayes_train_test_rmse_vs_epochs.png"
    plot_constant_train_test(naive_bayes_train_stats["rmse"], naive_bayes_test_stats["rmse"],
                             "Naïve Bayes: Error vs Epochs", naive_bayes_curve_path)

    # Plot comparison between Naïve Baseline and Naïve Bayes (similar to Fig 1)
    naive_comparison_path = results_dir/"naive_baseline_vs_naive_bayes.png"
    plot_naive_bayes_comparison(naive_test_stats, naive_bayes_test_stats, naive_comparison_path)

    learning_sizes, learning_curve_values = compute_learning_curves(args, base_config, Xfull_tr, Yfull_tr, Xte, Yte, Xtr.shape[-1])
    learning_curve_path = None
    if learning_sizes:
        learning_curve_path = plot_learning_curves(learning_sizes, learning_curve_values, results_dir/"ml_error_vs_samples.png")

    sizes, ets_curve, arima_curve = classical_curves_vs_samples(prep["train_df"][target_col].values, prep["test_df"][target_col].values, min_frac=0.3, steps=6)
    class_curve_path = plot_classical_curves(sizes, ets_curve, arima_curve, results_dir/"classical_error_vs_samples.png") if sizes is not None else None
    ets_rmse_full = float(np.nan if ets_curve is None else ets_curve[-1])
    arima_rmse_full = float(np.nan if arima_curve is None else arima_curve[-1])

    # Noise robustness
    noise_sd = 0.05
    Xte_noisy = Xte + np.random.normal(0, noise_sd, size=Xte.shape).astype(Xte.dtype)
    rgan_noisy_stats, _ = (compute_metrics_backend(rgan_out["G"], Xte_noisy, Yte))
    if args.backend == "torch":
        import torch
        device = next(lstm_out["model"].parameters()).device
        with torch.no_grad():
            yp = lstm_out["model"](torch.from_numpy(Xte_noisy).to(device)).cpu().numpy()
        lstm_noisy_stats = error_stats(Yte.reshape(-1), yp.reshape(-1))
    else:
        lstm_noisy_pred = lstm_out["model"].predict(Xte_noisy, verbose=0)
        lstm_noisy_stats = error_stats(Yte.reshape(-1), lstm_noisy_pred.reshape(-1))

    test_errors = {"R-GAN": rgan_out["test_stats"]["rmse"], "LSTM": lstm_out["test_stats"]["rmse"], "Naïve Baseline": naive_test_stats["rmse"], "Naïve Bayes": naive_bayes_test_stats["rmse"]}
    train_errors = {"R-GAN": rgan_out["train_stats"]["rmse"], "LSTM": lstm_out["train_stats"]["rmse"], "Naïve Baseline": naive_train_stats["rmse"], "Naïve Bayes": naive_bayes_train_stats["rmse"]}
    compare_test = results_dir/"models_test_error.png"; compare_train = results_dir/"models_train_error.png"
    plot_compare_models_bars(train_errors, test_errors, compare_test, compare_train)

    # Create comprehensive error metrics table
    model_results = {
        "R-GAN": {
            "train": rgan_out["train_stats"],
            "test": rgan_out["test_stats"]
        },
        "LSTM": {
            "train": lstm_out["train_stats"],
            "test": lstm_out["test_stats"]
        },
        "Naïve Baseline": {
            "train": naive_train_stats,
            "test": naive_test_stats
        },
        "Naïve Bayes": {
            "train": naive_bayes_train_stats,
            "test": naive_bayes_test_stats
        }
    }
    
    # Generate and save error metrics table
    metrics_table = create_error_metrics_table(model_results, results_dir/"error_metrics_table.csv")
    print("\n" + "="*80)
    print("ERROR METRICS TABLE (RMSE, MSE, BIAS, MAE)")
    print("="*80)
    print(metrics_table.to_string(index=False))
    print("="*80)

    rgan_architecture = {
        "generator": describe_model(G),
        "discriminator": describe_model(D),
    }
    lstm_architecture = describe_model(lstm_out["model"])

    learning_curves_serializable = {k: [float(v) for v in vals] for k, vals in learning_curve_values.items()}

    metrics = dict(
        dataset=args.csv,
        tuning_dataset=used_tune,
        time_col_used=time_used,
        target_col=target_col,
        L=args.L,
        H=args.H,
        train_size=int(prep["split"]),
        test_size=len(prep["test_df"]),
        num_train_windows=int(Xfull_tr.shape[0]),
        num_test_windows=int(Xte.shape[0]),
        rgan=dict(
            train=rgan_out["train_stats"],
            test=rgan_out["test_stats"],
            noisy=rgan_noisy_stats,
            curve=str(results_dir/"rgan_train_test_rmse_vs_epochs.png"),
            history=rgan_out["history"],
            architecture=rgan_architecture,
            config=dict(
                units_g=base_config["units_g"],
                units_d=base_config["units_d"],
                lambda_reg=base_config["lambda_reg"],
                lrG=base_config["lrG"],
                lrD=base_config["lrD"],
                dropout=base_config["dropout"],
                g_layers=args.g_layers,
                d_layers=args.d_layers,
                g_activation=args.g_activation,
                g_recurrent=args.g_recurrent_activation,
                g_dense=g_dense_act if g_dense_act else "linear",
                d_activation=args.d_activation,
                d_recurrent=args.d_recurrent_activation,
            ),
        ),
        lstm=dict(
            train=lstm_out["train_stats"],
            test=lstm_out["test_stats"],
            noisy=lstm_noisy_stats,
            curve=str(results_dir/"lstm_train_test_rmse_vs_epochs.png"),
            history=lstm_out["history"],
            architecture=lstm_architecture,
            config=dict(units=base_config["units_g"], lr=base_config["lrG"], dropout=base_config["dropout"]),
        ),
        naive_baseline=dict(
            train=naive_train_stats,
            test=naive_test_stats,
            curve=str(naive_curve_path),
        ),
        naive_bayes=dict(
            train=naive_bayes_train_stats,
            test=naive_bayes_test_stats,
            curve=str(naive_bayes_curve_path),
        ),
        compare_plots=dict(test=str(compare_test), train=str(compare_train), naive_comparison=str(naive_comparison_path)),
        classical=dict(
            ets_rmse_full=ets_rmse_full,
            arima_rmse_full=arima_rmse_full,
            curves=str(class_curve_path) if class_curve_path else "",
        ),
        learning_curves=dict(
            sizes=[int(s) for s in learning_sizes],
            curves=learning_curves_serializable,
            plot=str(learning_curve_path) if learning_curve_path else "",
        ),
        created=datetime.utcnow().isoformat(),
    )
    with open(results_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    # make src importable if run directly
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
    main()
