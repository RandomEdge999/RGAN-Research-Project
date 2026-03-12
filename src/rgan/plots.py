from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover - optional dependency
    go = None
    make_subplots = None
    pio = None
    _HAS_PLOTLY = False


_PALETTE = [
    "#6366F1",
    "#EC4899",
    "#22D3EE",
    "#F59E0B",
    "#10B981",
    "#8B5CF6",
    "#F97316",
]


def _ensure_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _finalise_static(fig, out_path: Path) -> str:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out_path)


def _write_interactive(fig, out_path: Path) -> str:
    if not _HAS_PLOTLY or fig is None:
        return ""
    html_path = out_path.with_suffix(".html")
    pio.write_html(fig, file=str(html_path), include_plotlyjs="cdn", full_html=True, auto_open=False)
    return str(html_path)


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.35)
    ax.set_facecolor("#f8fafc")
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)


def plot_single_train_test(curve_epoch, train_rmse, test_rmse, title, out_path, ylabel="RMSE") -> Dict[str, str]:
    out_path = _ensure_path(out_path)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(curve_epoch, train_rmse, label="Train RMSE", color=_PALETTE[0], linewidth=2.2, marker="o")
    ax.plot(curve_epoch, test_rmse, label="Test RMSE", color=_PALETTE[1], linewidth=2.2, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    _style_axes(ax)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        fig_i.add_trace(
            go.Scatter(
                x=list(curve_epoch),
                y=list(train_rmse),
                name="Train RMSE",
                mode="lines+markers",
                line=dict(color=_PALETTE[0], width=3),
                marker=dict(size=8),
            )
        )
        fig_i.add_trace(
            go.Scatter(
                x=list(curve_epoch),
                y=list(test_rmse),
                name="Test RMSE",
                mode="lines+markers",
                line=dict(color=_PALETTE[1], width=3),
                marker=dict(size=8),
            )
        )
        fig_i.update_layout(
            title=dict(text=title, x=0.05),
            xaxis=dict(title="Epochs"),
            yaxis=dict(title=ylabel),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=60, r=30, t=70, b=60),
        )
        fig_i.update_traces(hovertemplate="Epoch %{x}<br>%{y:.6f}")
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


def plot_constant_train_test(train_value, test_value, title, out_path, ylabel="RMSE") -> Dict[str, str]:
    out_path = _ensure_path(out_path)
    steps = [0, 1]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(steps, [train_value, train_value], label=f"Train {ylabel}", color=_PALETTE[0], linewidth=2.4)
    ax.plot(steps, [test_value, test_value], label=f"Test {ylabel}", color=_PALETTE[1], linewidth=2.4)
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(steps, ["Start", "End"])
    ax.legend(frameon=False)
    _style_axes(ax)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        fig_i.add_trace(
            go.Scatter(
                x=steps,
                y=[train_value, train_value],
                name=f"Train {ylabel}",
                mode="lines",
                line=dict(color=_PALETTE[0], width=4),
            )
        )
        fig_i.add_trace(
            go.Scatter(
                x=steps,
                y=[test_value, test_value],
                name=f"Test {ylabel}",
                mode="lines",
                line=dict(color=_PALETTE[1], width=4),
            )
        )
        fig_i.update_layout(
            title=dict(text=title, x=0.05),
            xaxis=dict(title="Epochs", tickmode="array", tickvals=steps, ticktext=["Start", "End"]),
            yaxis=dict(title=ylabel),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=60, r=30, t=70, b=60),
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


def plot_compare_models_bars(train_errors: dict, test_errors: dict, out_test_path: str, out_train_path: str) -> Dict[str, Dict[str, str]]:
    models = list(test_errors.keys())
    test_vals = [test_errors[m] for m in models]
    train_vals = [train_errors[m] for m in models]

    # Static plots
    fig_test, ax_test = plt.subplots(figsize=(7.4, 4.4))
    bars_test = ax_test.bar(models, test_vals, color=_PALETTE[: len(models)])
    ax_test.set_title("Test Error by Model")
    ax_test.set_xlabel("Models")
    ax_test.set_ylabel("RMSE")
    for bar, value in zip(bars_test, test_vals):
        ax_test.annotate(f"{value:.6f}", xy=(bar.get_x() + bar.get_width() / 2, value), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=9, color="#1f2937")
    _style_axes(ax_test)
    static_test = _finalise_static(fig_test, _ensure_path(out_test_path))

    fig_train, ax_train = plt.subplots(figsize=(7.4, 4.4))
    bars_train = ax_train.bar(models, train_vals, color=_PALETTE[: len(models)])
    ax_train.set_title("Train Error by Model")
    ax_train.set_xlabel("Models")
    ax_train.set_ylabel("RMSE")
    for bar, value in zip(bars_train, train_vals):
        ax_train.annotate(f"{value:.6f}", xy=(bar.get_x() + bar.get_width() / 2, value), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=9, color="#1f2937")
    _style_axes(ax_train)
    static_train = _finalise_static(fig_train, _ensure_path(out_train_path))

    # Interactive plots
    html_test = html_train = ""
    if _HAS_PLOTLY:
        fig_i_test = go.Figure()
        fig_i_test.add_trace(
            go.Bar(x=models, y=test_vals, marker_color=_PALETTE[: len(models)], name="Test RMSE")
        )
        fig_i_test.update_layout(
            title="Test Error by Model",
            xaxis_title="Models",
            yaxis_title="RMSE",
            template="plotly_white",
            margin=dict(l=60, r=30, t=70, b=60),
        )
        html_test = _write_interactive(fig_i_test, _ensure_path(out_test_path))

        fig_i_train = go.Figure()
        fig_i_train.add_trace(
            go.Bar(x=models, y=train_vals, marker_color=_PALETTE[: len(models)], name="Train RMSE")
        )
        fig_i_train.update_layout(
            title="Train Error by Model",
            xaxis_title="Models",
            yaxis_title="RMSE",
            template="plotly_white",
            margin=dict(l=60, r=30, t=70, b=60),
        )
        html_train = _write_interactive(fig_i_train, _ensure_path(out_train_path))

    return {
        "test": {"static": static_test, "interactive": html_test},
        "train": {"static": static_train, "interactive": html_train},
    }


def plot_classical_curves(sizes, ets_curve, arima_curve, out_path) -> Optional[Dict[str, str]]:
    if sizes is None:
        return None

    out_path = _ensure_path(out_path)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(sizes, ets_curve, marker="o", label="ETS (RMSE)", color=_PALETTE[0], linewidth=2.2)
    ax.plot(sizes, arima_curve, marker="o", label="ARIMA (RMSE)", color=_PALETTE[1], linewidth=2.2)
    ax.set_title("Classical Models: Error vs Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=False)
    _style_axes(ax)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        fig_i.add_trace(
            go.Scatter(
                x=list(sizes),
                y=list(ets_curve),
                name="ETS (RMSE)",
                mode="lines+markers",
                line=dict(color=_PALETTE[0], width=3),
                marker=dict(size=8),
            )
        )
        fig_i.add_trace(
            go.Scatter(
                x=list(sizes),
                y=list(arima_curve),
                name="ARIMA (RMSE)",
                mode="lines+markers",
                line=dict(color=_PALETTE[1], width=3),
                marker=dict(size=8),
            )
        )
        fig_i.update_layout(
            title="Classical Models: Error vs Number of Samples",
            xaxis_title="Number of Samples",
            yaxis_title="RMSE",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=60, r=30, t=70, b=60),
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


def plot_learning_curves(sizes, curves: dict, out_path, curve_stds: dict = None, ylabel="RMSE") -> Optional[Dict[str, str]]:
    if sizes is None or len(sizes) == 0:
        return None

    out_path = _ensure_path(out_path)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for idx, (label, values) in enumerate(curves.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(sizes, values, marker="o", label=label, color=color, linewidth=2.2)
        if curve_stds and label in curve_stds:
            std_vals = curve_stds[label]
            if len(std_vals) == len(values):
                upper = np.array(values) + np.array(std_vals)
                lower = np.array(values) - np.array(std_vals)
                ax.fill_between(sizes, lower, upper, color=color, alpha=0.15)
    ax.set_title("Model RMSE vs Training Sample Size")
    ax.set_xlabel("Number of Training Windows")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    _style_axes(ax)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        for idx, (label, values) in enumerate(curves.items()):
            color = _PALETTE[idx % len(_PALETTE)]
            fig_i.add_trace(
                go.Scatter(
                    x=list(sizes),
                    y=list(values),
                    name=label,
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                )
            )
            if curve_stds and label in curve_stds:
                std_vals = curve_stds[label]
                if len(std_vals) == len(values):
                    upper = np.array(values) + np.array(std_vals)
                    lower = np.array(values) - np.array(std_vals)
                    fig_i.add_trace(
                        go.Scatter(
                            x=list(sizes) + list(reversed(list(sizes))),
                            y=list(upper) + list(reversed(list(lower))),
                            fill="toself",
                            fillcolor=_rgba(color, 0.18),
                            line=dict(color="rgba(0,0,0,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{label} ±1σ",
                        )
                    )
        fig_i.update_layout(
            title="Model RMSE vs Training Sample Size",
            xaxis_title="Number of Training Windows",
            yaxis_title=ylabel,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=60, r=30, t=70, b=60),
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


def create_error_metrics_table(model_results: dict, out_path: str = None):
    data = []

    def _extract(stats: dict, key: str):
        if stats is None:
            return None
        return stats.get(f"{key}_orig", stats.get(key))

    def _fmt(value):
        return "-" if value is None else f"{value:.8f}"

    def _fmt_ci(stats: dict, key: str) -> str:
        low = stats.get(f"{key}_orig_ci_low", stats.get(f"{key}_ci_low"))
        high = stats.get(f"{key}_orig_ci_high", stats.get(f"{key}_ci_high"))
        if low is None or high is None:
            std = stats.get(f"{key}_orig_std", stats.get(f"{key}_std"))
            if std is None:
                return "-"
            return f"±{std:.8f}"
        return f"[{low:.8f}, {high:.8f}]"

    for model_name, results in model_results.items():
        if "train" in results and "test" in results:
            data.append(
                {
                    "Model": f"{model_name} (Train)",
                    "RMSE": _fmt(_extract(results["train"], "rmse")),
                    "RMSE_CI": _fmt_ci(results["train"], "rmse"),
                    "MSE": _fmt(_extract(results["train"], "mse")),
                    "BIAS": _fmt(_extract(results["train"], "bias")),
                    "MAE": _fmt(_extract(results["train"], "mae")),
                    "MAE_CI": _fmt_ci(results["train"], "mae"),
                }
            )
            data.append(
                {
                    "Model": f"{model_name} (Test)",
                    "RMSE": _fmt(_extract(results["test"], "rmse")),
                    "RMSE_CI": _fmt_ci(results["test"], "rmse"),
                    "MSE": _fmt(_extract(results["test"], "mse")),
                    "BIAS": _fmt(_extract(results["test"], "bias")),
                    "MAE": _fmt(_extract(results["test"], "mae")),
                    "MAE_CI": _fmt_ci(results["test"], "mae"),
                }
            )
    df = pd.DataFrame(data)
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Error metrics table saved to: {out_path}")
    return df


def plot_predictions(predictions_dict: Dict[str, np.ndarray], save_path: str, n_samples: int = 4):
    """
    Plot ground truth vs model predictions for a few samples.
    
    Args:
        predictions_dict: Dictionary {model_name: predictions_array}
                          predictions_array shape: (N, H, 1) or (N, H)
        save_path: Path to save the plot (without extension)
        n_samples: Number of samples to plot
    """
    if not predictions_dict:
        return

    # keys = list(predictions_dict.keys())
    # num_models = len(keys)
    
    # Get ground truth if available for reference
    truth = predictions_dict.get('True')
    
    # Check available samples
    available_samples = 0
    for k, v in predictions_dict.items():
        if v is not None:
            available_samples = len(v)
            break
            
    n_samples = min(n_samples, available_samples)
    if n_samples <= 0:
        return

    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        for model_name, preds in predictions_dict.items():
            if preds is None:
                continue
            
            # Handle shapes (N, H, 1) or (N, H)
            try:
                if len(preds) <= i: continue
                y = preds[i].flatten()
            except:
                continue
            
            style = '-'
            alpha = 0.7
            width = 1.5
            color = None
            
            if model_name == 'True':
                style = 'k-'
                alpha = 1.0
                width = 2.0
                label = 'Ground Truth'
            else:
                label = model_name
                
            ax.plot(y, style, label=label, alpha=alpha, linewidth=width)
            
        ax.set_title(f"Sample {i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    _finalise_static(fig, Path(save_path))
    
    # Interactive plot if possible
    if _HAS_PLOTLY:
        try:
            fig_go = make_subplots(rows=n_samples, cols=1, subplot_titles=[f"Sample {i}" for i in range(n_samples)])
            
            for i in range(n_samples):
                for model_name, preds in predictions_dict.items():
                    if preds is None:
                        continue
                    try:
                        if len(preds) <= i: continue
                        y = preds[i].flatten()
                    except: continue

                    line_dict = dict(width=2)
                    if model_name == 'True':
                        line_dict['color'] = 'black'
                        line_dict['dash'] = 'solid'
                    
                    fig_go.add_trace(
                        go.Scatter(y=y, mode='lines', name=model_name if i == 0 else None,
                                  line=line_dict, showlegend=(i==0)),
                        row=i+1, col=1
                    )
            
            fig_go.update_layout(height=300*n_samples, title_text="Predictions Comparison")
            _write_interactive(fig_go, Path(save_path))
        except Exception:
            pass


def create_noise_robustness_table(
    noise_results: List[dict],
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build a noise robustness table: RMSE per model at each noise level.

    Args:
        noise_results: List of dicts from run_training, one per noise level.
            Each dict has keys like 'sd', 'rgan', 'lstm', ... where each model
            value is a stats dict with 'rmse_orig' (or 'rmse').
        out_path: Optional CSV path.

    Returns:
        DataFrame with models as rows, noise levels as columns, values = RMSE.
    """
    model_keys = [
        ("rgan", "RGAN"),
        ("lstm", "LSTM"),
        ("dlinear", "DLinear"),
        ("nlinear", "NLinear"),
        ("fits", "FITS"),
        ("patchtst", "PatchTST"),
        ("itransformer", "iTransformer"),
        ("naive_baseline", "Naive"),
        ("arima", "ARIMA"),
        ("arma", "ARMA"),
        ("tree_ensemble", "Tree Ensemble"),
    ]

    sds = [r["sd"] for r in noise_results]

    rows = []
    for key, label in model_keys:
        row = {"Model": label}
        baseline_rmse = None
        for nr in noise_results:
            sd = nr["sd"]
            stats = nr.get(key, {})
            if not stats:
                row[f"σ={sd}"] = None
                continue
            rmse = stats.get("rmse_orig", stats.get("rmse"))
            row[f"σ={sd}"] = rmse
            if sd == 0.0 and rmse is not None:
                baseline_rmse = rmse

        # Compute degradation (% increase from clean to noisiest)
        if baseline_rmse and len(sds) > 1:
            noisiest = noise_results[-1]
            noisy_stats = noisiest.get(key, {})
            noisy_rmse = noisy_stats.get("rmse_orig", noisy_stats.get("rmse"))
            if noisy_rmse is not None and baseline_rmse > 0:
                row["Degradation %"] = f"{(noisy_rmse - baseline_rmse) / baseline_rmse * 100:+.1f}%"
            else:
                row["Degradation %"] = "-"
        else:
            row["Degradation %"] = "-"

        rows.append(row)

    df = pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, index=False)
    return df


def plot_noise_robustness(
    noise_results: List[dict],
    out_path: str,
) -> Dict[str, str]:
    """Plot RMSE vs noise level for all models — the core paper figure.

    Args:
        noise_results: List of dicts from run_training.
        out_path: Save path (without extension).

    Returns:
        Dict with 'static' and 'interactive' paths.
    """
    model_keys = [
        ("rgan", "RGAN", "#E74C3C", "-", "o"),
        ("lstm", "LSTM", "#3498DB", "--", "s"),
        ("dlinear", "DLinear", "#2ECC71", "-.", "^"),
        ("nlinear", "NLinear", "#9B59B6", ":", "d"),
        ("fits", "FITS", "#F39C12", "-.", "v"),
        ("patchtst", "PatchTST", "#1ABC9C", "--", "P"),
        ("itransformer", "iTransformer", "#E67E22", "-", "X"),
        ("naive_baseline", "Naive", "#95A5A6", ":", "*"),
    ]

    sds = [r["sd"] for r in noise_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, label, color, ls, marker in model_keys:
        rmses = []
        for nr in noise_results:
            stats = nr.get(key, {})
            if not stats:
                rmses.append(np.nan)
                continue
            rmse = stats.get("rmse_orig", stats.get("rmse"))
            rmses.append(rmse if rmse is not None else np.nan)

        if all(np.isnan(v) for v in rmses):
            continue

        ax.plot(sds, rmses, label=label, color=color, linestyle=ls,
                marker=marker, markersize=6, linewidth=2)

    ax.set_xlabel("Noise Standard Deviation (σ)", fontsize=12)
    ax.set_ylabel("Test RMSE (original scale)", fontsize=12)
    ax.set_title("Noise Robustness: RMSE Degradation Under Input Perturbation", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.005)
    plt.tight_layout()

    out = Path(out_path)
    static_path = _finalise_static(fig, out)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        for key, label, color, ls, marker in model_keys:
            rmses = []
            for nr in noise_results:
                stats = nr.get(key, {})
                if not stats:
                    rmses.append(None)
                    continue
                rmse = stats.get("rmse_orig", stats.get("rmse"))
                rmses.append(rmse)
            if all(v is None for v in rmses):
                continue
            dash_map = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}
            fig_i.add_trace(go.Scatter(
                x=sds, y=rmses, name=label,
                line=dict(color=color, dash=dash_map.get(ls, "solid"), width=2),
                mode="lines+markers",
            ))
        fig_i.update_layout(
            title="Noise Robustness: RMSE Degradation Under Input Perturbation",
            xaxis_title="Noise Standard Deviation (σ)",
            yaxis_title="Test RMSE (original scale)",
            template="plotly_white",
            height=500, width=900,
        )
        html_path = _write_interactive(fig_i, out)

    return {"static": static_path, "interactive": html_path}

