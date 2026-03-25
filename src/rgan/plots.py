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


# ---------------------------------------------------------------------------
# Unified model style registry — consistent colors, linestyles, markers
# across all charts. Kept in display-order (GAN first, neural, classical).
# ---------------------------------------------------------------------------
_MODEL_STYLES = {
    "RGAN":           {"color": "#E74C3C", "ls": "-",  "marker": "o"},
    "LSTM":           {"color": "#3498DB", "ls": "--", "marker": "s"},
    "DLinear":        {"color": "#2ECC71", "ls": "-.", "marker": "^"},
    "NLinear":        {"color": "#9B59B6", "ls": ":",  "marker": "d"},
    "FITS":           {"color": "#F39C12", "ls": "-.", "marker": "v"},
    "PatchTST":       {"color": "#1ABC9C", "ls": "--", "marker": "P"},
    "iTransformer":   {"color": "#E67E22", "ls": "-",  "marker": "X"},
    "Autoformer":     {"color": "#C0392B", "ls": "--", "marker": "H"},
    "Informer":       {"color": "#2C3E50", "ls": "-.", "marker": "p"},
    "Naive":          {"color": "#95A5A6", "ls": ":",  "marker": "*"},
    "ARIMA":          {"color": "#7F8C8D", "ls": "--", "marker": ">"},
    "ARMA":           {"color": "#BDC3C7", "ls": "-.", "marker": "<"},
    "Tree Ensemble":  {"color": "#27AE60", "ls": ":",  "marker": "D"},
    "Naïve Baseline": {"color": "#95A5A6", "ls": ":",  "marker": "*"},
}

# Flat palette derived from styles (for bar charts etc.)
_PALETTE = [s["color"] for s in _MODEL_STYLES.values()]

# Map from noise_results dict keys → display names
_NOISE_KEY_MAP = {
    "rgan": "RGAN", "lstm": "LSTM", "dlinear": "DLinear", "nlinear": "NLinear",
    "fits": "FITS", "patchtst": "PatchTST", "itransformer": "iTransformer",
    "autoformer": "Autoformer", "informer": "Informer", "naive_baseline": "Naive",
    "arima": "ARIMA", "arma": "ARMA", "tree_ensemble": "Tree Ensemble",
}

_DASH_MAP = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(png_path)


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


def _get_style(model_name: str) -> dict:
    """Get style for a model, with fallback for unknown models."""
    return _MODEL_STYLES.get(model_name, {"color": "#333333", "ls": "-", "marker": "o"})


# ---------------------------------------------------------------------------
# 1. Training Curves Overlay (replaces 13 individual per-model charts)
# ---------------------------------------------------------------------------
def plot_training_curves_overlay(
    model_histories: Dict[str, Dict[str, list]],
    classical_baselines: Dict[str, float],
    out_path: str,
    metric: str = "test_rmse",
    ylabel: str = "Test RMSE",
) -> Dict[str, str]:
    """All neural model training curves on one chart, classical baselines as hlines.

    Args:
        model_histories: {model_name: {"epoch": [...], "train_rmse": [...], "test_rmse": [...]}}.
        classical_baselines: {model_name: final_test_rmse}.
        out_path: Save path (without extension).
        metric: Key in history dict to plot (default "test_rmse").
        ylabel: Y-axis label.
    """
    out_path = _ensure_path(out_path)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Neural model curves
    for name, hist in model_histories.items():
        epochs = hist.get("epoch", [])
        values = hist.get(metric, [])
        if not epochs or not values:
            continue
        s = _get_style(name)
        ax.plot(epochs, values, label=name, color=s["color"], linestyle=s["ls"],
                marker=s["marker"], markersize=4, linewidth=1.8, markevery=max(1, len(epochs) // 10))

    # Classical baselines as horizontal dashed lines
    for name, val in classical_baselines.items():
        if val is None:
            continue
        s = _get_style(name)
        ax.axhline(y=val, color=s["color"], linestyle=":", linewidth=1.2, alpha=0.7)
        ax.annotate(f"{name} ({val:.6f})", xy=(1.01, val), xycoords=("axes fraction", "data"),
                    fontsize=7, color=s["color"], va="center")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Training Convergence: All Models", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    _style_axes(ax)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        for name, hist in model_histories.items():
            epochs = hist.get("epoch", [])
            values = hist.get(metric, [])
            if not epochs or not values:
                continue
            s = _get_style(name)
            fig_i.add_trace(go.Scatter(
                x=list(epochs), y=list(values), name=name, mode="lines+markers",
                line=dict(color=s["color"], dash=_DASH_MAP.get(s["ls"], "solid"), width=2),
                marker=dict(size=4),
            ))
        for name, val in classical_baselines.items():
            if val is None:
                continue
            s = _get_style(name)
            fig_i.add_hline(y=val, line_dash="dot", line_color=s["color"],
                            annotation_text=f"{name} ({val:.6f})",
                            annotation_position="right")
        fig_i.update_layout(
            title="Training Convergence: All Models",
            xaxis_title="Epoch", yaxis_title=ylabel,
            template="plotly_white", height=550, width=1000,
            legend=dict(font=dict(size=9)),
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


# ---------------------------------------------------------------------------
# 2. Ranked Model Bars with CI (replaces plot_compare_models_bars)
# ---------------------------------------------------------------------------
def plot_ranked_model_bars(
    model_stats: Dict[str, Dict[str, float]],
    out_path: str,
    metric: str = "rmse_orig",
    title: str = "Model Comparison: Test RMSE (95% CI)",
) -> Dict[str, str]:
    """Horizontal bar chart sorted by metric, with bootstrap CI error bars.

    Args:
        model_stats: {model_name: {"rmse_orig": float, "rmse_orig_ci_low": float, ...}}.
        out_path: Save path (without extension).
        metric: Metric key to sort and display.
        title: Chart title.
    """
    out_path = _ensure_path(out_path)

    # Build sorted data
    items = []
    for name, stats in model_stats.items():
        val = stats.get(metric)
        if val is None:
            continue
        ci_low = stats.get(f"{metric}_ci_low")
        ci_high = stats.get(f"{metric}_ci_high")
        items.append((name, val, ci_low, ci_high))

    items.sort(key=lambda x: x[1])  # ascending (best at top in horizontal)

    names = [it[0] for it in items]
    values = [it[1] for it in items]
    colors = [_get_style(n)["color"] for n in names]

    # Compute error bar arrays (asymmetric)
    xerr_low = []
    xerr_high = []
    has_ci = False
    for _, val, ci_low, ci_high in items:
        if ci_low is not None and ci_high is not None:
            xerr_low.append(val - ci_low)
            xerr_high.append(ci_high - val)
            has_ci = True
        else:
            xerr_low.append(0)
            xerr_high.append(0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(items) * 0.45)))
    y_pos = np.arange(len(items))

    if has_ci:
        ax.barh(y_pos, values, color=colors, xerr=[xerr_low, xerr_high],
                capsize=3, ecolor="#555555", height=0.6)
    else:
        ax.barh(y_pos, values, color=colors, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(metric.replace("_orig", "").upper(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.invert_yaxis()  # Best at top

    # Value annotations
    for i, (_, val, _, _) in enumerate(items):
        ax.annotate(f"{val:.6f}", xy=(val, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#333")

    _style_axes(ax)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        fig_i.add_trace(go.Bar(
            y=names, x=values, orientation="h",
            marker_color=colors,
            error_x=dict(type="data", symmetric=False,
                         array=xerr_high, arrayminus=xerr_low) if has_ci else None,
            text=[f"{v:.6f}" for v in values],
            textposition="outside",
        ))
        fig_i.update_layout(
            title=title, xaxis_title=metric.replace("_orig", "").upper(),
            template="plotly_white", height=max(400, len(items) * 40), width=900,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=120, r=80, t=70, b=50),
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


# ---------------------------------------------------------------------------
# 3. Noise Robustness Heatmap (NEW)
# ---------------------------------------------------------------------------
def plot_noise_robustness_heatmap(
    noise_results: List[dict],
    out_path: str,
    normalize: bool = True,
) -> Dict[str, str]:
    """Heatmap: models (rows) x noise levels (cols), color = degradation % or raw RMSE.

    Args:
        noise_results: List of dicts from run_training.
        out_path: Save path (without extension).
        normalize: If True, show % degradation from clean baseline.
    """
    out_path = _ensure_path(out_path)
    sds = [r["sd"] for r in noise_results]

    # Build matrix
    model_names = []
    matrix = []
    for key, label in _NOISE_KEY_MAP.items():
        row = []
        baseline = None
        for nr in noise_results:
            stats = nr.get(key, {})
            rmse = stats.get("rmse_orig", stats.get("rmse")) if stats else None
            if nr["sd"] == 0.0 and rmse is not None:
                baseline = rmse
            row.append(rmse)

        if all(v is None for v in row):
            continue

        if normalize and baseline and baseline > 0:
            row = [(((v - baseline) / baseline) * 100 if v is not None else np.nan) for v in row]
        else:
            row = [(v if v is not None else np.nan) for v in row]

        model_names.append(label)
        matrix.append(row)

    matrix = np.array(matrix)

    # Sort by degradation at highest noise level (best robustness at top)
    sort_col = matrix[:, -1] if matrix.shape[1] > 0 else matrix[:, 0]
    sort_idx = np.argsort(sort_col)
    matrix = matrix[sort_idx]
    model_names = [model_names[i] for i in sort_idx]

    # Static
    fig, ax = plt.subplots(figsize=(max(8, len(sds) * 1.5), max(5, len(model_names) * 0.45)))
    cmap = "YlOrRd" if normalize else "viridis"
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(sds)))
    ax.set_xticklabels([f"σ={s}" for s in sds], fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    # Cell annotations
    for i in range(len(model_names)):
        for j in range(len(sds)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            fmt = f"{val:+.0f}%" if normalize else f"{val:.4f}"
            text_color = "white" if val > np.nanpercentile(matrix, 70) else "black"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8, color=text_color)

    label = "RMSE Degradation (%)" if normalize else "Test RMSE"
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_title("Noise Robustness Heatmap", fontsize=13)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fmt = ".0f" if normalize else ".4f"
        suffix = "%" if normalize else ""
        fig_i = go.Figure(data=go.Heatmap(
            z=matrix.tolist(),
            x=[f"σ={s}" for s in sds],
            y=model_names,
            colorscale="YlOrRd" if normalize else "Viridis",
            text=[[f"{v:{fmt}}{suffix}" if not np.isnan(v) else "" for v in row] for row in matrix],
            texttemplate="%{text}",
            colorbar=dict(title=label),
        ))
        fig_i.update_layout(
            title="Noise Robustness Heatmap",
            template="plotly_white", height=max(400, len(model_names) * 40), width=700,
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


# ---------------------------------------------------------------------------
# 4. Multi-Metric Radar Chart (NEW)
# ---------------------------------------------------------------------------
def plot_multi_metric_radar(
    model_stats: Dict[str, Dict[str, float]],
    out_path: str,
    metrics: List[str] = None,
    models: List[str] = None,
) -> Dict[str, str]:
    """Radar chart comparing models across multiple metrics.

    Args:
        model_stats: {model_name: {metric_key: value}}.
        out_path: Save path (without extension).
        metrics: Metric keys to plot. Default: rmse_orig, mae_orig, smape_orig, maape_orig, mase_orig.
        models: Model names to include. Default: top 5 by RMSE + RGAN.
    """
    out_path = _ensure_path(out_path)

    if metrics is None:
        metrics = ["rmse_orig", "mae_orig", "smape_orig", "maape_orig", "mase_orig"]

    metric_labels = [m.replace("_orig", "").upper() for m in metrics]

    # Select models
    if models is None:
        # Top 5 by RMSE + ensure RGAN is included
        sorted_models = sorted(
            [(n, s.get("rmse_orig", float("inf"))) for n, s in model_stats.items()],
            key=lambda x: x[1]
        )
        top_names = [n for n, _ in sorted_models[:5]]
        if "RGAN" not in top_names and "RGAN" in model_stats:
            top_names.append("RGAN")
        models = top_names

    # Build normalized values (lower is better → invert so larger area = better)
    raw = {}
    for name in models:
        stats = model_stats.get(name, {})
        raw[name] = [stats.get(m) for m in metrics]

    # Find min/max per metric for normalization
    all_vals = {m: [] for m in metrics}
    for name, vals in raw.items():
        for i, v in enumerate(vals):
            if v is not None:
                all_vals[metrics[i]].append(v)

    mins = {m: min(vs) if vs else 0 for m, vs in all_vals.items()}
    maxs = {m: max(vs) if vs else 1 for m, vs in all_vals.items()}

    normalized = {}
    for name, vals in raw.items():
        norm = []
        for i, v in enumerate(vals):
            m = metrics[i]
            rng = maxs[m] - mins[m]
            if v is None or rng == 0:
                norm.append(0.5)
            else:
                norm.append(1.0 - (v - mins[m]) / rng)  # invert: higher = better
        normalized[name] = norm

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name, vals in normalized.items():
        s = _get_style(name)
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, color=s["color"], linewidth=2, label=name)
        ax.fill(angles, vals_closed, color=s["color"], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["worst", "", "", "best"], fontsize=8, color="#666")
    ax.set_title("Multi-Metric Model Comparison", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    static_path = _finalise_static(fig, out_path)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        for name, vals in normalized.items():
            s = _get_style(name)
            fig_i.add_trace(go.Scatterpolar(
                r=vals + vals[:1],
                theta=metric_labels + metric_labels[:1],
                name=name,
                line=dict(color=s["color"], width=2),
                fill="toself",
                fillcolor=_rgba(s["color"], 0.1),
            ))
        fig_i.update_layout(
            title="Multi-Metric Model Comparison",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white", height=600, width=700,
        )
        html_path = _write_interactive(fig_i, out_path)

    return {"static": static_path, "interactive": html_path}


# ---------------------------------------------------------------------------
# 5. Predictions Comparison (KEPT as-is)
# ---------------------------------------------------------------------------
def plot_predictions(predictions_dict: Dict[str, np.ndarray], save_path: str, n_samples: int = 4):
    """Plot ground truth vs model predictions for a few samples."""
    if not predictions_dict:
        return

    available_samples = 0
    for k, v in predictions_dict.items():
        if v is not None:
            available_samples = len(v)
            break

    n_samples = min(n_samples, available_samples)
    if n_samples <= 0:
        return

    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]
        for model_name, preds in predictions_dict.items():
            if preds is None:
                continue
            try:
                if len(preds) <= i:
                    continue
                y = preds[i].flatten()
            except Exception:
                continue

            if model_name == "True":
                ax.plot(y, "k-", label="Ground Truth", alpha=1.0, linewidth=2.0)
            else:
                s = _get_style(model_name)
                ax.plot(y, label=model_name, color=s["color"], alpha=0.7, linewidth=1.5)

        ax.set_title(f"Sample {i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=3)

    plt.tight_layout()
    _finalise_static(fig, Path(save_path))

    if _HAS_PLOTLY:
        try:
            fig_go = make_subplots(rows=n_samples, cols=1,
                                   subplot_titles=[f"Sample {i}" for i in range(n_samples)])
            for i in range(n_samples):
                for model_name, preds in predictions_dict.items():
                    if preds is None:
                        continue
                    try:
                        if len(preds) <= i:
                            continue
                        y = preds[i].flatten()
                    except Exception:
                        continue

                    line_dict = dict(width=2)
                    if model_name == "True":
                        line_dict["color"] = "black"
                    else:
                        s = _get_style(model_name)
                        line_dict["color"] = s["color"]

                    fig_go.add_trace(
                        go.Scatter(y=y, mode="lines",
                                   name=model_name if i == 0 else None,
                                   line=line_dict, showlegend=(i == 0)),
                        row=i + 1, col=1
                    )
            fig_go.update_layout(height=300 * n_samples, title_text="Predictions Comparison")
            _write_interactive(fig_go, Path(save_path))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 6. Error Metrics Table (KEPT as-is)
# ---------------------------------------------------------------------------
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
            return f"\u00b1{std:.8f}"
        return f"[{low:.8f}, {high:.8f}]"

    for model_name, results in model_results.items():
        if "train" in results and "test" in results:
            data.append({
                "Model": f"{model_name} (Train)",
                "RMSE": _fmt(_extract(results["train"], "rmse")),
                "RMSE_CI": _fmt_ci(results["train"], "rmse"),
                "MSE": _fmt(_extract(results["train"], "mse")),
                "BIAS": _fmt(_extract(results["train"], "bias")),
                "MAE": _fmt(_extract(results["train"], "mae")),
                "MAE_CI": _fmt_ci(results["train"], "mae"),
            })
            data.append({
                "Model": f"{model_name} (Test)",
                "RMSE": _fmt(_extract(results["test"], "rmse")),
                "RMSE_CI": _fmt_ci(results["test"], "rmse"),
                "MSE": _fmt(_extract(results["test"], "mse")),
                "BIAS": _fmt(_extract(results["test"], "bias")),
                "MAE": _fmt(_extract(results["test"], "mae")),
                "MAE_CI": _fmt_ci(results["test"], "mae"),
            })
    df = pd.DataFrame(data)
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Error metrics table saved to: {out_path}")
    return df


# ---------------------------------------------------------------------------
# 7. Noise Robustness Table (FIXED — added Autoformer, Informer)
# ---------------------------------------------------------------------------
def create_noise_robustness_table(
    noise_results: List[dict],
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build a noise robustness table: RMSE per model at each noise level."""
    model_keys = list(_NOISE_KEY_MAP.items())  # all 13 models

    sds = [r["sd"] for r in noise_results]

    rows = []
    for key, label in model_keys:
        row = {"Model": label}
        baseline_rmse = None
        for nr in noise_results:
            sd = nr["sd"]
            stats = nr.get(key, {})
            if not stats:
                row[f"\u03c3={sd}"] = None
                continue
            rmse = stats.get("rmse_orig", stats.get("rmse"))
            row[f"\u03c3={sd}"] = rmse
            if sd == 0.0 and rmse is not None:
                baseline_rmse = rmse

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


# ---------------------------------------------------------------------------
# 8. Noise Robustness Line Plot (FIXED — all 13 models via _MODEL_STYLES)
# ---------------------------------------------------------------------------
def plot_noise_robustness(
    noise_results: List[dict],
    out_path: str,
) -> Dict[str, str]:
    """Plot RMSE vs noise level for all models — the core paper figure."""
    sds = [r["sd"] for r in noise_results]
    out = _ensure_path(out_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, label in _NOISE_KEY_MAP.items():
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

        s = _MODEL_STYLES.get(label, {"color": "#333", "ls": "-", "marker": "o"})
        ax.plot(sds, rmses, label=label, color=s["color"], linestyle=s["ls"],
                marker=s["marker"], markersize=6, linewidth=2)

    ax.set_xlabel("Noise Standard Deviation (\u03c3)", fontsize=12)
    ax.set_ylabel("Test RMSE (original scale)", fontsize=12)
    ax.set_title("Noise Robustness: RMSE Degradation Under Input Perturbation", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.005)
    plt.tight_layout()
    static_path = _finalise_static(fig, out)

    html_path = ""
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        for key, label in _NOISE_KEY_MAP.items():
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
            s = _MODEL_STYLES.get(label, {"color": "#333", "ls": "-"})
            fig_i.add_trace(go.Scatter(
                x=sds, y=rmses, name=label,
                line=dict(color=s["color"], dash=_DASH_MAP.get(s["ls"], "solid"), width=2),
                mode="lines+markers",
            ))
        fig_i.update_layout(
            title="Noise Robustness: RMSE Degradation Under Input Perturbation",
            xaxis_title="Noise Standard Deviation (\u03c3)",
            yaxis_title="Test RMSE (original scale)",
            template="plotly_white",
            height=500, width=900,
        )
        html_path = _write_interactive(fig_i, out)

    return {"static": static_path, "interactive": html_path}
