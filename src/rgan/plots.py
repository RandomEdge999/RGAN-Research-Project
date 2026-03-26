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
    ax.legend(loc="upper right", fontsize=10, ncol=2, frameon=False)
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
    noise_results = sorted(noise_results, key=lambda r: r["sd"])
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
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=10)
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
            ax.legend(fontsize=10, ncol=3)

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
    sds = sorted(set(r["sd"] for r in noise_results))
    # Re-sort noise_results by sd to ensure consistent x-axis ordering
    noise_results = sorted(noise_results, key=lambda r: r["sd"])
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
    ax.legend(loc="upper left", fontsize=10, ncol=2)
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


# ===== Cross-Dataset Aggregation Charts (for multi-seed, multi-dataset papers) =====


def _rank_models(metrics: Dict[str, float]) -> Dict[str, int]:
    """Rank models by metric value (lower is better). Returns {model: rank}."""
    sorted_models = sorted(metrics.items(), key=lambda x: x[1])
    return {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}


def plot_noise_robustness_multi_dataset(
    all_noise: Dict[str, List[dict]],
    out_path: str,
) -> str:
    """Side-by-side noise robustness plots across datasets.

    Parameters
    ----------
    all_noise : dict
        {dataset_name: [noise_results_list]} where each noise_results_list
        is the same format as plot_noise_robustness receives.
    out_path : str
        Output file path for the combined figure.
    """
    n_datasets = len(all_noise)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, (ds_name, noise_results) in zip(axes, all_noise.items()):
        noise_results = sorted(noise_results, key=lambda r: r["sd"])
        sds = [r["sd"] for r in noise_results]

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
            ax.plot(sds, rmses, label=label, color=s["color"],
                    linestyle=s["ls"], linewidth=2, marker="o", markersize=5)

        ax.set_title(ds_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Noise SD (σ)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Test RMSE (original scale)", fontsize=11)

    axes[-1].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("Noise Robustness Across Datasets", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = _ensure_path(out_path)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_seed_boxplots(
    seed_metrics: Dict[str, Dict[str, List[float]]],
    metric_name: str,
    out_path: str,
) -> str:
    """Box plots showing metric variance across seeds for each model.

    Parameters
    ----------
    seed_metrics : dict
        {dataset_name: {model_name: [metric_per_seed]}}
    metric_name : str
        Name of the metric (e.g., "RMSE", "MAE") for axis label.
    out_path : str
        Output file path.
    """
    n_datasets = len(seed_metrics)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), sharey=False)
    if n_datasets == 1:
        axes = [axes]

    for ax, (ds_name, models) in zip(axes, seed_metrics.items()):
        model_names = sorted(models.keys())
        data = [models[m] for m in model_names]
        colors = [_MODEL_STYLES.get(m, {"color": "#999"})["color"] for m in model_names]

        bp = ax.boxplot(data, labels=model_names, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual seed points
        for i, (m, vals) in enumerate(zip(model_names, data)):
            jitter = [i + 1 + (v - 0.5) * 0.1 for v in [0.2, 0.4, 0.6, 0.8, 1.0][:len(vals)]]
            ax.scatter(jitter, vals, color="black", s=20, zorder=3, alpha=0.7)

        ax.set_title(ds_name, fontsize=13, fontweight="bold")
        ax.set_ylabel(f"Test {metric_name}", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{metric_name} Variance Across Seeds (5 seeds per model)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = _ensure_path(out_path)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def create_mean_std_table(
    seed_metrics: Dict[str, Dict[str, List[float]]],
    metric_name: str = "RMSE",
    out_path: Optional[str] = None,
) -> str:
    """Create a mean ± std summary table across seeds — the core paper table.

    Parameters
    ----------
    seed_metrics : dict
        {dataset_name: {model_name: [metric_per_seed]}}
    metric_name : str
        Name of metric for column headers.
    out_path : str or None
        If provided, saves as CSV. Always returns the formatted string.

    Returns
    -------
    str
        Formatted table string.
    """
    import numpy as np

    # Collect all model names across datasets
    all_models = sorted(set(
        m for ds in seed_metrics.values() for m in ds.keys()
    ))

    rows = []
    for model in all_models:
        row = {"Model": model}
        for ds_name, models in seed_metrics.items():
            vals = models.get(model, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[ds_name] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[ds_name] = "—"
        rows.append(row)

    # Format as aligned text table
    headers = ["Model"] + list(seed_metrics.keys())
    col_widths = {h: max(len(h), max(len(r.get(h, "")) for r in rows)) for h in headers}

    lines = []
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-+-".join("-" * col_widths[h] for h in headers))

    for row in rows:
        line = " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)
        lines.append(line)

    table_str = "\n".join(lines)

    if out_path:
        out = _ensure_path(out_path)
        # Save as CSV for easy import
        import csv
        with open(str(out), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    return table_str


def plot_ranking_stability(
    seed_metrics: Dict[str, Dict[str, List[float]]],
    out_path: str,
) -> str:
    """Bump chart showing model rankings across datasets.

    Parameters
    ----------
    seed_metrics : dict
        {dataset_name: {model_name: [metric_per_seed]}}
    out_path : str
        Output file path.
    """
    import numpy as np

    datasets = list(seed_metrics.keys())
    all_models = sorted(set(
        m for ds in seed_metrics.values() for m in ds.keys()
    ))

    # Compute mean metric per model per dataset, then rank
    rankings = {}  # {model: [rank_per_dataset]}
    for ds_name in datasets:
        means = {m: np.mean(seed_metrics[ds_name].get(m, [float("inf")])) for m in all_models}
        ranked = _rank_models(means)
        for m in all_models:
            rankings.setdefault(m, []).append(ranked.get(m, len(all_models)))

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(datasets)), 5))
    x_pos = list(range(len(datasets)))

    for model, ranks in rankings.items():
        s = _MODEL_STYLES.get(model, {"color": "#333", "ls": "-"})
        ax.plot(x_pos, ranks, label=model, color=s["color"],
                linestyle=s["ls"], linewidth=2.5, marker="o", markersize=8)
        # Label endpoint
        ax.annotate(model, (x_pos[-1], ranks[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=9, color=s["color"], va="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Rank (1 = best)", fontsize=12)
    ax.set_title("Model Ranking Stability Across Datasets", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Rank 1 at top
    ax.set_yticks(range(1, len(all_models) + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _ensure_path(out_path)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_clean_vs_noisy_rankings(
    all_noise: Dict[str, List[dict]],
    out_path: str,
    clean_sd: float = 0.0,
    noisy_sd: float = 0.2,
) -> str:
    """Compare model rankings under clean vs noisy conditions across datasets.

    Parameters
    ----------
    all_noise : dict
        {dataset_name: [noise_results_list]}
    out_path : str
        Output file path.
    clean_sd : float
        Noise level considered "clean".
    noisy_sd : float
        Noise level considered "noisy".
    """
    import numpy as np

    datasets = list(all_noise.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (condition, sd) in zip(axes, [("Clean", clean_sd), ("Noisy", noisy_sd)]):
        all_rankings = {}
        for ds_name in datasets:
            noise_results = all_noise[ds_name]
            target = [r for r in noise_results if abs(r["sd"] - sd) < 1e-6]
            if not target:
                continue
            nr = target[0]
            metrics = {}
            for key, label in _NOISE_KEY_MAP.items():
                stats = nr.get(key, {})
                if stats:
                    rmse = stats.get("rmse_orig", stats.get("rmse"))
                    if rmse is not None:
                        metrics[label] = rmse
            ranked = _rank_models(metrics)
            for m, rank in ranked.items():
                all_rankings.setdefault(m, []).append(rank)

        # Plot mean rank with error bars
        models = sorted(all_rankings.keys())
        means = [np.mean(all_rankings[m]) for m in models]
        stds = [np.std(all_rankings[m], ddof=1) if len(all_rankings[m]) > 1 else 0 for m in models]
        colors = [_MODEL_STYLES.get(m, {"color": "#999"})["color"] for m in models]

        bars = ax.barh(models, means, xerr=stds, color=colors, alpha=0.7,
                       edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_xlabel("Mean Rank (lower = better)", fontsize=11)
        ax.set_title(f"{condition} (σ={sd})", fontsize=13, fontweight="bold")
        ax.invert_xaxis()
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Clean vs Noisy Model Rankings", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = _ensure_path(out_path)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)
