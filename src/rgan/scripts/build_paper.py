#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def fmt_value(val):
    try:
        return f"{float(val):.8f}"
    except (TypeError, ValueError):
        return str(val)


def to_itemize(lines):
    if not lines:
        return "\\item (none)"
    return "\n".join(f"\\item {line}" for line in lines)


def hyperparam_summary(cfg):
    dense_act = cfg.get("g_dense") or "linear"
    disc_act = cfg.get("d_activation") or "sigmoid"
    return (
        f"{cfg['g_layers']} generator LSTM layer(s) with {cfg['units_g']} units "
        f"(dense activation {dense_act}). "
        f"{cfg['d_layers']} discriminator LSTM layer(s) with {cfg['units_d']} units "
        f"(activation {disc_act}). Regularisation: $\\lambda={fmt_value(cfg['lambda_reg'])}$, "
        f"dropout={fmt_value(cfg['dropout'])}, $\\texttt{{lrG}}={fmt_value(cfg['lrG'])}$, $\\texttt{{lrD}}={fmt_value(cfg['lrD'])}$."
    )


def build_error_rows(metrics):
    rows = []

    def _get(stats: dict, key: str):
        if not stats:
            return None
        return stats.get(f"{key}_orig", stats.get(key))

    for key, label in [
        ("rgan", "R-GAN"),
        ("lstm", "LSTM"),
        ("naive_baseline", "Naïve Baseline"),
        ("tree_ensemble", "Tree Ensemble"),
        ("arima", "ARIMA"),
        ("arma", "ARMA"),
    ]:
        if key not in metrics:
            continue
        train = metrics[key].get("train", {})
        test = metrics[key].get("test", {})
        row = "{} & {} & {} & {} & {} & {} & {} \\\\".format(
            label,
            fmt_value(_get(train, "rmse")),
            fmt_value(_get(test, "rmse")),
            fmt_value(_get(train, "mse")),
            fmt_value(_get(test, "mse")),
            fmt_value(_get(train, "bias")),
            fmt_value(_get(test, "bias")),
        )
        rows.append(row)
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="research_paper.tex")
    ap.add_argument("--template", default="", help="Optional path to a LaTeX template file.")
    args = ap.parse_args()
    metrics_path = Path(args.metrics)
    m = json.loads(metrics_path.read_text())

    # template.tex lives in papers/ at the project root
    _project_root = Path(__file__).resolve().parents[3]
    template = Path(args.template) if args.template else (_project_root / "papers" / "template.tex")
    if not template.exists():
        print(f"[build_paper] ERROR: LaTeX template not found at {template}", file=sys.stderr)
        print("[build_paper] Create papers/template.tex or pass --template <path>.", file=sys.stderr)
        sys.exit(1)

    # Support both current schema ("charts") and legacy ("learning_curves"/"compare_plots")
    charts = m.get("charts", {})
    compare_plots = m.get("compare_plots", {})
    learning_curves = m.get("learning_curves", {})

    # Resolve learning curve: current schema uses charts.training_curves_overlay
    learning_curve_plot = (
        charts.get("training_curves_overlay", "")
        or learning_curves.get("plot", "")
    )

    def _check(path_str: str) -> str:
        if not path_str:
            return ""
        path_obj = Path(path_str)
        candidates = [path_obj]
        if not path_obj.is_absolute():
            candidates.append(metrics_path.parent / path_obj.name)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        print(f"[build_paper] Warning: referenced figure not found: {path_obj}", file=sys.stderr)
        return ""

    # Build substitution dict supporting both schemas
    ranked_bars = charts.get("ranked_model_bars", "")
    overlay = charts.get("training_curves_overlay", "")

    filled = template.read_text() % dict(
        rgan_curve=_check(m.get("rgan", {}).get("curve", "")),
        lstm_curve=_check(m.get("lstm", {}).get("curve", "")),
        naive_curve=_check(m.get("naive_baseline", {}).get("curve", "")),
        tree_ensemble_curve=_check(m.get("tree_ensemble", {}).get("curve", "")),
        compare_test=_check(ranked_bars or compare_plots.get("test", "")),
        compare_train=_check(ranked_bars or compare_plots.get("train", "")),
        naive_comparison=_check(ranked_bars or compare_plots.get("naive_comparison", "")),
        classical_curve=_check(overlay),
        learning_curve=_check(learning_curve_plot),
        generator_arch=to_itemize(m.get("rgan", {}).get("architecture", {}).get("generator", [])),
        discriminator_arch=to_itemize(m.get("rgan", {}).get("architecture", {}).get("discriminator", [])),
        rgan_hparams=hyperparam_summary(m.get("rgan", {}).get("config", {})),
        error_table_rows=build_error_rows(m),
    )

    Path(args.out).write_text(filled)
    print(f"Wrote LaTeX to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
