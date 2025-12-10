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
    args = ap.parse_args()
    m = json.loads(Path(args.metrics).read_text())

    template = Path(__file__).parent / "template.tex"

    learning_curve_plot = m["learning_curves"].get("plot", "") or m["compare_plots"]["test"]

    def _check(path_str: str) -> str:
        if not path_str:
            return ""
        path_obj = Path(path_str)
        if not path_obj.exists():
            print(f"[build_paper] Warning: referenced figure not found: {path_obj}", file=sys.stderr)
            return ""
        return path_str

    filled = template.read_text() % dict(
        rgan_curve=_check(m["rgan"]["curve"]),
        lstm_curve=_check(m["lstm"]["curve"]),
        naive_curve=_check(m["naive_baseline"].get("curve") or m["rgan"]["curve"]),
        tree_ensemble_curve=_check(m.get("tree_ensemble", {}).get("curve") or m["lstm"]["curve"]),
        compare_test=_check(m["compare_plots"]["test"]),
        compare_train=_check(m["compare_plots"]["train"]),
        naive_comparison=_check(m["compare_plots"].get("naive_comparison", m["compare_plots"]["test"])),
        classical_curve=_check(m["classical"].get("curves") or m["compare_plots"]["test"]),
        learning_curve=_check(learning_curve_plot),
        generator_arch=to_itemize(m["rgan"]["architecture"]["generator"]),
        discriminator_arch=to_itemize(m["rgan"]["architecture"]["discriminator"]),
        rgan_hparams=hyperparam_summary(m["rgan"]["config"]),
        error_table_rows=build_error_rows(m),
    )

    Path(args.out).write_text(filled)
    print(f"Wrote LaTeX to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
