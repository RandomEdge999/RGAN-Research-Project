#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fmt_value(val):
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def to_itemize(lines):
    if not lines:
        return "\\item (none)"
    return "\n".join(f"\\item {line}" for line in lines)


def hyperparam_summary(cfg):
    return (
        f"{cfg['g_layers']} generator LSTM layer(s) with {cfg['units_g']} units "
        f"({cfg['g_activation']}/{cfg['g_recurrent']}, dense activation {cfg['g_dense']}). "
        f"{cfg['d_layers']} discriminator LSTM layer(s) with {cfg['units_d']} units "
        f"({cfg['d_activation']}/{cfg['d_recurrent']}). Regularisation: $\\lambda={fmt_value(cfg['lambda_reg'])}$, "
        f"dropout={fmt_value(cfg['dropout'])}, $\\texttt{{lrG}}={fmt_value(cfg['lrG'])}$, $\\texttt{{lrD}}={fmt_value(cfg['lrD'])}$."
    )


def build_error_rows(metrics):
    rows = []
    for key, label in [
        ("rgan", "R-GAN"),
        ("lstm", "LSTM"),
        ("naive_baseline", "Naïve Baseline"),
        ("naive_bayes", "Naïve Bayes"),
    ]:
        if key not in metrics:
            continue
        train = metrics[key].get("train", {})
        test = metrics[key].get("test", {})
        row = "{} & {} & {} & {} & {} & {} & {} \\\\".format(
            label,
            fmt_value(train.get("rmse", "-")),
            fmt_value(test.get("rmse", "-")),
            fmt_value(train.get("mse", "-")),
            fmt_value(test.get("mse", "-")),
            fmt_value(train.get("bias", "-")),
            fmt_value(test.get("bias", "-")),
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

    filled = template.read_text() % dict(
        rgan_curve=m["rgan"]["curve"],
        lstm_curve=m["lstm"]["curve"],
        naive_curve=m["naive_baseline"].get("curve") or m["rgan"]["curve"],
        naive_bayes_curve=m["naive_bayes"].get("curve") or m["lstm"]["curve"],
        compare_test=m["compare_plots"]["test"],
        compare_train=m["compare_plots"]["train"],
        naive_comparison=m["compare_plots"].get("naive_comparison", m["compare_plots"]["test"]),
        classical_curve=m["classical"]["curves"] if m["classical"].get("curves") else m["compare_plots"]["test"],
        learning_curve=learning_curve_plot,
        generator_arch=to_itemize(m["rgan"]["architecture"]["generator"]),
        discriminator_arch=to_itemize(m["rgan"]["architecture"]["discriminator"]),
        rgan_hparams=hyperparam_summary(m["rgan"]["config"]),
        error_table_rows=build_error_rows(m),
    )

    Path(args.out).write_text(filled)
    print(f"Wrote LaTeX to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
