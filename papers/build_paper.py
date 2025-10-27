#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="research_paper.tex")
    args = ap.parse_args()
    m = json.loads(Path(args.metrics).read_text())

    ets = m["classical"].get("ets_rmse_full", "n/a")
    ets_str = f"{ets:.6f}" if isinstance(ets, float) else str(ets)
    arima = m["classical"].get("arima_rmse_full", "n/a")
    arima_str = f"{arima:.6f}" if isinstance(arima, float) else str(arima)

    tex = Path(__file__).parent/"template.tex"
    filled = tex.read_text() % dict(
        rgan_curve=m["rgan"]["curve"],
        lstm_curve=m["lstm"]["curve"],
        compare_test=m["compare_plots"]["test"],
        compare_train=m["compare_plots"]["train"],
        classical_curve=m["classical"]["curves"] if m["classical"]["curves"] else "",
    )
    Path(args.out).write_text(filled)
    print(f"Wrote LaTeX to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
