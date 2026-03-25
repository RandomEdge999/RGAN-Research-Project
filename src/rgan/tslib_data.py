"""Download and prepare standard TSLib benchmark datasets.

Datasets are fetched from the HuggingFace ``thuml/Time-Series-Library``
repository via the ``datasets`` library and cached locally as CSVs
under ``data/tslib/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ── Dataset registry ────────────────────────────────────────────────
# Split sizes follow the TSLib convention exactly:
#   ETTh*: 12 months train / 4 months val / 4 months test (hourly)
#   ETTm*: same ratio but 15-min granularity (4× more data points)
#   Weather, Exchange, ECL, Traffic, ILI: 70 / 10 / 20 ratio split
#
# ``hf_name`` is the config name passed to ``datasets.load_dataset``.

TSLIB_DATASETS: Dict[str, dict] = {
    # ── ETT family (Electricity Transformer Temperature) ────────────
    "ETTh1": {
        "hf_name": "ETTh1",
        "target": "OT",
        "split": (8640, 2880, 2880),       # 14,400 rows, hourly
    },
    "ETTh2": {
        "hf_name": "ETTh2",
        "target": "OT",
        "split": (8640, 2880, 2880),       # 14,400 rows, hourly
    },
    "ETTm1": {
        "hf_name": "ETTm1",
        "target": "OT",
        "split": (34560, 11520, 11520),    # 57,600 rows, 15-min
    },
    "ETTm2": {
        "hf_name": "ETTm2",
        "target": "OT",
        "split": (34560, 11520, 11520),    # 57,600 rows, 15-min
    },
    # ── Large multivariate benchmarks ───────────────────────────────
    "Weather": {
        "hf_name": "weather",
        "target": "OT",
        "split_ratio": (0.7, 0.1, 0.2),   # ~52,696 rows, 21 features
    },
    "Exchange": {
        "hf_name": "exchange_rate",
        "target": "OT",
        "split_ratio": (0.7, 0.1, 0.2),   # ~7,588 rows, 8 currencies
    },
    "ECL": {
        "hf_name": "electricity",
        "target": "OT",
        "split_ratio": (0.7, 0.1, 0.2),   # ~26,304 rows, 321 clients
    },
    "Traffic": {
        "hf_name": "traffic",
        "target": "OT",
        "split_ratio": (0.7, 0.1, 0.2),   # ~17,544 rows, 862 sensors
    },
    # ── Domain-specific ─────────────────────────────────────────────
    "ILI": {
        "hf_name": "national_illness",
        "target": "OT",
        "split_ratio": (0.7, 0.1, 0.2),   # ~966 rows, 7 features
    },
}


def _download_hf(name: str, data_dir: str) -> Path:
    """Download a dataset from HuggingFace and cache as CSV."""
    info = TSLIB_DATASETS[name]
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / f"{info['hf_name']}.csv"

    if local_path.exists():
        return local_path

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for benchmark downloads. "
            "Install it with: pip install -e '.[benchmark]'"
        )

    print(f"Downloading {name} from HuggingFace (thuml/Time-Series-Library) ...")
    ds = hf_load("thuml/Time-Series-Library", info["hf_name"], split="train")
    df = ds.to_pandas()
    df.to_csv(local_path, index=False)
    print(f"  Saved to {local_path}  ({local_path.stat().st_size / 1024:.0f} KB)")
    return local_path


def load_dataset(name: str, data_dir: str = "data/tslib") -> Tuple[pd.DataFrame, str]:
    """Download (if needed) and load a TSLib dataset.

    Returns ``(dataframe, target_column_name)``.
    """
    if name not in TSLIB_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(TSLIB_DATASETS)}"
        )
    path = _download_hf(name, data_dir)
    df = pd.read_csv(path)
    target = TSLIB_DATASETS[name]["target"]
    return df, target


def split_and_standardize(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
) -> Dict:
    """Apply TSLib-standard train/val/test split and z-score normalisation.

    Returns a dict with keys:
        scaled_train, scaled_val, scaled_test  (pd.DataFrame)
        target_mean, target_std                (float)
        target_col                             (str)
    """
    info = TSLIB_DATASETS[dataset_name]

    # ── determine split boundaries ──────────────────────────────────
    n = len(df)
    if "split" in info:
        n_train, n_val, n_test = info["split"]
        if n_train + n_val + n_test > n:
            ratio = n / (n_train + n_val + n_test)
            n_train = int(n_train * ratio)
            n_val = int(n_val * ratio)
            n_test = n - n_train - n_val
    else:
        r_train, r_val, _ = info["split_ratio"]
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        n_test = n - n_train - n_val

    # ── interpolate missing values ──────────────────────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for c in numeric_cols:
        df[c] = df[c].astype(float).interpolate().ffill().bfill()

    # ── z-score using train statistics ──────────────────────────────
    train_slice = df.iloc[:n_train]
    t_mean = float(train_slice[target_col].mean())
    t_std = float(train_slice[target_col].std())
    if t_std == 0:
        t_std = 1.0

    df_scaled = df.copy()
    df_scaled[target_col] = (df_scaled[target_col] - t_mean) / t_std

    return {
        "scaled_train": df_scaled.iloc[:n_train].reset_index(drop=True),
        "scaled_val": df_scaled.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "scaled_test": df_scaled.iloc[n_train + n_val :].reset_index(drop=True),
        "target_mean": t_mean,
        "target_std": t_std,
        "target_col": target_col,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
