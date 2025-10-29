import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

COMMON_TIME_COLS = ["calc_time", "date", "datetime", "time", "timestamp", "ts"]

def load_csv_series(
    path: str,
    target: str = "auto",
    time_col: str = "auto",
    resample: str = "",
    agg: str = "last",
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    df = pd.read_csv(path)
    if target == "auto":
        if "index_value" in df.columns:
            target_col = "index_value"
        else:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric:
                raise ValueError("No numeric columns found; please specify --target.")
            target_col = numeric[-1]
    else:
        if target not in df.columns:
            raise ValueError(f"--target '{target}' not in columns: {list(df.columns)}")
        target_col = target

    time_used = None
    if time_col == "auto":
        for c in COMMON_TIME_COLS:
            if c in df.columns:
                time_used = c; break
    elif time_col and time_col in df.columns:
        time_used = time_col

    if time_used is not None:
        df[time_used] = pd.to_datetime(df[time_used], errors="coerce")
        df = df.dropna(subset=[time_used]).sort_values(time_used).reset_index(drop=True)
        if resample:
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            df = df.set_index(time_used).resample(resample).agg(agg)
            df = df[num_cols].reset_index()

    return df, target_col, time_used

def interpolate_and_standardize(df: pd.DataFrame, target_col: str, train_ratio: float = 0.8) -> Dict:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    dfx = df.copy()
    for c in dfx.select_dtypes(include=["number"]).columns:
        dfx[c] = dfx[c].astype(float).interpolate().ffill().bfill()

    n = len(dfx)
    split = int(train_ratio * n)
    if split <= 0 or split >= n:
        raise ValueError(
            "train_ratio produced an empty train or test set. "
            "Provide more data or adjust --train_ratio."
        )
    train_df = dfx.iloc[:split].reset_index(drop=True)
    test_df  = dfx.iloc[split:].reset_index(drop=True)

    t_mean = float(train_df[target_col].mean())
    t_std  = float(train_df[target_col].std()) if train_df[target_col].std() > 0 else 1.0
    dfx[target_col] = (dfx[target_col] - t_mean) / t_std

    covars = [c for c in dfx.select_dtypes(include=["number"]).columns if c != target_col]
    cov_means = {c: float(train_df[c].mean()) for c in covars}
    cov_stds  = {c: float(train_df[c].std()) if train_df[c].std() > 0 else 1.0 for c in covars}
    for c in covars:
        dfx[c] = (dfx[c] - cov_means[c]) / cov_stds[c]

    scaled_train = dfx.iloc[:split].reset_index(drop=True)
    scaled_test  = dfx.iloc[split:].reset_index(drop=True)

    return dict(
        df_scaled=dfx, target_col=target_col, covariates=covars,
        train_df=train_df, test_df=test_df, split=split,
        scaled_train=scaled_train, scaled_test=scaled_test,
        target_mean=t_mean, target_std=t_std, cov_means=cov_means, cov_stds=cov_stds
    )

def _validate_window_lengths(total_length: int, L: int, H: int) -> None:
    if L <= 0 or H <= 0:
        raise ValueError("Window lengths L and H must be positive integers.")
    if total_length < L + H:
        raise ValueError(
            "Insufficient rows to create even a single window. "
            f"Require at least {L + H} rows, received {total_length}."
        )


def make_windows_univariate(scaled: pd.DataFrame, target_col: str, L: int, H: int):
    series = scaled[target_col].values.astype(np.float32)
    _validate_window_lengths(len(series), L, H)
    X, Y = [], []
    for i in range(len(series) - L - H + 1):
        X.append(series[i:i+L])
        Y.append(series[i+L:i+L+H])
    return np.array(X, dtype=np.float32)[..., None], np.array(Y, dtype=np.float32)[..., None]

def make_windows_with_covariates(
    scaled: pd.DataFrame, target_col: str, covariates: list, L: int, H: int
):
    cols = [target_col] + list(covariates)
    arr = scaled[cols].values.astype(np.float32)
    tgt = scaled[target_col].values.astype(np.float32)
    _validate_window_lengths(len(scaled), L, H)
    X, Y = [], []
    for i in range(len(scaled) - L - H + 1):
        X.append(arr[i:i+L, :])
        Y.append(tgt[i+L:i+L+H])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)[..., None]
    return X, Y


