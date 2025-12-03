import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, TypedDict, cast
import logging

# Configure logging for data operations
logger = logging.getLogger(__name__)

COMMON_TIME_COLS = ["calc_time", "date", "datetime", "time", "timestamp", "ts"]


class DataSplit(TypedDict):
    """Type definition for the data split dictionary."""
    df_scaled: pd.DataFrame
    target_col: str
    covariates: List[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split: int
    scaled_train: pd.DataFrame
    scaled_test: pd.DataFrame
    target_mean: float
    target_std: float
    cov_means: Dict[str, float]
    cov_stds: Dict[str, float]


def load_csv_series(
    path: str,
    target: str = "auto",
    time_col: str = "auto",
    resample: str = "",
    agg: str = "last",
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Loads and preprocesses a time series from a CSV file.

    Args:
        path: Path to the CSV file.
        target: Name of the target column. If 'auto', attempts to detect it.
        time_col: Name of the time column. If 'auto', attempts to detect it.
        resample: Pandas offset string for resampling (e.g., '1H').
        agg: Aggregation method for resampling ('mean', 'sum', 'last', etc.).

    Returns:
        A tuple containing:
        - The processed DataFrame.
        - The name of the target column.
        - The name of the time column (or None if not found/used).

    Raises:
        ValueError: If the target column cannot be found or is missing.
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # 1. Target Column Detection
    target_col = target
    if target == "auto":
        if "index_value" in df.columns:
            target_col = "index_value"
        else:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric:
                raise ValueError("No numeric columns found; please specify --target.")
            target_col = numeric[-1]
            logger.info(f"Auto-detected target column: {target_col}")
    else:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in CSV columns: {list(df.columns)}")

    # 2. Time Column Detection
    time_used = None
    if time_col == "auto":
        for c in COMMON_TIME_COLS:
            if c in df.columns:
                time_used = c
                break
    elif time_col and time_col in df.columns:
        time_used = time_col

    # 3. Time-based Preprocessing
    if time_used is not None:
        logger.info(f"Using time column: {time_used}")
        # Coerce to datetime, logging dropped rows
        original_len = len(df)
        df[time_used] = pd.to_datetime(df[time_used], errors="coerce")
        df = df.dropna(subset=[time_used]).sort_values(time_used).reset_index(drop=True)
        dropped = original_len - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows due to invalid datetime parsing in '{time_used}'.")

        if resample:
            logger.info(f"Resampling data to frequency '{resample}' using aggregation '{agg}'.")
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            df = df.set_index(time_used).resample(resample).agg(agg)
            df = df[num_cols].reset_index()

    return df, target_col, time_used


def interpolate_and_standardize(
    df: pd.DataFrame, target_col: str, train_ratio: float = 0.8
) -> DataSplit:
    """Interpolates missing values and standardizes the data (Z-score normalization).

    Args:
        df: The input DataFrame.
        target_col: The name of the target column.
        train_ratio: The fraction of data to use for training (0 < ratio < 1).

    Returns:
        A DataSplit dictionary containing scaled DataFrames, split indices, and scaling statistics.

    Raises:
        ValueError: If train_ratio is invalid or results in empty splits.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    dfx = df.copy()
    # Interpolate missing values for numeric columns
    for c in dfx.select_dtypes(include=["number"]).columns:
        dfx[c] = dfx[c].astype(float).interpolate().ffill().bfill()

    n = len(dfx)
    split = int(train_ratio * n)
    if split <= 0 or split >= n:
        raise ValueError(
            f"train_ratio={train_ratio} produced an empty train or test set (N={n}, split={split}). "
            "Provide more data or adjust --train_ratio."
        )

    train_df = dfx.iloc[:split].reset_index(drop=True)
    test_df = dfx.iloc[split:].reset_index(drop=True)

    # Standardize Target
    t_mean = float(train_df[target_col].mean())
    t_std = float(train_df[target_col].std()) if train_df[target_col].std() > 0 else 1.0
    dfx[target_col] = (dfx[target_col] - t_mean) / t_std

    # Standardize Covariates
    covars = [c for c in dfx.select_dtypes(include=["number"]).columns if c != target_col]
    cov_means = {c: float(train_df[c].mean()) for c in covars}
    cov_stds = {c: float(train_df[c].std()) if train_df[c].std() > 0 else 1.0 for c in covars}
    for c in covars:
        dfx[c] = (dfx[c] - cov_means[c]) / cov_stds[c]

    scaled_train = dfx.iloc[:split].reset_index(drop=True)
    scaled_test = dfx.iloc[split:].reset_index(drop=True)

    return {
        "df_scaled": dfx,
        "target_col": target_col,
        "covariates": covars,
        "train_df": train_df,
        "test_df": test_df,
        "split": split,
        "scaled_train": scaled_train,
        "scaled_test": scaled_test,
        "target_mean": t_mean,
        "target_std": t_std,
        "cov_means": cov_means,
        "cov_stds": cov_stds,
    }


def _validate_window_lengths(total_length: int, L: int, H: int) -> None:
    """Validates that the data is sufficient for the requested window sizes."""
    if L <= 0 or H <= 0:
        raise ValueError(f"Window lengths L={L} and H={H} must be positive integers.")
    if total_length < L + H:
        raise ValueError(
            f"Insufficient rows to create even a single window. "
            f"Require at least {L + H} rows, received {total_length}."
        )


def make_windows_univariate(
    scaled: pd.DataFrame, target_col: str, L: int, H: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sliding windows for univariate time series forecasting.

    Args:
        scaled: The standardized DataFrame.
        target_col: The target column name.
        L: Input sequence length (Lookback).
        H: Output sequence length (Horizon).

    Returns:
        A tuple (X, Y) where:
        - X has shape (num_samples, L, 1)
        - Y has shape (num_samples, H, 1)
    """
    series = scaled[target_col].values.astype(np.float32)
    _validate_window_lengths(len(series), L, H)
    
    # Use stride_tricks or simple list comprehension (list comp is safer for variable steps)
    # For simplicity and readability, we stick to the loop but optimize type casting
    X, Y = [], []
    for i in range(len(series) - L - H + 1):
        X.append(series[i : i + L])
        Y.append(series[i + L : i + L + H])
    
    X_arr = np.array(X, dtype=np.float32)[..., None] # Add feature dim
    Y_arr = np.array(Y, dtype=np.float32)[..., None]
    return X_arr, Y_arr


def make_windows_with_covariates(
    scaled: pd.DataFrame, target_col: str, covariates: List[str], L: int, H: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sliding windows with multivariate covariates.

    Args:
        scaled: The standardized DataFrame.
        target_col: The target column name.
        covariates: List of covariate column names.
        L: Input sequence length.
        H: Output sequence length.

    Returns:
        A tuple (X, Y) where:
        - X has shape (num_samples, L, 1 + num_covariates)
        - Y has shape (num_samples, H, 1)
    """
    cols = [target_col] + list(covariates)
    arr = scaled[cols].values.astype(np.float32)
    tgt = scaled[target_col].values.astype(np.float32)
    
    _validate_window_lengths(len(scaled), L, H)
    
    X, Y = [], []
    for i in range(len(scaled) - L - H + 1):
        X.append(arr[i : i + L, :])
        Y.append(tgt[i + L : i + L + H])
    
    X_arr = np.array(X, dtype=np.float32)
    Y_arr = np.array(Y, dtype=np.float32)[..., None]
    return X_arr, Y_arr


