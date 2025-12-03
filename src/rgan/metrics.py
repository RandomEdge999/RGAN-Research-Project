import math
from typing import Dict, Optional, List
import numpy as np

def error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic error statistics (RMSE, MAE, MSE, Bias)."""
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 0
    if not np.any(mask):
        return float("nan")
    diff = np.abs(y_true - y_pred)
    return float(np.mean(2.0 * diff[mask] / denominator[mask]))

def maape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Arctangent Absolute Percentage Error."""
    epsilon = 1e-8
    return float(np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + epsilon)))))

def mase(y_true: np.ndarray, y_pred: np.ndarray, training_series: np.ndarray) -> float:
    """Mean Absolute Scaled Error."""
    denom = np.mean(np.abs(np.diff(training_series)))
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)

def diebold_mariano(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, power: int = 2) -> Dict[str, float]:
    """Perform the Diebold-Mariano test for predictive accuracy."""
    loss_a = np.abs(actual - pred_a) ** power
    loss_b = np.abs(actual - pred_b) ** power
    diff = loss_a - loss_b
    mean_diff = diff.mean()
    n = diff.size
    if n < 2:
        return {"stat": float("nan"), "pvalue": float("nan")}
    diff_centered = diff - mean_diff
    autocov = np.sum(diff_centered[1:] * diff_centered[:-1]) / (n - 1)
    var_diff = diff_centered.var(ddof=1) + 2 * autocov
    var_diff = max(var_diff, 1e-12)
    stat = mean_diff / math.sqrt(var_diff / n)
    cdf = 0.5 * (1.0 + math.erf(stat / math.sqrt(2.0)))
    pvalue = 2 * (1 - cdf)
    return {"stat": float(stat), "pvalue": float(pvalue)}

def summarise_with_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 300,
    seed: Optional[int] = None,
    original_mean: Optional[float] = None,
    original_std: Optional[float] = None,
    training_series: Optional[np.ndarray] = None,
    training_series_orig: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return base error statistics plus bootstrap uncertainty estimates."""

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    stats = error_stats(y_true_flat, y_pred_flat)
    stats["smape"] = smape(y_true_flat, y_pred_flat)
    stats["maape"] = maape(y_true_flat, y_pred_flat)

    mase_val = float("nan")
    if training_series is not None:
        mase_val = mase(y_true_flat, y_pred_flat, np.asarray(training_series).reshape(-1))
    stats["mase"] = mase_val

    orig_enabled = original_mean is not None and original_std is not None
    if orig_enabled:
        y_true_orig = y_true_flat * original_std + original_mean
        y_pred_orig = y_pred_flat * original_std + original_mean
        orig_stats = error_stats(y_true_orig, y_pred_orig)
        stats.update({f"{k}_orig": v for k, v in orig_stats.items()})
        stats["smape_orig"] = smape(y_true_orig, y_pred_orig)
        stats["maape_orig"] = maape(y_true_orig, y_pred_orig)
        mase_orig_val = float("nan")
        if training_series_orig is not None:
            mase_orig_val = mase(y_true_orig, y_pred_orig, np.asarray(training_series_orig).reshape(-1))
        stats["mase_orig"] = mase_orig_val

    if n_bootstrap <= 0 or y_true_flat.size < 2:
        return stats

    rng = np.random.default_rng(seed)
    rmse_samples = []
    mae_samples = []
    rmse_orig_samples = [] if orig_enabled else None
    mae_orig_samples = [] if orig_enabled else None
    n = y_true_flat.size
    diff_flat = y_pred_flat - y_true_flat
    if orig_enabled:
        diff_orig_flat = diff_flat * original_std
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = diff_flat[idx]
        rmse_samples.append(np.sqrt(np.mean(diff ** 2)))
        mae_samples.append(np.mean(np.abs(diff)))
        if orig_enabled and rmse_orig_samples is not None and mae_orig_samples is not None:
            diff_orig = diff_orig_flat[idx]
            rmse_orig_samples.append(np.sqrt(np.mean(diff_orig ** 2)))
            mae_orig_samples.append(np.mean(np.abs(diff_orig)))

    rmse_arr = np.asarray(rmse_samples, dtype=float)
    mae_arr = np.asarray(mae_samples, dtype=float)

    stats.update(
        {
            "rmse_std": float(np.std(rmse_arr, ddof=1)),
            "rmse_ci_low": float(np.percentile(rmse_arr, 2.5)),
            "rmse_ci_high": float(np.percentile(rmse_arr, 97.5)),
            "mae_std": float(np.std(mae_arr, ddof=1)),
            "mae_ci_low": float(np.percentile(mae_arr, 2.5)),
            "mae_ci_high": float(np.percentile(mae_arr, 97.5)),
        }
    )

    if orig_enabled and rmse_orig_samples and mae_orig_samples:
        rmse_orig_arr = np.asarray(rmse_orig_samples, dtype=float)
        mae_orig_arr = np.asarray(mae_orig_samples, dtype=float)
        stats.update(
            {
                "rmse_orig_std": float(np.std(rmse_orig_arr, ddof=1)),
                "rmse_orig_ci_low": float(np.percentile(rmse_orig_arr, 2.5)),
                "rmse_orig_ci_high": float(np.percentile(rmse_orig_arr, 97.5)),
                "mae_orig_std": float(np.std(mae_orig_arr, ddof=1)),
                "mae_orig_ci_low": float(np.percentile(mae_orig_arr, 2.5)),
                "mae_orig_ci_high": float(np.percentile(mae_orig_arr, 97.5)),
            }
        )
    return stats

def describe_model(model) -> List[str]:
    """Return a human-readable description of the model layers."""
    description = []
    if hasattr(model, "named_children"):
        for name, module in model.named_children():
            description.append(f"{name}: {module.__class__.__name__}")
        if not description:
            description.append(model.__class__.__name__)
    else:
        description.append(model.__class__.__name__)
    return description
