from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from .metrics import error_stats as _error_stats
from .logging_utils import log_step, log_warn, log_info


try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False
    ConvergenceWarning = None

if _HAS_STATSMODELS and ConvergenceWarning is not None:
    _STATSMODELS_WARNING_CATEGORIES = (UserWarning, ConvergenceWarning)
else:
    _STATSMODELS_WARNING_CATEGORIES = (UserWarning,)

def naive_baseline(X, Y):
    """Naive baseline that repeats the last observed value."""
    log_step(f"Naive baseline: X.shape={X.shape}, Y.shape={Y.shape}")
    H = Y.shape[1]
    last_vals = X[:, -1, 0:1]
    naive_pred = np.repeat(last_vals[:, None, :], H, axis=1)
    stats = _error_stats(Y.reshape(-1), naive_pred.reshape(-1))
    log_step(f"Naive baseline RMSE={stats.get('rmse', 'N/A'):.6f}")
    return stats, naive_pred

def arima_forecast(X_train, Y_train, X_eval=None, Y_eval=None, order=(1, 1, 1)):
    """ARIMA forecast using statsmodels."""
    log_step(f"ARIMA forecast: order={order}, X_train.shape={X_train.shape}")
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels is required for ARIMA.")

    if X_eval is None:
        X_eval = X_train
        Y_eval = Y_train
    elif Y_eval is None:
        raise ValueError("Y_eval must be provided when X_eval is supplied.")

    # Reconstruct training series from windows (assuming stride=1)
    # X_train shape: (N, L, 1)
    # We take the first value of each window, plus the rest of the last window
    # Actually, simpler: just flatten Y_train (targets) if they are contiguous?
    # No, let's just use the targets as the series to fit, assuming they are sequential.
    # But X_train contains the history.
    # Let's assume standard sliding window construction.
    
    # Fit on the target series (Y_train). 
    # Note: This assumes Y_train windows are sequential and overlapping or contiguous.
    # If they are shuffled, this won't work well for fitting.
    # Ideally we should pass the full original training series.
    # For now, let's try to reconstruct or just fit on the flattened targets if they look continuous.
    # A better approach for the baseline is to fit on the history of each window? No, that's too slow.
    
    # We will fit on the concatenated targets of the training set. 
    # This is a simplification.
    train_series = Y_train[:, 0, 0] # First step of each target window
    
    with warnings.catch_warnings():
        for category in _STATSMODELS_WARNING_CATEGORIES:
            warnings.simplefilter("ignore", category=category)

        try:
            model = ARIMA(train_series, order=order)
            fit = model.fit()
        except Exception:
            # ARIMA fit can fail on certain series (singular Hessian, etc.)
            H = Y_eval.shape[1]
            predictions = np.zeros((len(X_eval), H, 1), dtype=np.float32)
            stats = _error_stats(Y_eval.reshape(-1), predictions.reshape(-1))
            return stats, predictions

    predictions = []
    H = Y_eval.shape[1]

    for i in range(len(X_eval)):
        history = X_eval[i, :, 0]
        try:
            res = fit.apply(history)
            fc = res.forecast(steps=H)
            predictions.append(fc)
        except Exception:
            predictions.append(np.zeros(H))

    predictions = np.array(predictions).reshape(len(X_eval), H, 1)
    stats = _error_stats(Y_eval.reshape(-1), predictions.reshape(-1))
    log_step(f"ARIMA forecast RMSE={stats.get('rmse', 'N/A'):.6f}")
    return stats, predictions


def arma_forecast(X_train, Y_train, X_eval=None, Y_eval=None, order=(1, 1)):
    """ARMA forecast (ARIMA with d=0)."""
    log_step(f"ARMA forecast: order={order}")
    return arima_forecast(X_train, Y_train, X_eval, Y_eval, order=(order[0], 0, order[1]))


def tree_ensemble_forecast(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray | None = None,
    Y_eval: np.ndarray | None = None,
    *,
    random_state: int | None = 42,
    return_model: bool = False,
) -> tuple[dict, np.ndarray] | tuple[dict, np.ndarray, MultiOutputRegressor]:
    """Gradient boosted tree baseline over flattened windows.

    A strong classical baseline that trains one regressor per prediction horizon
    via ``MultiOutputRegressor``. Features are simply the lagged window values,
    making this baseline equivalent to a powerful boosted autoregression.

    Args:
        X_train: Training windows of shape (n_samples, L, n_features).
        Y_train: Training targets of shape (n_samples, H, 1).
        X_eval: Optional evaluation windows. Defaults to the training windows.
        Y_eval: Optional evaluation targets. Must be supplied when ``X_eval`` is
            provided.
        random_state: Seed forwarded to the underlying gradient boosting
            estimator.
        return_model: When ``True`` the fitted ``MultiOutputRegressor`` is
            returned alongside the metrics and predictions so downstream code
            can reuse the model without re-fitting.
    """

    if X_eval is None:
        X_eval = X_train
        Y_eval = Y_train
    elif Y_eval is None:
        raise ValueError("Y_eval must be provided when X_eval is supplied.")

    n_samples, H, _ = Y_eval.shape
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_eval_flat = X_eval.reshape(X_eval.shape[0], -1)

    log_step(f"Tree ensemble: X_train.shape={X_train.shape}, n_estimators=400, max_depth=3")
    base = GradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.7,
        max_depth=3,
        random_state=random_state,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train_flat, Y_train.reshape(Y_train.shape[0], H))
    preds = model.predict(X_eval_flat).reshape(n_samples, H, 1)

    stats = _error_stats(Y_eval.reshape(-1), preds.reshape(-1))
    log_step(f"Tree ensemble RMSE={stats.get('rmse', 'N/A'):.6f}")
    if return_model:
        return stats, preds, model
    return stats, preds

def classical_curves_vs_samples(train_series, test_series, min_frac=0.3, steps=6):
    if not _HAS_STATSMODELS:
        return None, None, None
    with warnings.catch_warnings():
        for category in _STATSMODELS_WARNING_CATEGORIES:
            warnings.simplefilter("ignore", category=category)

        N = len(train_series)
        sizes = np.linspace(max(int(min_frac * N), 10), N, steps, dtype=int)
        ets_curve, arima_curve = [], []
        for s in sizes:
            tr = train_series[:s]
            try:
                ets = ExponentialSmoothing(tr, trend=None, seasonal=None, initialization_method="estimated")
                ets_fit = ets.fit(optimized=True)
                ets_fc = ets_fit.forecast(steps=len(test_series))
                ets_rmse = float(np.sqrt(np.mean((ets_fc - test_series) ** 2)))
            except Exception:
                ets_rmse = np.nan
            ets_curve.append(ets_rmse)

            best_aic, best_fit = np.inf, None
            for p in range(0, 3):
                for d in (0, 1):
                    for q in range(0, 3):
                        try:
                            fit = ARIMA(tr, order=(p, d, q)).fit()
                            if fit.aic < best_aic:
                                best_aic, best_fit = fit.aic, fit
                        except Exception:
                            continue
            if best_fit is not None:
                arima_fc = best_fit.forecast(steps=len(test_series))
                arima_rmse = float(np.sqrt(np.mean((arima_fc - test_series) ** 2)))
            else:
                arima_rmse = np.nan
            arima_curve.append(arima_rmse)
    return sizes, ets_curve, arima_curve


