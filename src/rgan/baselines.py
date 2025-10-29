import numpy as np
from sklearn.naive_bayes import GaussianNB

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

def _error_stats(y_true: np.ndarray, y_pred: np.ndarray):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}

def naive_baseline(X, Y):
    """Naive baseline that repeats the last observed value."""
    H = Y.shape[1]
    last_vals = X[:, -1, 0:1]
    naive_pred = np.repeat(last_vals[:, None, :], H, axis=1)
    stats = _error_stats(Y.reshape(-1), naive_pred.reshape(-1))
    return stats, naive_pred

def naive_bayes_forecast(X_train, Y_train, X_eval=None, Y_eval=None):
    """Gaussian Naïve Bayes baseline for multi-step forecasting.

    Each look-back window is flattened into a feature vector and an
    independent Naïve Bayes classifier is trained for every forecast horizon
    step. Continuous regression targets are discretised into adaptive quantile
    bins; the posterior expectation over these bins provides a smooth
    continuous forecast. By default the evaluation is performed on the
    training data (producing in-sample metrics). Supplying ``X_eval``/``Y_eval``
    enables proper out-of-sample evaluation.

    Args:
        X_train: Training windows of shape (n_samples, L, n_features).
        Y_train: Training targets of shape (n_samples, H, 1).
        X_eval: Optional evaluation windows. When omitted the training windows
            are re-used for evaluation.
        Y_eval: Optional evaluation targets. Must be provided when ``X_eval``
            is not ``None``.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: error statistics (RMSE, MSE, MAE,
        Bias) and the forecasts for the evaluation windows.
    """

    if X_eval is None:
        X_eval = X_train
        Y_eval = Y_train
    elif Y_eval is None:
        raise ValueError("Y_eval must be provided when X_eval is supplied.")

    H = Y_train.shape[1]
    n_train = X_train.shape[0]

    X_train_flat = X_train.reshape(n_train, -1)
    X_eval_flat = X_eval.reshape(X_eval.shape[0], -1)

    predictions = np.zeros((X_eval.shape[0], H, 1), dtype=np.float32)

    for h in range(H):
        y_train_h = Y_train[:, h, 0]

        n_bins = int(np.clip(np.sqrt(len(y_train_h)), 5, 50))

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.unique(np.quantile(y_train_h, quantiles, method="nearest"))

        if bin_edges.size <= 2:
            predictions[:, h, 0] = np.mean(y_train_h, dtype=np.float64)
            continue

        bins = bin_edges[1:-1]
        y_classes = np.digitize(y_train_h, bins, right=False)
        n_effective_bins = int(bin_edges.size - 1)

        class_means = np.zeros(n_effective_bins, dtype=np.float64)
        for cls in range(n_effective_bins):
            mask = y_classes == cls
            if np.any(mask):
                class_means[cls] = float(np.mean(y_train_h[mask]))
            else:
                left, right = bin_edges[cls], bin_edges[cls + 1]
                class_means[cls] = float(0.5 * (left + right))

        nb_model = GaussianNB()
        nb_model.fit(X_train_flat, y_classes)

        proba = nb_model.predict_proba(X_eval_flat)
        class_indices = nb_model.classes_.astype(int)
        ordered_means = class_means[class_indices]
        predictions[:, h, 0] = proba @ ordered_means

    stats = _error_stats(Y_eval.reshape(-1), predictions.reshape(-1))
    return stats, predictions

def classical_curves_vs_samples(train_series, test_series, min_frac=0.3, steps=6):
    if not _HAS_STATSMODELS:
        return None, None, None
    N = len(train_series)
    sizes = np.linspace(max(int(min_frac*N), 10), N, steps, dtype=int)
    ets_curve, arima_curve = [], []
    for s in sizes:
        tr = train_series[:s]
        try:
            ets = ExponentialSmoothing(tr, trend=None, seasonal=None, initialization_method="estimated")
            ets_fit = ets.fit(optimized=True)
            ets_fc = ets_fit.forecast(steps=len(test_series))
            ets_rmse = float(np.sqrt(np.mean((ets_fc - test_series)**2)))
        except Exception:
            ets_rmse = np.nan
        ets_curve.append(ets_rmse)

        best_aic, best_fit = np.inf, None
        for p in range(0,3):
            for d in (0,1):
                for q in range(0,3):
                    try:
                        fit = ARIMA(tr, order=(p,d,q)).fit()
                        if fit.aic < best_aic:
                            best_aic, best_fit = fit.aic, fit
                    except Exception:
                        continue
        if best_fit is not None:
            arima_fc = best_fit.forecast(steps=len(test_series))
            arima_rmse = float(np.sqrt(np.mean((arima_fc - test_series)**2)))
        else:
            arima_rmse = np.nan
        arima_curve.append(arima_rmse)
    return sizes, ets_curve, arima_curve


