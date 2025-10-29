import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

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

def naive_bayes_forecast(X, Y):
    """
    Naive Bayes classifier for time series forecasting.
    Treats each time step as a feature and uses Gaussian Naive Bayes.
    """
    H = Y.shape[1]
    n_samples, L, n_features = X.shape
    
    # Flatten the input sequences for Naive Bayes
    X_flat = X.reshape(n_samples, -1)  # Shape: (n_samples, L * n_features)
    
    # For multi-step forecasting, we'll predict each horizon step separately
    predictions = np.zeros_like(Y)
    
    for h in range(H):
        # Target is the h-th step ahead
        y_h = Y[:, h, 0]  # Shape: (n_samples,)
        
        # Train Naive Bayes for this horizon
        nb_model = GaussianNB()
        nb_model.fit(X_flat, y_h)
        
        # Predict for this horizon
        pred_h = nb_model.predict(X_flat)
        predictions[:, h, 0] = pred_h
    
    stats = _error_stats(Y.reshape(-1), predictions.reshape(-1))
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
