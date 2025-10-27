import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

def naive_baseline(X, Y):
    H = Y.shape[1]
    last_vals = X[:, -1, 0:1]
    naive_pred = np.repeat(last_vals[:, None, :], H, axis=1)
    y_true = Y.reshape(-1); y_pred = naive_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae

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
