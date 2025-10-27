import numpy as np
import tensorflow as tf
from typing import Dict


def _error_stats(y_true: np.ndarray, y_pred: np.ndarray):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}

def train_lstm_supervised(config: Dict, data_splits, results_dir: str, tag="lstm"):
    L, H = config["L"], config["H"]
    n_in = data_splits["Xtr"].shape[-1]
    x_in = tf.keras.Input(shape=(L,n_in))
    x = tf.keras.layers.LSTM(config["units_g"], return_sequences=False)(x_in)
    x = tf.keras.layers.Dense(H)(x)
    y = tf.keras.layers.Reshape((H,1))(x)
    model = tf.keras.Model(x_in, y)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["lrG"]), loss="mse")

    hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []}
    best_val, best_weights = float("inf"), None
    patience, bad_epochs = config["patience"], 0

    for epoch in range(1, config["epochs"]+1):
        model.fit(data_splits["Xtr"], data_splits["Ytr"], batch_size=config["batch_size"], epochs=1, verbose=0, shuffle=True)
        tr = model.predict(data_splits["Xtr"], verbose=0)
        te = model.predict(data_splits["Xte"], verbose=0)
        va = model.predict(data_splits["Xval"], verbose=0)
        tr_rmse = float(np.sqrt(np.mean((tr.reshape(-1)-data_splits["Ytr"].reshape(-1))**2)))
        te_rmse = float(np.sqrt(np.mean((te.reshape(-1)-data_splits["Yte"].reshape(-1))**2)))
        va_rmse = float(np.sqrt(np.mean((va.reshape(-1)-data_splits["Yval"].reshape(-1))**2)))
        hist["epoch"].append(epoch); hist["train_rmse"].append(tr_rmse); hist["test_rmse"].append(te_rmse); hist["val_rmse"].append(va_rmse)
        print(f"[LSTM] Epoch {epoch:03d} | Train {tr_rmse:.5f} | Val {va_rmse:.5f} | Test {te_rmse:.5f}")
        if va_rmse < best_val - 1e-7:
            best_val = va_rmse; best_weights = model.get_weights(); bad_epochs=0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[LSTM] Early stopping at epoch {epoch}."); break
    if best_weights is not None:
        model.set_weights(best_weights)

    tr = model.predict(data_splits["Xtr"], verbose=0)
    te = model.predict(data_splits["Xte"], verbose=0)
    train_stats = _error_stats(data_splits["Ytr"].reshape(-1), tr.reshape(-1))
    test_stats = _error_stats(data_splits["Yte"].reshape(-1), te.reshape(-1))
    return {"model": model, "history": hist, "train_stats": train_stats, "test_stats": test_stats}
