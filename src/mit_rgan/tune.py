import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict
from .models_keras import build_generator, build_discriminator
from .rgan_keras import train_rgan_keras

def tune_rgan_keras(hp_grid: Dict, base_config: Dict, data_splits: Dict, results_dir: str):
    Xtr_full, Ytr_full = data_splits["Xtr_full"], data_splits["Ytr_full"]
    n = len(Xtr_full); n_val = max(1, int(0.1*n))
    Xtr, Ytr = Xtr_full[:-n_val], Ytr_full[:-n_val]
    Xval, Yval = Xtr_full[-n_val:], Ytr_full[-n_val:]

    best = {"val_rmse": float("inf")}
    results = []
    for units_g in hp_grid.get("units_g", [64]):
        for units_d in hp_grid.get("units_d", [64]):
            for lambda_reg in hp_grid.get("lambda_reg", [0.1]):
                cfg = dict(base_config); cfg.update({"units_g": units_g, "units_d": units_d, "lambda_reg": lambda_reg, "epochs": hp_grid.get("epochs_each", 30)})
                tf.keras.backend.clear_session()
                G = build_generator(cfg["L"], cfg["H"], n_in=Xtr.shape[-1], units=units_g, dropout=cfg["dropout"])
                D = build_discriminator(cfg["L"], cfg["H"], units=units_d, dropout=cfg["dropout"])
                splits = dict(Xtr=Xtr, Ytr=Ytr, Xval=Xval, Yval=Yval, Xte=Xval, Yte=Yval)
                out = train_rgan_keras(cfg, (G,D), splits, results_dir, tag=f"tune_g{units_g}_d{units_d}_l{lambda_reg}")
                val_rmse = out["history"]["val_rmse"][-1]
                results.append({"units_g": units_g, "units_d": units_d, "lambda_reg": lambda_reg, "val_rmse": val_rmse})
                if val_rmse < best["val_rmse"]:
                    best = {"units_g": units_g, "units_d": units_d, "lambda_reg": lambda_reg, "val_rmse": val_rmse}
    import pandas as pd
    return best, pd.DataFrame(results)
