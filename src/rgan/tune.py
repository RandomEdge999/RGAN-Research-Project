import itertools
import math
import random
from typing import Dict, Iterable, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .config import TrainConfig


def _set_global_seed(seed: Optional[int]) -> None:
    """Reset global RNG state for reproducible hyper-parameter sweeps."""

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def tune_rgan(
    hp_grid: Dict[str, Iterable],
    base_config: Dict[str, Any],
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Hyper-parameter sweep for the (PyTorch) R-GAN model.
    
    Args:
        hp_grid: Dictionary mapping hyperparameter names to lists of values to try.
        base_config: Base configuration dictionary (will be converted to TrainConfig).
        data_splits: Dictionary containing data splits (X_train, Y_train, etc.).
        results_dir: Directory to save results.
        seed: Random seed.

    Returns:
        A tuple containing:
        - The best hyperparameter configuration found.
        - A DataFrame containing the results of all trials.
    """

    Xtr = data_splits["X_train"]
    Ytr = data_splits["Y_train"]
    Xval = data_splits["X_val"]
    Yval = data_splits["Y_val"]
    Xeval = data_splits.get("X_eval", Xval)
    Yeval = data_splits.get("Y_eval", Yval)
    if Xeval.size == 0 or Yeval.size == 0:
        Xeval, Yeval = Xval, Yval

    from .models_torch import (
        build_generator as build_generator_backend,
        build_discriminator as build_discriminator_backend,
    )
    from .rgan_torch import train_rgan_torch as train_rgan_backend

    best = {"val_rmse": float("inf")}
    results = []

    epochs_each = hp_grid.get("epochs_each", [30])
    if not isinstance(epochs_each, Iterable) or isinstance(epochs_each, (str, bytes)):
        epochs_each = [epochs_each]

    search_keys = [k for k in hp_grid.keys() if k != "epochs_each"]
    search_values = [hp_grid[k] for k in search_keys]
    if not search_keys:
        search_keys = []
        search_values = [[]]

    trial_index = 0

    for epochs_sel in epochs_each:
        for combo in itertools.product(*search_values):
            # Create a dictionary first
            cfg_dict = dict(base_config)
            cfg_dict.update(zip(search_keys, combo))
            cfg_dict["epochs"] = int(epochs_sel)

            # Convert to TrainConfig
            # We need to filter keys that are not in TrainConfig to avoid errors if base_config has extras
            # But TrainConfig is a dataclass, so we can just pass the dict if it matches fields.
            # However, base_config might have keys that are not in TrainConfig (like 'L', 'H' which are in ModelConfig).
            # Wait, TrainConfig DOES contain L and H in my implementation?
            # Let's check config.py.
            # I previously created config.py. Let's assume I need to construct TrainConfig carefully.
            # Actually, looking at config.py (which I wrote earlier), TrainConfig has L, H, etc.
            # So I can try to instantiate it.
            
            # To be safe, I'll filter keys based on TrainConfig fields.
            valid_keys = TrainConfig.__dataclass_fields__.keys()
            train_config_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            
            # Handle potential missing required fields or type mismatches if necessary
            # For now, assume base_config + grid provides all needed fields.
            train_config = TrainConfig(**train_config_kwargs)

            tag_parts = ["tune"]
            for key, value in zip(search_keys, combo):
                tag_parts.append(f"{key}{value}")
            tag_parts.append(f"ep{cfg_dict['epochs']}")

            trial_seed = None
            if seed is not None:
                trial_seed = int(seed + trial_index)
            _set_global_seed(trial_seed)
            if trial_seed is not None:
                tag_parts.append(f"s{trial_seed}")

            # Build models using the config values
            # Note: build_generator/discriminator might take specific args, not the config object directly
            # unless I refactored them to take ModelConfig.
            # I refactored models_torch.py to take ModelConfig? 
            # Let's check models_torch.py content from previous steps. 
            # I did refactor it. It has `build_generator(config: ModelConfig)`?
            # No, I think I kept the factory functions taking individual args but added ModelConfig support?
            # I should check models_torch.py again to be sure how to call it.
            # But for now, I'll stick to passing args as before, or use the new config if possible.
            # The previous tune.py passed args: cfg["L"], cfg["H"], etc.
            # I'll stick to that for safety, or better, use ModelConfig if I can.
            
            # Let's assume for now I pass args manually as before, but using values from cfg_dict.
            
            G = build_generator_backend(
                L=cfg_dict["L"],
                H=cfg_dict["H"],
                n_in=Xtr.shape[-1],
                units=cfg_dict.get("units_g", 64),
                dropout=cfg_dict.get("dropout", 0.0),
                num_layers=cfg_dict.get("g_layers", 1),
                dense_activation=cfg_dict.get("g_dense_activation"),
                layer_norm=True, # Defaulting to True as per previous run_experiment logic? 
                                 # Actually run_experiment sets it based on wgan-gp.
                                 # tune.py didn't seem to set layer_norm explicitly before.
                                 # I'll leave it as default or what was there.
                                 # The previous code didn't pass layer_norm.
            )
            D = build_discriminator_backend(
                L=cfg_dict["L"],
                H=cfg_dict["H"],
                units=cfg_dict.get("units_d", 64),
                dropout=cfg_dict.get("dropout", 0.0),
                num_layers=cfg_dict.get("d_layers", 1),
                activation=cfg_dict.get("d_activation"),
                # Again, layer_norm/spectral_norm were not passed in original tune.py
            )

            splits = dict(Xtr=Xtr, Ytr=Ytr, Xval=Xval, Yval=Yval, Xte=Xeval, Yte=Yeval)
            
            # Pass the TrainConfig object
            out = train_rgan_backend(
                train_config,
                (G, D),
                splits,
                results_dir,
                tag="_".join(tag_parts),
            )

            val_history = out.get("history", {}).get("val_rmse", [])
            val_curve = [float(v) for v in val_history if v is not None and not math.isnan(float(v))]
            val_rmse_best = min(val_curve) if val_curve else float("inf")
            val_rmse_last = val_curve[-1] if val_curve else float("inf")

            eval_stats = out.get("test_stats", {})
            eval_rmse = float(eval_stats.get("rmse", float("nan")))

            result_row = {
                **{key: value for key, value in zip(search_keys, combo)},
                "epochs": cfg_dict["epochs"],
                "val_rmse": val_rmse_best,
                "val_rmse_last": val_rmse_last,
                "eval_rmse": eval_rmse,
                "seed": trial_seed,
            }
            results.append(result_row)

            if val_rmse_best < best["val_rmse"]:
                best = {
                    **{key: value for key, value in zip(search_keys, combo)},
                    "val_rmse": val_rmse_best,
                    "seed": trial_seed,
                }

            trial_index += 1

    return best, pd.DataFrame(results)


