#!/usr/bin/env python3
"""SageMaker entry point for the augmentation experiment.

This script runs inside the SageMaker PyTorch container. It translates
SageMaker environment variables into CLI args for run_augmentation.py.

SageMaker provides:
  SM_CHANNEL_DATA    -> /opt/ml/input/data/data/    (uploaded CSV)
  SM_CHANNEL_MODELS  -> /opt/ml/input/data/models/  (trained model weights + metrics.json)
  SM_MODEL_DIR       -> /opt/ml/model/               (archived as model.tar.gz)
  SM_OUTPUT_DATA_DIR -> /opt/ml/output/data/         (save results here)
  SM_NUM_GPUS        -> number of GPUs available

The launcher uploads:
  - CSV dataset to the 'data' channel
  - Trained model directory (models/*.pt + metrics.json) to the 'models' channel
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TERM"] = "dumb"
    os.environ["NO_COLOR"] = "1"
    # Non-interactive matplotlib backend
    os.environ["MPLBACKEND"] = "Agg"

    # SageMaker environment
    data_dir = Path(os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    models_dir = Path(os.environ.get("SM_CHANNEL_MODELS", "/opt/ml/input/data/models"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    # Install the rgan package
    code_dir = Path("/opt/ml/code")
    if (code_dir / "pyproject.toml").exists():
        print("[entry_point_augment] Installing extra dependencies...")
        deps_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "statsmodels>=0.13", "plotly>=5.22", "psutil>=5.9",
             "scikit-learn>=1.3"],
            capture_output=True, text=True,
        )
        if deps_result.returncode != 0:
            print(f"[entry_point_augment] WARNING: extra deps install issue:\n{deps_result.stderr}")

        print("[entry_point_augment] Installing rgan package...")
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(code_dir), "--no-deps"],
            capture_output=True, text=True,
        )
        if install_result.returncode != 0:
            print(f"[entry_point_augment] pip install failed:\n{install_result.stderr}")
            sys.exit(1)
        print("[entry_point_augment] Package installed successfully.")

    # Find the CSV
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    csv_path = csv_files[0]
    print(f"[entry_point_augment] Dataset: {csv_path}")
    print(f"[entry_point_augment] GPUs available: {num_gpus}")

    # Reconstruct the training results directory structure that run_augmentation
    # expects from --results_from: a directory with models/ subdir and metrics.json
    #
    # SageMaker mounts the 'models' channel flat, so we need to check whether
    # it already has the right structure or needs reorganizing.
    results_from_dir = models_dir
    models_subdir = models_dir / "models"
    metrics_json = models_dir / "metrics.json"

    # If the channel has models/*.pt and metrics.json at root, the structure
    # already matches what --results_from expects.
    # If the .pt files are at root level (flat upload), reorganize.
    pt_files_root = list(models_dir.glob("*.pt"))
    pt_files_sub = list(models_subdir.glob("*.pt")) if models_subdir.exists() else []

    if pt_files_root and not pt_files_sub:
        # Flat layout — reorganize into expected structure
        print("[entry_point_augment] Reorganizing flat model uploads into models/ subdirectory...")
        models_subdir.mkdir(exist_ok=True)
        for pt_file in pt_files_root:
            dest = models_subdir / pt_file.name
            shutil.copy2(pt_file, dest)
            print(f"  Copied {pt_file.name} -> models/{pt_file.name}")

    # Verify we have what we need
    gen_ema = models_subdir / "rgan_generator_ema.pt"
    gen_std = models_subdir / "rgan_generator.pt"
    if gen_ema.exists():
        print(f"[entry_point_augment] Found EMA generator: {gen_ema}")
    elif gen_std.exists():
        print(f"[entry_point_augment] Found generator: {gen_std}")
    else:
        print("[entry_point_augment] WARNING: No RGAN generator found in models channel!")
        print(f"  Contents of {models_dir}:")
        for f in sorted(models_dir.rglob("*")):
            print(f"    {f.relative_to(models_dir)}")

    if metrics_json.exists():
        print(f"[entry_point_augment] Found metrics.json")
    else:
        print("[entry_point_augment] WARNING: metrics.json not found in models channel")

    # Build the rgan-augment command
    results_output = output_dir / "augmentation"
    args = [
        sys.executable, "-u", "-m", "rgan.scripts.run_augmentation",
        "--csv", str(csv_path),
        "--results_from", str(results_from_dir),
        "--results_dir", str(results_output),
    ]

    # Map SageMaker hyperparameters to CLI args
    hp_mapping = {
        "SM_HP_TARGET_COL": "--target_col",
        "SM_HP_TIME_COL": "--time_col",
        "SM_HP_L": "--L",
        "SM_HP_H": "--H",
        "SM_HP_TRAIN_SPLIT": "--train_split",
        "SM_HP_NN_EPOCHS": "--nn_epochs",
        "SM_HP_NN_PATIENCE": "--nn_patience",
        "SM_HP_TIMEGAN_EPOCHS": "--timegan_epochs",
        "SM_HP_WGAN_MODEL": "--wgan_model",
    }

    bool_flags = {
        "SM_HP_SKIP_TIMEGAN": "--skip_timegan",
    }

    for env_key, cli_flag in hp_mapping.items():
        val = os.environ.get(env_key)
        if val is not None and val != "":
            args.extend([cli_flag, val])

    for env_key, cli_flag in bool_flags.items():
        val = os.environ.get(env_key)
        if val is not None and val.lower() in ("true", "1", "yes"):
            args.append(cli_flag)

    print(f"[entry_point_augment] Running: {' '.join(args)}")
    result = subprocess.run(args, check=False)

    # Copy results to model dir (SageMaker archives this as model.tar.gz)
    if results_output.exists():
        dest = model_dir / "augmentation"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(results_output, dest)
        print(f"[entry_point_augment] Results copied to {dest}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
