#!/usr/bin/env python3
"""SageMaker training entry point.

This script runs inside the SageMaker PyTorch container. It translates
SageMaker environment variables into CLI args for run_training.py.

SageMaker provides:
  SM_CHANNEL_DATA   -> /opt/ml/input/data/data/  (your uploaded CSV)
  SM_MODEL_DIR      -> /opt/ml/model/             (save checkpoints here)
  SM_OUTPUT_DATA_DIR -> /opt/ml/output/data/      (save results here)
  SM_NUM_GPUS       -> number of GPUs available

Usage in SageMaker:
  The launcher uploads the CSV to S3, SageMaker mounts it at SM_CHANNEL_DATA.
  This script finds the CSV, runs the experiment, copies results to output.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def main():
    # SageMaker environment
    data_dir = Path(os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    # Install the rgan package — SageMaker extracts the tarball to /opt/ml/code
    # but doesn't run setup.sh automatically
    code_dir = Path("/opt/ml/code")
    if (code_dir / "pyproject.toml").exists():
        print("[entry_point] Installing rgan package...")
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(code_dir), "--no-deps"],
            capture_output=True, text=True,
        )
        if install_result.returncode != 0:
            print(f"[entry_point] pip install failed:\n{install_result.stderr}")
            sys.exit(1)
        print("[entry_point] Package installed successfully.")

    # Find the CSV in the data channel
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    csv_path = csv_files[0]
    print(f"[entry_point] Using dataset: {csv_path}")
    print(f"[entry_point] GPUs available: {num_gpus}")

    # Build the rgan-train command from hyperparameters
    # SageMaker passes hyperparameters as env vars prefixed with SM_HP_
    args = [
        sys.executable, "-m", "rgan.scripts.run_training",
        "--csv", str(csv_path),
        "--results_dir", str(output_dir / "experiment"),
        "--checkpoint_dir", "/opt/ml/checkpoints",
    ]

    # Map SageMaker hyperparameters to CLI args
    # Value-based flags (--flag value)
    hp_mapping = {
        "SM_HP_EPOCHS": "--epochs",
        "SM_HP_BATCH_SIZE": "--batch_size",
        "SM_HP_L": "--L",
        "SM_HP_H": "--H",
        "SM_HP_NOISE_LEVELS": "--noise_levels",
        "SM_HP_CHECKPOINT_EVERY": "--checkpoint_every",
        "SM_HP_LAMBDA_REG": "--lambda_reg",
        "SM_HP_GAN_VARIANT": "--gan_variant",
        "SM_HP_UNITS_G": "--units_g",
        "SM_HP_UNITS_D": "--units_d",
        "SM_HP_G_LAYERS": "--g_layers",
        "SM_HP_D_LAYERS": "--d_layers",
        "SM_HP_LRG": "--lrG",
        "SM_HP_LRD": "--lrD",
        "SM_HP_PATIENCE": "--patience",
        "SM_HP_SEED": "--seed",
        "SM_HP_TARGET": "--target",
        "SM_HP_TRAIN_RATIO": "--train_ratio",
        "SM_HP_EMA_DECAY": "--ema_decay",
        "SM_HP_BOOTSTRAP_SAMPLES": "--bootstrap_samples",
        "SM_HP_MAX_TRAIN_WINDOWS": "--max_train_windows",
        "SM_HP_DROPOUT": "--dropout",
        "SM_HP_SUPERVISED_WARMUP_EPOCHS": "--supervised_warmup_epochs",
        "SM_HP_NUM_WORKERS": "--num_workers",
        # New: data preprocessing
        "SM_HP_RESAMPLE": "--resample",
        "SM_HP_AGG": "--agg",
        # New: WGAN-GP / training dynamics
        "SM_HP_WGAN_GP_LAMBDA": "--wgan_gp_lambda",
        "SM_HP_D_STEPS": "--d_steps",
        "SM_HP_G_STEPS": "--g_steps",
        "SM_HP_GRAD_CLIP": "--grad_clip",
        "SM_HP_WEIGHT_DECAY": "--weight_decay",
        "SM_HP_ADV_WEIGHT": "--adv_weight",
    }

    # Boolean flags (--flag, no value)
    bool_flags = {
        "SM_HP_SKIP_CLASSICAL": "--skip_classical",
        "SM_HP_REQUIRE_CUDA": "--require_cuda",
    }

    for env_key, cli_flag in hp_mapping.items():
        val = os.environ.get(env_key)
        if val is not None and val != "":
            args.extend([cli_flag, val])

    for env_key, cli_flag in bool_flags.items():
        val = os.environ.get(env_key)
        if val is not None and val.lower() in ("true", "1", "yes"):
            args.append(cli_flag)

    # If GPU available, ensure CUDA is used and enforce it
    if num_gpus > 0:
        args.extend(["--gpu_id", "0"])
        if "--require_cuda" not in args:
            args.append("--require_cuda")

    # Check for resume checkpoint
    latest_ckpt = Path("/opt/ml/checkpoints") / "checkpoint_latest.pt"
    if latest_ckpt.exists():
        print(f"[entry_point] Resuming from checkpoint: {latest_ckpt}")
        args.extend(["--resume_from", str(latest_ckpt)])

    print(f"[entry_point] Running: {' '.join(args)}")
    result = subprocess.run(args, check=False)

    # Copy results to model dir too (SageMaker archives model_dir as model.tar.gz)
    results_path = output_dir / "experiment"
    if results_path.exists():
        dest = model_dir / "results"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(results_path, dest)
        print(f"[entry_point] Results copied to {dest}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
