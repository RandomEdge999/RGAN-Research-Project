#!/usr/bin/env python3
"""Launch a SageMaker training job from local machine.

Usage:
    python -m cloud.launch --csv data/binance/Binance_Data.csv --epochs 80
    python -m cloud.launch --csv data/binance/Binance_Data.csv --epochs 80 --spot false
    python -m cloud.launch --csv data/binance/Binance_Data.csv --job_name my-experiment
    python -m cloud.launch --csv data/binance/Binance_Data.csv --resume_job rgan-20260312-160656

This script:
  1. Ensures the S3 bucket exists
  2. Uploads the CSV dataset to S3
  3. Uploads the source code as a tarball
  4. Launches a SageMaker PyTorch training job via boto3
  5. Prints the job name for monitoring/syncing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import yaml


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _coalesce_override(cli_value, default_value):
    """Prefer explicit CLI values while allowing empty-string fallbacks for text args."""
    if cli_value is None:
        return default_value
    if isinstance(cli_value, str) and cli_value == "":
        return default_value
    return cli_value


def _build_hyperparameters(args, defaults: dict) -> dict[str, str]:
    """Build SageMaker hyperparameters from CLI args and config defaults."""
    hyperparameters = {
        "epochs": str(_coalesce_override(args.epochs, defaults["epochs"])),
        "batch_size": str(_coalesce_override(args.batch_size, defaults["batch_size"])),
        "L": str(_coalesce_override(args.L, defaults["L"])),
        "H": str(_coalesce_override(args.H, defaults["H"])),
        "noise_levels": str(_coalesce_override(args.noise_levels, defaults["noise_levels"])),
        "checkpoint_every": str(_coalesce_override(args.checkpoint_every, defaults["checkpoint_every"])),
        "target": args.target,
        "time_col": args.time_col,
        "resample": str(_coalesce_override(args.resample, defaults.get("resample", ""))),
        "agg": str(_coalesce_override(args.agg, defaults.get("agg", "last"))),
        "units_g": str(defaults.get("units_g", 128)),
        "units_d": str(defaults.get("units_d", 128)),
        "g_layers": str(defaults.get("g_layers", 2)),
        "d_layers": str(defaults.get("d_layers", 2)),
        "lrG": str(defaults.get("lrG", 0.0005)),
        "lrD": str(defaults.get("lrD", 0.0005)),
        "lambda_reg": str(defaults.get("lambda_reg", 0.5)),
        "gan_variant": str(_coalesce_override(args.gan_variant, defaults.get("gan_variant", "wgan-gp"))),
        "wgan_gp_lambda": str(defaults.get("wgan_gp_lambda", 10.0)),
        "d_steps": str(defaults.get("d_steps", 3)),
        "g_steps": str(defaults.get("g_steps", 1)),
        "grad_clip": str(defaults.get("grad_clip", 1.0)),
        "dropout": str(defaults.get("dropout", 0.1)),
        "ema_decay": str(defaults.get("ema_decay", 0.999)),
        "supervised_warmup_epochs": str(defaults.get("supervised_warmup_epochs", 10)),
        "patience": str(defaults.get("patience", 25)),
        "bootstrap_samples": str(defaults.get("bootstrap_samples", 300)),
        "num_workers": str(defaults.get("num_workers", 2)),
        "seed": str(args.seed if args.seed is not None else defaults.get("seed", 42)),
        "train_ratio": str(defaults.get("train_ratio", 0.8)),
        "weight_decay": str(defaults.get("weight_decay", 0.0)),
        "adv_weight": str(defaults.get("adv_weight", 1.0)),
        "eval_every": str(defaults.get("eval_every", 5)),
        "eval_batch_size": str(defaults.get("eval_batch_size", 2048)),
        "compile_mode": str(_coalesce_override(args.compile_mode, defaults.get("compile_mode", "off"))),
        "preload_to_device": str(_coalesce_override(args.preload_to_device, defaults.get("preload_to_device", "never"))),
        "critic_reg_interval": str(
            _coalesce_override(args.critic_reg_interval, defaults.get("critic_reg_interval", 1))
        ),
        "critic_arch": str(_coalesce_override(args.critic_arch, defaults.get("critic_arch", "tcn"))),
    }
    if args.skip_classical or defaults.get("skip_classical", False):
        hyperparameters["skip_classical"] = "true"
    if args.skip_noise_robustness or defaults.get("skip_noise_robustness", False):
        hyperparameters["skip_noise_robustness"] = "true"
    if args.deterministic:
        hyperparameters["deterministic"] = "true"
    if args.max_train_windows:
        hyperparameters["max_train_windows"] = str(args.max_train_windows)
    if args.only_models:
        hyperparameters["only_models"] = args.only_models
    if args.prior_results:
        hyperparameters["prior_results"] = args.prior_results
    return hyperparameters


def _create_source_tarball(project_root: Path, tmp_dir: str) -> str:
    """Create a source tarball with entry_point.py, src/, and pyproject.toml."""
    tar_path = os.path.join(tmp_dir, "sourcedir.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        # Entry point at root of tarball
        tar.add(project_root / "cloud" / "entry_point.py", arcname="entry_point.py")
        # Source code
        tar.add(project_root / "src", arcname="src")
        # pyproject.toml for pip install -e .
        tar.add(project_root / "pyproject.toml", arcname="pyproject.toml")
        # setup script to install the package inside the container
        setup_script = os.path.join(tmp_dir, "setup.sh")
        with open(setup_script, "w") as f:
            f.write("#!/bin/bash\ncd /opt/ml/code && pip install -e . --no-deps\n")
        tar.add(setup_script, arcname="setup.sh")

    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"  Source tarball: {size_mb:.1f} MB")
    return tar_path


def _get_pytorch_image_uri(region: str, pt_version: str, py_version: str) -> str:
    """Get the official AWS Deep Learning Container image URI for PyTorch."""
    # AWS DLC registry accounts per region
    account_map = {
        "us-east-1": "763104351884",
        "us-east-2": "763104351884",
        "us-west-1": "763104351884",
        "us-west-2": "763104351884",
        "eu-west-1": "763104351884",
        "eu-central-1": "763104351884",
        "ap-northeast-1": "763104351884",
        "ap-southeast-1": "763104351884",
    }
    account = account_map.get(region, "763104351884")
    # Ubuntu version depends on PyTorch version:
    # PT 2.5+ uses ubuntu22.04, PT 2.0-2.4 uses ubuntu20.04
    major_minor = tuple(int(x) for x in pt_version.split(".")[:2])
    ubuntu_ver = "ubuntu22.04" if major_minor >= (2, 5) else "ubuntu20.04"
    cuda_ver = "cu124" if major_minor >= (2, 5) else "cu121"
    tag = f"{pt_version}-gpu-{py_version}-{cuda_ver}-{ubuntu_ver}-sagemaker"
    return f"{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{tag}"


def main():
    ap = argparse.ArgumentParser(description="Launch RGAN training on SageMaker")
    ap.add_argument("--csv", required=True, help="Local path to CSV dataset")
    ap.add_argument("--target", default="auto", help="Target column name")
    ap.add_argument("--time_col", default="auto", help="Time column name")
    ap.add_argument("--job_name", default="", help="Custom job name (auto-generated if empty)")
    ap.add_argument("--instance_type", default="", help="Override instance type from config")
    ap.add_argument("--spot", default="", help="Use spot instances (true/false, default from config)")
    ap.add_argument("--resample", default=None, help="Override the configured resampling frequency (e.g. '1min')")
    ap.add_argument("--agg", default=None, help="Override the configured resampling aggregation")

    # Training hyperparameters (override config defaults)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--noise_levels", default=None)
    ap.add_argument("--checkpoint_every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gan_variant", default=None)
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA mode in the training job.")
    ap.add_argument("--skip_classical", action="store_true")
    ap.add_argument("--skip_noise_robustness", action="store_true")
    ap.add_argument("--max_train_windows", type=int, default=None)
    ap.add_argument("--compile_mode", choices=["off", "reduce-overhead"], default=None)
    ap.add_argument("--preload_to_device", choices=["auto", "always", "never"], default=None)
    ap.add_argument("--critic_reg_interval", type=int, default=None)
    ap.add_argument("--critic_arch", choices=["lstm", "tcn"], default=None)
    ap.add_argument("--resume_job", default="", help="Resume from a previous job's checkpoints (pass the old job name)")
    ap.add_argument("--only_models", default="", help="Comma-separated models to retrain (e.g. 'patchtst,itransformer'). Requires --prior_results.")
    ap.add_argument("--prior_results", default="", help="Path to prior results dir for loading skipped models.")

    args = ap.parse_args()

    cfg = _load_config()
    project_root = Path(__file__).parent.parent

    # Validate
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    role_arn = cfg["sagemaker"]["role_arn"]
    if not role_arn:
        print("ERROR: sagemaker.role_arn is not set in cloud/config.yaml")
        sys.exit(1)

    region = cfg["aws"]["region"]
    bucket = cfg["s3"]["bucket"]
    sm_cfg = cfg["sagemaker"]
    defaults = cfg["defaults"]

    import boto3

    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)

    # 1. Ensure S3 bucket
    print("\n[1/5] Ensuring S3 bucket exists...")
    from cloud.s3_utils import ensure_bucket
    ensure_bucket(bucket, region)

    # 2. Upload dataset to S3
    print("\n[2/5] Uploading dataset...")
    data_key = f"{cfg['s3']['data_prefix']}{csv_path.name}"
    s3.upload_file(str(csv_path), bucket, data_key)
    data_s3_uri = f"s3://{bucket}/{cfg['s3']['data_prefix']}"
    print(f"  Dataset uploaded to s3://{bucket}/{data_key}")

    # 3. Upload source code tarball
    print("\n[3/5] Uploading source code...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = _create_source_tarball(project_root, tmp_dir)
        source_key = "code/sourcedir.tar.gz"
        s3.upload_file(tar_path, bucket, source_key)
    source_s3_uri = f"s3://{bucket}/{source_key}"
    print(f"  Source code uploaded to {source_s3_uri}")

    # 4. Build hyperparameters — merge CLI overrides with config defaults
    print("\n[4/5] Configuring training job...")
    hyperparameters = _build_hyperparameters(args, defaults)

    # Job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"rgan-{timestamp}"
    job_name = job_name.replace("_", "-")

    instance_type = args.instance_type or sm_cfg["instance_type"]
    use_spot = sm_cfg["use_spot"]
    if args.spot:
        use_spot = args.spot.lower() in ("true", "1", "yes")

    image_uri = _get_pytorch_image_uri(region, sm_cfg["pytorch_version"], sm_cfg["python_version"])

    # Copy checkpoints from a previous job if resuming
    if args.resume_job:
        old_prefix = f"checkpoints/{args.resume_job}/"
        new_prefix = f"checkpoints/{job_name}/"
        print(f"\n[*] Copying checkpoints from {args.resume_job} to {job_name}...")
        paginator = s3.get_paginator("list_objects_v2")
        copied = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=old_prefix):
            for obj in page.get("Contents", []):
                old_key = obj["Key"]
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                s3.copy_object(
                    Bucket=bucket,
                    CopySource={"Bucket": bucket, "Key": old_key},
                    Key=new_key,
                )
                copied += 1
        if copied:
            print(f"  Copied {copied} checkpoint file(s)")
        else:
            print(f"  WARNING: No checkpoints found for job {args.resume_job}")

    # 5. Launch training job via boto3
    print(f"\n[5/5] Launching SageMaker job: {job_name}")
    print(f"  Instance: {instance_type} ({'spot' if use_spot else 'on-demand'})")
    print(f"  Image: {image_uri}")
    print(f"  Hyperparameters: {json.dumps(hyperparameters, indent=4)}")

    training_params = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            # SageMaker passes these as SM_HP_<KEY> env vars
            **hyperparameters,
            # Tell the container to run our entry point
            "sagemaker_program": "entry_point.py",
            "sagemaker_submit_directory": source_s3_uri,
        },
        "InputDataConfig": [
            {
                "ChannelName": "data",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": data_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/{cfg['s3']['results_prefix']}",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": sm_cfg["instance_count"],
            "VolumeSizeInGB": sm_cfg["volume_size_gb"],
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": sm_cfg["max_run_seconds"],
        },
        "CheckpointConfig": {
            "S3Uri": f"s3://{bucket}/checkpoints/{job_name}/",
            "LocalPath": "/opt/ml/checkpoints",
        },
    }

    # Spot instance configuration
    if use_spot:
        training_params["EnableManagedSpotTraining"] = True
        training_params["StoppingCondition"]["MaxWaitTimeInSeconds"] = sm_cfg["max_wait_seconds"]

    response = sm.create_training_job(**training_params)

    print(f"\n{'='*60}")
    print(f"Job submitted: {job_name}")
    print(f"{'='*60}")
    print(f"\nMonitor in AWS Console:")
    print(f"  https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    print(f"\nOr via CLI:")
    print(f"  .venv/bin/aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
    print(f"\nSync results when done:")
    print(f"  python -m cloud.sync --job_name {job_name}")


if __name__ == "__main__":
    main()
