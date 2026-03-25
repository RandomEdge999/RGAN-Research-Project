#!/usr/bin/env python3
"""Launch augmentation experiment on SageMaker.

Usage:
    python -m cloud.launch_augment \
        --csv data/binance/Binance_Data.csv \
        --results_from results/cloud/results

This script:
  1. Uploads the CSV dataset to S3
  2. Uploads the trained model weights + metrics.json to S3
  3. Uploads the source code tarball
  4. Launches a SageMaker training job with two input channels:
     - 'data': the CSV
     - 'models': the trained weights directory
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

import boto3
import yaml


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _create_source_tarball(project_root: Path, tmp_dir: str) -> str:
    """Create a source tarball with augmentation entry point, src/, and pyproject.toml."""
    tar_path = os.path.join(tmp_dir, "sourcedir.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        # Entry point at root of tarball
        tar.add(project_root / "cloud" / "entry_point_augment.py", arcname="entry_point_augment.py")
        # Source code
        tar.add(project_root / "src", arcname="src")
        # pyproject.toml for pip install -e .
        tar.add(project_root / "pyproject.toml", arcname="pyproject.toml")

    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"  Source tarball: {size_mb:.1f} MB")
    return tar_path


def _get_pytorch_image_uri(region: str, pt_version: str, py_version: str) -> str:
    """Get the official AWS Deep Learning Container image URI for PyTorch."""
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
    major_minor = tuple(int(x) for x in pt_version.split(".")[:2])
    ubuntu_ver = "ubuntu22.04" if major_minor >= (2, 5) else "ubuntu20.04"
    cuda_ver = "cu124" if major_minor >= (2, 5) else "cu121"
    tag = f"{pt_version}-gpu-{py_version}-{cuda_ver}-{ubuntu_ver}-sagemaker"
    return f"{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{tag}"


def _upload_models_to_s3(s3_client, bucket: str, results_from: Path, s3_prefix: str):
    """Upload model weights and metrics.json to S3."""
    models_dir = results_from / "models"
    if not models_dir.exists():
        print(f"  ERROR: {models_dir} does not exist")
        sys.exit(1)

    uploaded = 0
    # Upload all .pt files from models/
    for pt_file in sorted(models_dir.glob("*.pt")):
        s3_key = f"{s3_prefix}models/{pt_file.name}"
        s3_client.upload_file(str(pt_file), bucket, s3_key)
        size_mb = pt_file.stat().st_size / (1024 * 1024)
        print(f"    {pt_file.name} ({size_mb:.1f} MB)")
        uploaded += 1

    # Upload metrics.json from the run directory
    metrics_json = results_from / "metrics.json"
    if metrics_json.exists():
        s3_key = f"{s3_prefix}metrics.json"
        s3_client.upload_file(str(metrics_json), bucket, s3_key)
        print(f"    metrics.json")
        uploaded += 1
    else:
        print(f"  WARNING: {metrics_json} not found — model will use state_dict inference")

    # Upload run_config.json if it exists
    run_config = results_from / "run_config.json"
    if run_config.exists():
        s3_key = f"{s3_prefix}run_config.json"
        s3_client.upload_file(str(run_config), bucket, s3_key)
        print(f"    run_config.json")
        uploaded += 1

    return uploaded


def main():
    ap = argparse.ArgumentParser(description="Launch RGAN augmentation experiment on SageMaker")
    ap.add_argument("--csv", required=True, help="Local path to CSV dataset")
    ap.add_argument("--results_from", required=True,
                    help="Local path to training results directory (contains models/ and metrics.json)")
    ap.add_argument("--target_col", default="auto", help="Target column name")
    ap.add_argument("--time_col", default="auto", help="Time column name")
    ap.add_argument("--job_name", default="", help="Custom job name (auto-generated if empty)")
    ap.add_argument("--instance_type", default="", help="Override instance type from config")
    ap.add_argument("--spot", default="", help="Use spot instances (true/false)")

    # Augmentation hyperparameters
    ap.add_argument("--nn_epochs", type=int, default=200)
    ap.add_argument("--nn_patience", type=int, default=25)
    ap.add_argument("--timegan_epochs", type=int, default=100)
    ap.add_argument("--skip_timegan", action="store_true")
    ap.add_argument("--L", type=int, default=None, help="Override lookback (auto-read from metrics.json)")
    ap.add_argument("--H", type=int, default=None, help="Override horizon (auto-read from metrics.json)")

    args = ap.parse_args()

    cfg = _load_config()
    project_root = Path(__file__).parent.parent

    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    results_from = Path(args.results_from)
    if not results_from.exists():
        print(f"ERROR: Results directory not found: {results_from}")
        sys.exit(1)

    models_dir = results_from / "models"
    if not models_dir.exists():
        print(f"ERROR: No models/ subdirectory in {results_from}")
        sys.exit(1)

    # Check for generator
    has_gen = (models_dir / "rgan_generator_ema.pt").exists() or (models_dir / "rgan_generator.pt").exists()
    if not has_gen:
        print(f"ERROR: No RGAN generator found in {models_dir}")
        sys.exit(1)

    role_arn = cfg["sagemaker"]["role_arn"]
    if not role_arn:
        print("ERROR: sagemaker.role_arn is not set in cloud/config.yaml")
        sys.exit(1)

    region = cfg["aws"]["region"]
    bucket = cfg["s3"]["bucket"]
    sm_cfg = cfg["sagemaker"]

    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)

    # 1. Ensure S3 bucket
    print("\n[1/5] Ensuring S3 bucket exists...")
    from cloud.s3_utils import ensure_bucket
    ensure_bucket(bucket, region)

    # 2. Upload dataset
    print("\n[2/5] Uploading dataset...")
    data_key = f"{cfg['s3']['data_prefix']}{csv_path.name}"
    s3.upload_file(str(csv_path), bucket, data_key)
    data_s3_uri = f"s3://{bucket}/{cfg['s3']['data_prefix']}"
    print(f"  Dataset uploaded to s3://{bucket}/{data_key}")

    # 3. Upload trained models
    print("\n[3/5] Uploading trained model weights...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    models_s3_prefix = f"augmentation/models/{timestamp}/"
    n_uploaded = _upload_models_to_s3(s3, bucket, results_from, models_s3_prefix)
    models_s3_uri = f"s3://{bucket}/{models_s3_prefix}"
    print(f"  Uploaded {n_uploaded} files to {models_s3_uri}")

    # 4. Upload source code
    print("\n[4/5] Uploading source code...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = _create_source_tarball(project_root, tmp_dir)
        source_key = "code/sourcedir_augment.tar.gz"
        s3.upload_file(tar_path, bucket, source_key)
    source_s3_uri = f"s3://{bucket}/{source_key}"
    print(f"  Source code uploaded to {source_s3_uri}")

    # 5. Launch training job
    job_name = args.job_name or f"rgan-augment-{timestamp}"
    job_name = job_name.replace("_", "-")

    instance_type = args.instance_type or sm_cfg["instance_type"]
    use_spot = sm_cfg["use_spot"]
    if args.spot:
        use_spot = args.spot.lower() in ("true", "1", "yes")

    image_uri = _get_pytorch_image_uri(region, sm_cfg["pytorch_version"], sm_cfg["python_version"])

    hyperparameters = {
        "target_col": args.target_col,
        "time_col": args.time_col,
        "nn_epochs": str(args.nn_epochs),
        "nn_patience": str(args.nn_patience),
        "timegan_epochs": str(args.timegan_epochs),
    }
    if args.skip_timegan:
        hyperparameters["skip_timegan"] = "true"
    if args.L is not None:
        hyperparameters["L"] = str(args.L)
    if args.H is not None:
        hyperparameters["H"] = str(args.H)

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
            **hyperparameters,
            "sagemaker_program": "entry_point_augment.py",
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
            },
            {
                "ChannelName": "models",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": models_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
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
    }

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
