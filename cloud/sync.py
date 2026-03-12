#!/usr/bin/env python3
"""Download SageMaker training results to local machine.

Usage:
    python -m cloud.sync --job_name rgan-20260312-143000
    python -m cloud.sync --list              # list all completed jobs
    python -m cloud.sync --latest            # sync most recent job
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import yaml


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_jobs(region: str) -> list[dict]:
    """List recent SageMaker training jobs."""
    sm = boto3.client("sagemaker", region_name=region)
    response = sm.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
        NameContains="rgan",
    )
    jobs = []
    for job in response.get("TrainingJobSummaries", []):
        jobs.append({
            "name": job["TrainingJobName"],
            "status": job["TrainingJobStatus"],
            "created": job["CreationTime"].strftime("%Y-%m-%d %H:%M"),
        })
    return jobs


def get_job_output_path(job_name: str, region: str) -> str:
    """Get the S3 output path for a training job."""
    sm = boto3.client("sagemaker", region_name=region)
    response = sm.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]
    print(f"  Job status: {status}")

    if status not in ("Completed", "Stopped"):
        if status == "Failed":
            reason = response.get("FailureReason", "Unknown")
            print(f"  Failure reason: {reason}")
        return ""

    # model artifacts
    return response.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")


def sync_job(job_name: str, local_dir: str, region: str, bucket: str):
    """Download and extract results for a training job."""
    print(f"\nSyncing job: {job_name}")

    model_uri = get_job_output_path(job_name, region)
    if not model_uri:
        print("  No output artifacts available.")
        return

    print(f"  Model artifacts: {model_uri}")

    # Download model.tar.gz
    s3 = boto3.client("s3", region_name=region)
    # Parse S3 URI
    parts = model_uri.replace("s3://", "").split("/", 1)
    s3_bucket = parts[0]
    s3_key = parts[1]

    local_path = Path(local_dir) / job_name
    local_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        print(f"  Downloading model.tar.gz...")
        s3.download_file(s3_bucket, s3_key, tmp.name)

        print(f"  Extracting to {local_path}...")
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(path=local_path)

    # Also check for output data (results saved separately)
    results_prefix = f"results/{job_name}/output/"
    try:
        paginator = s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=results_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(results_prefix):]
                if not rel:
                    continue
                dest = local_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
                count += 1
        if count:
            print(f"  Downloaded {count} additional output files.")
    except Exception as e:
        print(f"  Note: Could not fetch additional outputs: {e}")

    print(f"\n  Results saved to: {local_path}")

    # List what we got
    files = sorted(local_path.rglob("*"))
    files = [f for f in files if f.is_file()]
    if files:
        print(f"\n  Files ({len(files)}):")
        for f in files[:20]:
            print(f"    {f.relative_to(local_path)}")
        if len(files) > 20:
            print(f"    ... and {len(files) - 20} more")


def main():
    ap = argparse.ArgumentParser(description="Sync SageMaker training results")
    ap.add_argument("--job_name", default="", help="Training job name to sync")
    ap.add_argument("--list", action="store_true", help="List recent training jobs")
    ap.add_argument("--latest", action="store_true", help="Sync most recent completed job")
    ap.add_argument("--output_dir", default="./results/cloud",
                    help="Local directory for downloaded results")
    args = ap.parse_args()

    cfg = _load_config()
    region = cfg["aws"]["region"]
    bucket = cfg["s3"]["bucket"]

    if args.list:
        jobs = list_jobs(region)
        if not jobs:
            print("No RGAN training jobs found.")
            return
        print(f"\n{'Job Name':<40} {'Status':<15} {'Created'}")
        print("-" * 75)
        for j in jobs:
            print(f"{j['name']:<40} {j['status']:<15} {j['created']}")
        return

    if args.latest:
        jobs = list_jobs(region)
        completed = [j for j in jobs if j["status"] == "Completed"]
        if not completed:
            print("No completed RGAN jobs found.")
            return
        args.job_name = completed[0]["name"]
        print(f"Syncing latest completed job: {args.job_name}")

    if not args.job_name:
        print("Provide --job_name, --latest, or --list")
        sys.exit(1)

    sync_job(args.job_name, args.output_dir, region, bucket)


if __name__ == "__main__":
    main()
