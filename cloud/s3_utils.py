"""S3 utilities for uploading data and downloading results."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError


def get_config() -> dict:
    """Load cloud/config.yaml and return as dict."""
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cloud config not found at {config_path}. "
            "Create cloud/config.yaml with your AWS settings (see README for format)."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_s3_client(region: Optional[str] = None):
    """Create an S3 client using the configured region."""
    cfg = get_config()
    region = region or cfg["aws"]["region"]
    return boto3.client("s3", region_name=region)


def ensure_bucket(bucket: Optional[str] = None, region: Optional[str] = None) -> str:
    """Create S3 bucket if it doesn't exist. Returns bucket name."""
    cfg = get_config()
    bucket = bucket or cfg["s3"]["bucket"]
    region = region or cfg["aws"]["region"]
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket s3://{bucket} already exists.")
    except ClientError:
        print(f"  Creating bucket s3://{bucket} in {region}...")
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"  Created s3://{bucket}")

    return bucket


def upload_directory(local_dir: str, s3_prefix: str,
                     bucket: Optional[str] = None,
                     region: Optional[str] = None) -> int:
    """Upload a local directory to S3. Returns number of files uploaded."""
    cfg = get_config()
    bucket = bucket or cfg["s3"]["bucket"]
    region = region or cfg["aws"]["region"]
    s3 = boto3.client("s3", region_name=region)

    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    count = 0
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            key = f"{s3_prefix}{file_path.relative_to(local_path)}"
            print(f"  Uploading {file_path.name} -> s3://{bucket}/{key}")
            s3.upload_file(str(file_path), bucket, key)
            count += 1

    print(f"  Uploaded {count} files to s3://{bucket}/{s3_prefix}")
    return count


def upload_file(local_path: str, s3_key: str,
                bucket: Optional[str] = None,
                region: Optional[str] = None) -> str:
    """Upload a single file to S3. Returns the S3 URI."""
    cfg = get_config()
    bucket = bucket or cfg["s3"]["bucket"]
    region = region or cfg["aws"]["region"]
    s3 = boto3.client("s3", region_name=region)

    print(f"  Uploading {local_path} -> s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def download_directory(s3_prefix: str, local_dir: str,
                       bucket: Optional[str] = None,
                       region: Optional[str] = None) -> int:
    """Download all files under an S3 prefix to a local directory."""
    cfg = get_config()
    bucket = bucket or cfg["s3"]["bucket"]
    region = region or cfg["aws"]["region"]
    s3 = boto3.client("s3", region_name=region)

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(s3_prefix):]
            if not rel:
                continue
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading s3://{bucket}/{key} -> {dest}")
            s3.download_file(bucket, key, str(dest))
            count += 1

    print(f"  Downloaded {count} files to {local_dir}")
    return count


def list_results(bucket: Optional[str] = None,
                 region: Optional[str] = None) -> list[str]:
    """List all result prefixes (job folders) in S3."""
    cfg = get_config()
    bucket = bucket or cfg["s3"]["bucket"]
    region = region or cfg["aws"]["region"]
    prefix = cfg["s3"]["results_prefix"]
    s3 = boto3.client("s3", region_name=region)

    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    prefixes = [p["Prefix"] for p in result.get("CommonPrefixes", [])]
    return prefixes
