#!/usr/bin/env python3
"""Launch all 15 augmentation jobs on SageMaker in parallel.

Usage:
    source .venv/bin/activate && python -m cloud.launch_augment_batch
"""

import subprocess
import sys
from pathlib import Path

# 15 runs: 3 datasets × 5 seeds
JOBS = [
    # Binance BTCBVOL
    {"csv": "data/binance/Binance_Data.csv",
     "results_from": "results/cloud/rgan-binance-s42-eval4/results",
     "job_name": "rgan-aug-binance-s42-v2"},
    {"csv": "data/binance/Binance_Data.csv",
     "results_from": "results/cloud/rgan-binance-s43-eval4/results",
     "job_name": "rgan-aug-binance-s43-v2"},
    {"csv": "data/binance/Binance_Data.csv",
     "results_from": "results/cloud/rgan-binance-s44-eval4/results",
     "job_name": "rgan-aug-binance-s44-v2"},
    {"csv": "data/binance/Binance_Data.csv",
     "results_from": "results/cloud/rgan-binance-s45-eval4/results",
     "job_name": "rgan-aug-binance-s45-v2"},
    {"csv": "data/binance/Binance_Data.csv",
     "results_from": "results/cloud/rgan-binance-s46-eval4/results",
     "job_name": "rgan-aug-binance-s46-v2"},
    # NASA POWER Denver
    {"csv": "data/nasa_power_denver_2018_2023_hourly/nasa_power_denver_2018_2023_hourly.csv",
     "results_from": "results/cloud/rgan-nasa-s42-e100-v2/results",
     "job_name": "rgan-aug-nasa-s42-v2"},
    {"csv": "data/nasa_power_denver_2018_2023_hourly/nasa_power_denver_2018_2023_hourly.csv",
     "results_from": "results/cloud/rgan-nasa-s43-e100-v2/results",
     "job_name": "rgan-aug-nasa-s43-v2"},
    {"csv": "data/nasa_power_denver_2018_2023_hourly/nasa_power_denver_2018_2023_hourly.csv",
     "results_from": "results/cloud/rgan-nasa-s44-e100-v2/results",
     "job_name": "rgan-aug-nasa-s44-v2"},
    {"csv": "data/nasa_power_denver_2018_2023_hourly/nasa_power_denver_2018_2023_hourly.csv",
     "results_from": "results/cloud/rgan-nasa-s45-e100-v2/results",
     "job_name": "rgan-aug-nasa-s45-v2"},
    {"csv": "data/nasa_power_denver_2018_2023_hourly/nasa_power_denver_2018_2023_hourly.csv",
     "results_from": "results/cloud/rgan-nasa-s46-e100-v2/results",
     "job_name": "rgan-aug-nasa-s46-v2"},
    # Wind Turbine SCADA
    {"csv": "data/wind_turbine_scada/wind_turbine_scada.csv",
     "results_from": "results/cloud/rgan-wind-s42-e100-v2/results",
     "job_name": "rgan-aug-wind-s42-v2"},
    {"csv": "data/wind_turbine_scada/wind_turbine_scada.csv",
     "results_from": "results/cloud/rgan-wind-s43-e100-v2/results",
     "job_name": "rgan-aug-wind-s43-v2"},
    {"csv": "data/wind_turbine_scada/wind_turbine_scada.csv",
     "results_from": "results/cloud/rgan-wind-s44-e100-v2/results",
     "job_name": "rgan-aug-wind-s44-v2"},
    {"csv": "data/wind_turbine_scada/wind_turbine_scada.csv",
     "results_from": "results/cloud/rgan-wind-s45-e100-v2/results",
     "job_name": "rgan-aug-wind-s45-v2"},
    {"csv": "data/wind_turbine_scada/wind_turbine_scada.csv",
     "results_from": "results/cloud/rgan-wind-s46-e100-v2/results",
     "job_name": "rgan-aug-wind-s46-v2"},
]


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Validate all paths first
    print("Validating all 15 jobs...")
    for job in JOBS:
        csv_path = project_root / job["csv"]
        results_path = project_root / job["results_from"]
        if not csv_path.exists():
            print(f"  MISSING CSV: {csv_path}")
            sys.exit(1)
        if not results_path.exists():
            print(f"  MISSING RESULTS: {results_path}")
            sys.exit(1)
        gen_ema = results_path / "models" / "rgan_generator_ema.pt"
        gen_std = results_path / "models" / "rgan_generator.pt"
        if not gen_ema.exists() and not gen_std.exists():
            print(f"  MISSING GENERATOR: {results_path / 'models'}")
            sys.exit(1)
    print("All 15 jobs validated.\n")

    # Launch all jobs
    launched = []
    failed = []
    for i, job in enumerate(JOBS, 1):
        print(f"[{i}/15] Launching {job['job_name']}...")
        cmd = [
            sys.executable, "-m", "cloud.launch_augment",
            "--csv", str(project_root / job["csv"]),
            "--results_from", str(project_root / job["results_from"]),
            "--job_name", job["job_name"],
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        if result.returncode == 0:
            launched.append(job["job_name"])
            print(f"  OK: {job['job_name']}")
        else:
            failed.append(job["job_name"])
            print(f"  FAILED: {job['job_name']}")
            print(f"  STDERR: {result.stderr[-500:]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Launched: {len(launched)}/15")
    if failed:
        print(f"Failed:   {len(failed)}/15")
        for f in failed:
            print(f"  - {f}")
    print(f"{'='*60}")

    # Print monitoring commands
    if launched:
        print("\nMonitor all jobs:")
        print("  source .venv/bin/activate && aws sagemaker list-training-jobs --sort-order Descending --max-results 15")
        print("\nJob names:")
        for name in launched:
            print(f"  {name}")


if __name__ == "__main__":
    main()
