# RGAN Full Experiment Runner
# This script runs a complete experiment using the Binance dataset with settings optimized for stability and GPU usage.

Write-Host "Starting RGAN Full Experiment..." -ForegroundColor Cyan

# Define the command with robust parameters
# --num_workers 0: Prevents hanging on WSL/Windows
# --gan_variant wgan-gp: Uses the stable Wasserstein GAN with Gradient Penalty
# --epochs 100: A reasonable length for a full run
$cmd = "python run_experiment.py --csv src/rgan/Binance_Data.csv --target index_value --time_col calc_time --results_dir results_binance_full --epochs 100 --gan_variant wgan-gp --num_workers 0"

Write-Host "Executing: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

Write-Host "Experiment Complete. Dashboard should launch automatically." -ForegroundColor Green
