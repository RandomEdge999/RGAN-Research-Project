# RGAN Automated Pipeline
# Runs the experiment and launches the dashboard

$ErrorActionPreference = "Stop"

# 1. Run the Experiment
Write-Host "Starting RGAN Experiment..." -ForegroundColor Cyan
python run_experiment.py --csv src/rgan/Binance_Data.csv --target index_value --time_col calc_time --results_dir results_auto --epochs 50 --gan_variant wgan-gp

if ($LASTEXITCODE -ne 0) {
    Write-Error "Experiment failed!"
    exit 1
}

Write-Host "Experiment completed successfully." -ForegroundColor Green

# 2. Launch Dashboard
$dashboardDir = "web_dashboard"
$url = "http://localhost:5173" # Default Vite port

if (Test-Path $dashboardDir) {
    Write-Host "Launching Dashboard..." -ForegroundColor Cyan
    
    # Check if port 5173 is already in use (simple check)
    $portActive = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue

    if (-not $portActive) {
        Write-Host "Starting Vite server..." -ForegroundColor Yellow
        # Start npm run dev in a new window
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $dashboardDir; npm run dev"
        Start-Sleep -Seconds 5 # Wait for server to start
    } else {
        Write-Host "Dashboard server appears to be running." -ForegroundColor Green
    }

    # Open Browser
    Write-Host "Opening Dashboard in Browser..." -ForegroundColor Cyan
    Start-Process $url
} else {
    Write-Error "Dashboard directory not found!"
}
