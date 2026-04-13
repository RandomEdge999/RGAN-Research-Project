#!/usr/bin/env python3
"""Generate a Colab notebook for the RGAN project."""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

def create_colab_notebook():
    """Create a Colab notebook with RGAN demo."""
    
    nb = new_notebook()
    
    # Title and introduction
    nb.cells.append(new_markdown_cell("""# RGAN Research Project - Google Colab Demo

**Noise-Resilient Time-Series Forecasting with LSTM-based GANs**

This notebook provides a complete, runnable demonstration of the RGAN (Recurrent Generative Adversarial Network) for time-series forecasting. The project trains an LSTM generator and discriminator adversarially to produce forecasts robust to real-world noise.

## What This Notebook Does

1. **Setup**: Installs all required dependencies and configures the environment
2. **Data Preparation**: Downloads a real-world dataset (NASA POWER Austin hourly meteorology) or creates synthetic data
3. **Training**: Runs RGAN training with sensible defaults for Colab
4. **Evaluation**: Computes metrics and visualizes results
5. **Optional Augmentation**: Demonstrates synthetic data generation and augmentation experiments

## Requirements

- Google Colab (free GPU)
- Internet connection (for dataset download)
- Approximately 2GB RAM, 5GB disk space

**Expected runtime**: ~10-15 minutes for the demo configuration

Let's get started!"""))
    
    # Cell 1: Setup and installation
    nb.cells.append(new_markdown_cell("""## 1. Setup & Installation

First, we'll install all required packages and set up the project structure."""))
    
    nb.cells.append(new_code_cell("""# Install nbformat for notebook generation (if not already installed)
!pip install -q nbformat

# Import our Colab utilities
import sys
sys.path.insert(0, '/content')

# Clone or copy the RGAN repository
import os
from pathlib import Path

# Check if we're in Colab
IN_COLAB = 'COLAB_GPU' in os.environ
if IN_COLAB:
    print("Running in Google Colab")
    # Clone the repository
    !git clone https://github.com/RandomEdge999/RGAN-Research-Project.git /content/RGAN-Research-Project 2>/dev/null || echo "Repository already exists"
    os.chdir('/content/RGAN-Research-Project')
else:
    print("Running locally")
    # Assume we're already in the project directory"""))
    
    nb.cells.append(new_code_cell("""# Import colab utilities
from colab_utils import setup_project

# Complete project setup
setup_result = setup_project(
    use_drive=False,  # Set to True to save results to Google Drive
    download_data=True,
    synthetic_fallback=True
)

print(f"Project ready at: {setup_result['project_path']}")
if setup_result['data_path']:
    print(f"Data available at: {setup_result['data_path']}")"""))
    
    # Cell 2: Configuration
    nb.cells.append(new_markdown_cell("""## 2. Configuration

We'll configure a fast demo run with reduced parameters suitable for Colab's runtime limits."""))
    
    nb.cells.append(new_code_cell("""import yaml
import json
from pathlib import Path

# Demo configuration
demo_config = {
    'data_path': str(setup_result['data_path']) if setup_result['data_path'] else 'data/synthetic_demo/synthetic_demo.csv',
    'target_col': 'value' if 'synthetic_demo' in str(setup_result.get('data_path', '')) else 'T2M',
    'time_col': 'time',
    'L': 30,  # Lookback window
    'H': 5,   # Forecast horizon
    'epochs': 15,  # Reduced for demo
    'batch_size': 64,
    'max_train_windows': 1000,  # Limit for speed
    'noise_levels': '0,0.05,0.1',
    'skip_classical': True,  # Skip slow ARIMA/ARMA for demo
    'skip_noise_robustness': False,
    'gan_variant': 'wgan-gp',
    'units_g': 64,
    'units_d': 64,
    'g_layers': 1,
    'd_layers': 1,
    'dropout': 0.1,
    'lr_g': 5e-4,
    'lr_d': 5e-4,
    'lambda_reg': 0.5,
    'results_dir': './results/colab_demo',
    'deterministic': False,
    'seed': 42,
}

# Save config for reference
config_path = Path('demo_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(demo_config, f)

print("Demo configuration:")
for key, value in demo_config.items():
    print(f"  {key}: {value}")"""))
    
    # Cell 3: Run training
    nb.cells.append(new_markdown_cell("""## 3. Run RGAN Training

Now we'll run the main training pipeline with our demo configuration."""))
    
    nb.cells.append(new_code_cell("""import subprocess
import sys

# Build command from config
cmd = [
    sys.executable, "-u", "-m", "rgan.scripts.run_training",
    "--csv", demo_config['data_path'],
    "--target", demo_config['target_col'],
    "--time_col", demo_config['time_col'],
    "--L", str(demo_config['L']),
    "--H", str(demo_config['H']),
    "--epochs", str(demo_config['epochs']),
    "--batch_size", str(demo_config['batch_size']),
    "--max_train_windows", str(demo_config['max_train_windows']),
    "--noise_levels", demo_config['noise_levels'],
    "--gan_variant", demo_config['gan_variant'],
    "--units_g", str(demo_config['units_g']),
    "--units_d", str(demo_config['units_d']),
    "--g_layers", str(demo_config['g_layers']),
    "--d_layers", str(demo_config['d_layers']),
    "--dropout", str(demo_config['dropout']),
    "--lr_g", str(demo_config['lr_g']),
    "--lr_d", str(demo_config['lr_d']),
    "--lambda_reg", str(demo_config['lambda_reg']),
    "--results_dir", demo_config['results_dir'],
    "--seed", str(demo_config['seed']),
]

if demo_config['skip_classical']:
    cmd.append("--skip_classical")

if demo_config['skip_noise_robustness']:
    cmd.append("--skip_noise_robustness")

if demo_config['deterministic']:
    cmd.append("--deterministic")

print("Running command:")
print(" ".join(cmd))
print("\\n" + "="*80)

# Execute training
result = subprocess.run(cmd, capture_output=True, text=True)

# Print output
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print(f"\\nTraining completed with exit code: {result.returncode}")
if result.returncode != 0:
    print("Training failed. Check error messages above.")
else:
    print("Training successful!")"""))
    
    # Cell 4: Load and display results
    nb.cells.append(new_markdown_cell("""## 4. Load and Display Results

Let's load the generated metrics and visualize the outcomes."""))
    
    nb.cells.append(new_code_cell("""import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = 'colab' if IN_COLAB else 'notebook'

results_dir = Path(demo_config['results_dir'])

# Load metrics
metrics_path = results_dir / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print("Key Metrics Summary:")
    print("=" * 50)
    
    # Extract RGAN metrics
    if 'rgan' in metrics:
        rgan_metrics = metrics['rgan']
        print(f"RGAN Test RMSE: {rgan_metrics.get('test_stats', {}).get('rmse', 'N/A'):.6f}")
        print(f"RGAN Test MAE: {rgan_metrics.get('test_stats', {}).get('mae', 'N/A'):.6f}")
    
    # Extract other model metrics
    model_keys = [k for k in metrics.keys() if k not in ('L', 'H', 'dataset', 'environment')]
    for model in model_keys:
        if model != 'rgan' and isinstance(metrics[model], dict):
            stats = metrics[model].get('test_stats', {})
            if stats:
                print(f"{model.upper()} Test RMSE: {stats.get('rmse', 'N/A'):.6f}")
else:
    print(f"Metrics file not found at {metrics_path}")"""))
    
    nb.cells.append(new_code_cell("""# Load noise robustness table if available
noise_table_path = results_dir / 'noise_robustness_table.csv'
if noise_table_path.exists():
    df_noise = pd.read_csv(noise_table_path)
    print("\\nNoise Robustness Table:")
    print(df_noise.to_string(index=False))
    
    # Plot noise robustness
    if 'noise_level' in df_noise.columns and 'rmse' in df_noise.columns:
        plt.figure(figsize=(10, 6))
        for model in df_noise['model'].unique():
            model_data = df_noise[df_noise['model'] == model]
            plt.plot(model_data['noise_level'], model_data['rmse'], marker='o', label=model)
        plt.xlabel('Noise Level (σ)')
        plt.ylabel('RMSE')
        plt.title('Model Robustness to Additive Gaussian Noise')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
else:
    print("Noise robustness table not generated (may have been skipped)")"""))
    
    nb.cells.append(new_code_cell("""# Load and display training curves
training_plots = list(results_dir.glob('*training_curves*.png'))
if training_plots:
    print(f"\\nTraining plots generated: {[p.name for p in training_plots]}")
    for plot_path in training_plots[:2]:  # Show first two
        from IPython.display import Image
        display(Image(filename=str(plot_path)))
else:
    print("No training plots found")"""))
    
    # Cell 5: Optional augmentation experiment
    nb.cells.append(new_markdown_cell("""## 5. Optional: Augmentation Experiment

If you have time and want to see synthetic data generation, run this optional augmentation experiment."""))
    
    nb.cells.append(new_code_cell("""# Only run if training was successful and we have a model
augmentation_cmd = [
    sys.executable, "-u", "-m", "rgan.scripts.run_augmentation",
    "--csv", demo_config['data_path'],
    "--target", demo_config['target_col'],
    "--time_col", demo_config['time_col'],
    "--L", str(demo_config['L']),
    "--H", str(demo_config['H']),
    "--results_from", str(results_dir),
    "--results_dir", str(results_dir / "augmentation"),
    "--skip_timegan",  # Skip TimeGAN for speed
    "--nn_epochs", "10",  # Reduced for demo
    "--nn_patience", "5",
]

print("Running augmentation experiment...")
print(" ".join(augmentation_cmd))

aug_result = subprocess.run(augmentation_cmd, capture_output=True, text=True)
print(aug_result.stdout)
if aug_result.stderr:
    print("STDERR:", aug_result.stderr)"""))
    
    nb.cells.append(new_code_cell("""# Display augmentation results if available
aug_dir = results_dir / "augmentation"
if aug_dir.exists():
    # Show synthetic quality table
    quality_table = aug_dir / "synthetic_quality_table.csv"
    if quality_table.exists():
        df_quality = pd.read_csv(quality_table)
        print("\\nSynthetic Quality Metrics:")
        print(df_quality.to_string(index=False))
    
    # Show augmentation comparison table
    aug_table = aug_dir / "data_augmentation_table.csv"
    if aug_table.exists():
        df_aug = pd.read_csv(aug_table)
        print("\\nData Augmentation Results (Real vs Mixed):")
        print(df_aug.to_string(index=False))"""))
    
    # Cell 6: Save results to Drive (optional)
    nb.cells.append(new_markdown_cell("""## 6. Save Results to Google Drive (Optional)

If you mounted Google Drive earlier, you can save your results there for later access."""))
    
    nb.cells.append(new_code_cell("""if IN_COLAB and setup_result.get('drive_path'):
    from google.colab import drive
    import shutil
    
    drive_results = setup_result['drive_path'] / 'RGAN_Results'
    drive_results.mkdir(exist_ok=True)
    
    # Copy results directory
    dest = drive_results / results_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(results_dir, dest)
    
    print(f"Results saved to Google Drive: {dest}")
else:
    print("Google Drive not mounted or not in Colab. Results remain in Colab's temporary storage.")"""))
    
    # Cell 7: Next steps and documentation
    nb.cells.append(new_markdown_cell("""## 7. Next Steps & Documentation

### Running Full Experiments
For more serious experiments:
1. Use full datasets (download via `scripts/fetch_natural_datasets.py`)
2. Increase training epochs (e.g., `--epochs 200`)
3. Enable all baselines (remove `--skip_classical`)
4. Use larger model sizes (`--units_g 128 --units_d 128 --g_layers 2 --d_layers 2`)

### Available Datasets
The project includes scripts to download multiple real-world noisy datasets:
- NASA POWER meteorological data (Austin, Phoenix, Denver)
- NOAA tide and buoy data
- USGS streamflow data
- Beijing air quality, household power, gas sensors, etc.

### Project Structure
- `src/rgan/` - Core implementation
- `scripts/` - Data fetching and utilities
- `cloud/` - AWS SageMaker deployment
- `docs/` - Detailed documentation

### CLI Usage Examples
```bash
# Full training run
rgan-train --csv data/nasa_power_austin_hourly/nasa_power_austin_hourly.csv --target T2M --epochs 200

# Noise robustness sweep
rgan-train --csv data/beijing_air/beijing_air.csv --target PM2.5 --noise_levels 0,0.01,0.05,0.1,0.2

# Augmentation experiment
rgan-augment --csv data/household_power/household_power.csv --results_from results/experiment_20250101_120000
```

### Troubleshooting
- **CUDA out of memory**: Reduce `--batch_size` or `--max_train_windows`
- **Training too slow**: Use `--skip_classical` and `--skip_noise_robustness`
- **Symlink errors**: Ignore (non-critical convenience feature)

---

**Enjoy experimenting with RGAN!**"""))
    
    return nb

def main():
    """Generate and save the notebook."""
    notebook = create_colab_notebook()
    
    # Write to file
    output_path = Path("RGAN_Colab_Demo.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, default=lambda x: x.__dict__)
    
    print(f"Generated Colab notebook: {output_path}")
    print(f"Size: {len(json.dumps(notebook))} bytes, {len(notebook.cells)} cells")

if __name__ == "__main__":
    main()