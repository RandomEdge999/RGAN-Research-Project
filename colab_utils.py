"""Colab-specific utilities for RGAN project."""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional

def is_colab() -> bool:
    """Detect if running in Google Colab."""
    return 'COLAB_GPU' in os.environ

def is_kaggle() -> bool:
    """Detect if running in Kaggle."""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

def is_notebook() -> bool:
    """Detect if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

def install_packages(colab_quiet=True):
    """Install required packages for Colab."""
    # Determine torch installation based on CUDA availability
    if is_colab() and 'COLAB_GPU' in os.environ and os.environ['COLAB_GPU']:
        # Colab provides CUDA 11.8 or 12.1; torch 2.2+ works
        torch_cmd = "torch>=2.2"
    else:
        torch_cmd = "torch>=2.2"
    
    packages = [
        torch_cmd,
        "numpy>=1.22",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "statsmodels>=0.13",
        "scikit-learn>=1.1",
        "rich>=13.0",
        "plotly>=5.22",
        "pyyaml",
        "psutil>=5.9",
        "datasets>=2.0",
        "ipywidgets",  # for interactive plots
    ]
    
    cmd = [sys.executable, "-m", "pip", "install"]
    if colab_quiet:
        cmd.append("-qq")
    cmd.extend(packages)
    
    print("Installing packages...")
    subprocess.check_call(cmd)
    print("Package installation complete.")

def setup_matplotlib_backend():
    """Set matplotlib backend for notebooks."""
    if is_colab() or is_kaggle() or is_notebook():
        import matplotlib
        matplotlib.use('inline')
        matplotlib.rcParams['figure.figsize'] = (10, 6)
        matplotlib.rcParams['figure.dpi'] = 100
        print(f"Matplotlib backend set to {matplotlib.get_backend()}")

def mount_google_drive(force_remount=False):
    """Mount Google Drive in Colab."""
    if not is_colab():
        return None
    
    from google.colab import drive
    drive.mount('/content/drive', force_remount=force_remount)
    return Path('/content/drive/MyDrive')

def setup_paths(use_drive=False, drive_project_path="RGAN-Research-Project"):
    """Set up project paths for Colab."""
    if is_colab():
        # Colab's temporary workspace
        colab_root = Path('/content')
        project_path = colab_root / 'RGAN-Research-Project'
        
        # Clone or copy project
        if not project_path.exists():
            # Check if we have a local repo (if running from GitHub)
            # For now, we assume the notebook is in the repo root
            # In practice, we'll copy from current directory
            pass
        
        if use_drive:
            drive_root = mount_google_drive()
            drive_project = drive_root / drive_project_path
            drive_project.mkdir(parents=True, exist_ok=True)
            return project_path, drive_project
        
        return project_path, None
    
    # Local execution
    return Path.cwd(), None

def download_sample_dataset(dataset_name="nasa_power_austin_hourly"):
    """Download a sample dataset using the fetch script."""
    from scripts.fetch_natural_datasets import main as fetch_main
    import sys
    
    # Save original argv
    orig_argv = sys.argv
    
    # Mock command line arguments for the fetch script
    # The script fetches all datasets; we need to modify it to fetch only one
    # For simplicity, we'll just run it and let it fetch all (they're small)
    try:
        print(f"Downloading {dataset_name} dataset...")
        # We'll need to import and run the fetch function directly
        # Since the script is modular, we can import SPECS and run the fetcher
        from scripts.fetch_natural_datasets import SPECS
        from pathlib import Path
        
        DATA_ROOT = Path('data')
        
        for spec in SPECS:
            if spec.slug == dataset_name:
                print(f"Fetching {spec.title}...")
                df = spec.fetcher()
                out_dir = DATA_ROOT / spec.slug
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / f"{spec.slug}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved to {csv_path}")
                return csv_path
        
        print(f"Dataset {dataset_name} not found in SPECS. Using synthetic data.")
        return None
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Will use synthetic data instead.")
        return None

def create_synthetic_dataset(n_samples=1000, L=30, H=5):
    """Create a synthetic sine wave dataset for demo purposes."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate time index
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples + L + H)]
    
    # Generate sine wave with noise
    t = np.arange(len(timestamps)) * 0.1
    signal = 10 * np.sin(t) + 0.5 * np.random.randn(len(timestamps))
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'value': signal,
        'noise': 0.1 * np.random.randn(len(timestamps))
    })
    
    # Save to data/synthetic_demo/
    output_dir = Path('data') / 'synthetic_demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'synthetic_demo.csv'
    df.to_csv(csv_path, index=False)
    
    # Create README
    readme = """# Synthetic Demo Dataset
- **Rows:** {n_samples}
- **Resolution:** Hourly
- **Period:** {start} to {end}
- **Source:** Synthetic (sine wave + Gaussian noise)
- **Target column:** `value`
- **Time column:** `time`
- **Noise type:** Additive Gaussian, σ=0.5
- **Purpose:** Demonstration and smoke testing
""".format(n_samples=len(df), start=start_time, end=timestamps[-1])
    
    (output_dir / 'README.md').write_text(readme)
    
    print(f"Created synthetic dataset at {csv_path}")
    return csv_path

def setup_project(use_drive=False, download_data=True, synthetic_fallback=True):
    """Complete project setup for Colab."""
    print("=" * 80)
    print("RGAN Research Project - Colab Setup")
    print("=" * 80)
    
    # 1. Install packages
    install_packages()
    
    # 2. Setup matplotlib
    setup_matplotlib_backend()
    
    # 3. Setup paths
    project_path, drive_path = setup_paths(use_drive=use_drive)
    print(f"Project path: {project_path}")
    if drive_path:
        print(f"Drive backup: {drive_path}")
    
    # 4. Ensure we're in the project directory
    os.chdir(project_path)
    
    # 5. Install the package in development mode
    if (project_path / 'pyproject.toml').exists():
        print("Installing RGAN package in development mode...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "."
        ])
    
    # 6. Download or create data
    data_path = None
    if download_data:
        try:
            data_path = download_sample_dataset("nasa_power_austin_hourly")
        except Exception as e:
            print(f"Failed to download real dataset: {e}")
            data_path = None
        
        if data_path is None and synthetic_fallback:
            print("Creating synthetic dataset as fallback...")
            data_path = create_synthetic_dataset()
    
    print("Setup complete!")
    return {
        'project_path': project_path,
        'drive_path': drive_path,
        'data_path': data_path
    }