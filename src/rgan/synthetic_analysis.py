"""
Synthetic Data Analysis Module

Provides comprehensive tools for:
1. Calculating synthetic data quality metrics (Fréchet Distance, Variance Difference, Discrimination Score)
2. Generating synthetic sequences from trained generators
3. Creating comparison tables and visualizations
4. Analyzing real vs synthetic data distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.linalg import sqrtm
import warnings

from .plots import _PALETTE, _style_axes, _finalise_static, _write_interactive, _HAS_PLOTLY, _ensure_path

warnings.filterwarnings('ignore')

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None


# ============================================================================
# PART 1: SYNTHETIC DATA QUALITY METRICS
# ============================================================================

def frechet_distance(real_samples, fake_samples):
    """
    Calculate Fréchet Distance between real and synthetic data distributions.

    Measures how similar two distributions are. Lower values indicate more similar distributions.

    Args:
        real_samples: np.ndarray of shape (n_samples, sequence_length, features) or (n_samples, features)
        fake_samples: np.ndarray of same shape as real_samples

    Returns:
        float: Fréchet Distance value (non-negative)
    """
    # Flatten to 2D if needed
    if real_samples.ndim == 3:
        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        fake_flat = fake_samples.reshape(fake_samples.shape[0], -1)
    else:
        real_flat = real_samples
        fake_flat = fake_samples

    # Calculate mean and covariance
    mu_real = np.mean(real_flat, axis=0)
    mu_fake = np.mean(fake_flat, axis=0)

    sigma_real = np.cov(real_flat.T)
    sigma_fake = np.cov(fake_flat.T)

    # Handle 1D covariance matrices
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_fake.ndim == 0:
        sigma_fake = np.array([[sigma_fake]])

    # Fréchet Distance = ||mu_real - mu_fake||^2 + Trace(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    mean_diff = np.sum((mu_real - mu_fake) ** 2)

    # Compute sqrt of matrix product
    try:
        sigma_prod = sqrtm(sigma_real @ sigma_fake)
        cov_trace = np.trace(sigma_real + sigma_fake - 2 * sigma_prod)
    except:
        # Fallback if matrix sqrt fails
        cov_trace = np.trace(sigma_real) + np.trace(sigma_fake)

    fd = mean_diff + cov_trace
    return float(np.real(fd))


def variance_difference(real_samples, fake_samples):
    """
    Compare variance statistics between real and synthetic data.

    Args:
        real_samples: np.ndarray of shape (n_samples, sequence_length, features) or (n_samples, features)
        fake_samples: np.ndarray of same shape as real_samples

    Returns:
        dict: {
            'var_real': float,
            'var_fake': float,
            'abs_diff': float,
            'rel_diff': float  # (var_fake - var_real) / var_real * 100
        }
    """
    real_flat = real_samples.flatten()
    fake_flat = fake_samples.flatten()

    var_real = float(np.var(real_flat))
    var_fake = float(np.var(fake_flat))

    abs_diff = float(np.abs(var_fake - var_real))
    rel_diff = float((var_fake - var_real) / (var_real + 1e-8) * 100)

    return {
        'var_real': var_real,
        'var_fake': var_fake,
        'abs_diff': abs_diff,
        'rel_diff': rel_diff
    }


def _extract_time_series_features(samples):
    """
    Extract rich features from time series for classification.
    Fully vectorized — no per-sample Python loop.

    Args:
        samples: np.ndarray of shape (n_samples, sequence_length, features) or (n_samples, sequence_length)

    Returns:
        np.ndarray of shape (n_samples, n_features)
    """
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

    if samples.ndim == 3:
        samples = samples.squeeze()  # Remove feature dimension if it's 1

    # samples: (n_samples, seq_len)
    n_samples, seq_len = samples.shape

    # Basic statistics — vectorized across all samples
    mean = np.mean(samples, axis=1)
    std = np.std(samples, axis=1)
    min_val = np.min(samples, axis=1)
    max_val = np.max(samples, axis=1)
    skewness = sp_skew(samples, axis=1)
    kurt = sp_kurtosis(samples, axis=1)

    # Trend (linear regression slope) — vectorized via least-squares formula
    x = np.arange(seq_len, dtype=samples.dtype)
    x_mean = x.mean()
    x_centered = x - x_mean
    samples_centered = samples - mean[:, None]
    trend = (samples_centered @ x_centered) / (x_centered @ x_centered)

    # Autocorrelation at lag 1 — vectorized
    s_prev = samples[:, :-1]  # (n, seq_len-1)
    s_next = samples[:, 1:]   # (n, seq_len-1)
    prev_mean = s_prev.mean(axis=1, keepdims=True)
    next_mean = s_next.mean(axis=1, keepdims=True)
    prev_std = s_prev.std(axis=1) + 1e-12
    next_std = s_next.std(axis=1) + 1e-12
    acf_val = np.mean((s_prev - prev_mean) * (s_next - next_mean), axis=1) / (prev_std * next_std)

    # Energy (sum of squares)
    energy = np.sum(samples ** 2, axis=1)

    # Spectral features — batched FFT
    fft_mag = np.abs(np.fft.fft(samples, axis=1))  # (n, seq_len)
    freq_idx = np.arange(seq_len, dtype=samples.dtype)
    fft_sum = fft_mag.sum(axis=1, keepdims=True) + 1e-12
    fft_norm = fft_mag / fft_sum  # normalized weights
    spectral_centroid = (fft_norm * freq_idx[None, :]).sum(axis=1)
    spectral_spread = np.sqrt((fft_norm * (freq_idx[None, :] - spectral_centroid[:, None]) ** 2).sum(axis=1))

    # Stack all features: (n_samples, 11)
    features = np.column_stack([
        mean, std, min_val, max_val, skewness, kurt,
        trend, acf_val, energy, spectral_centroid, spectral_spread,
    ])

    return features


def evaluate_discriminators(real_samples, fake_samples, n_folds=5):
    """
    Train multiple binary classifiers (RF, SVM, MLP) to distinguish real from synthetic.
    
    Args:
        real_samples: np.ndarray
        fake_samples: np.ndarray
        n_folds: int
        
    Returns:
        dict: {
            'Random Forest': {metrics},
            'SVM (RBF)': {metrics},
            'MLP': {metrics}
        }
    """
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    # Extract feature
    real_features = _extract_time_series_features(real_samples)
    fake_features = _extract_time_series_features(fake_samples)
    
    X = np.vstack([real_features, fake_features])
    y = np.hstack([np.zeros(len(real_features)), np.ones(len(fake_features))])
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', random_state=42)),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        try:
            # Cross-validation
            y_pred = cross_val_predict(clf, X, y, cv=min(n_folds, len(X) // 2))
            
            results[name] = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred, zero_division=0)),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0))
            }
        except Exception as e:
            print(f"Classifier {name} failed: {e}")
            results[name] = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
            
    return results


def discrimination_score(real_samples, fake_samples, n_folds=5):
    """Legacy wrapper for backward compatibility."""
    res = evaluate_discriminators(real_samples, fake_samples, n_folds)
    return res.get('Random Forest', {})


# ============================================================================
# PART 2: SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_sequences(generator, X_real, n_synthetic=None, device='cpu', batch_size=256):
    """
    Generate synthetic forecast sequences using trained generator.

    Args:
        generator: Trained Generator model (torch.nn.Module)
        X_real: Real input windows (n_samples, L, n_features) as numpy or torch tensor
        n_synthetic: Number of synthetic samples to generate (default: same as X_real)
        device: torch device ('cpu' or 'cuda')
        batch_size: Batch size for generation

    Returns:
        np.ndarray: Generated forecasts (n_synthetic, H, 1) in numpy format
    """
    generator.to(device)
    generator.eval()

    # Convert to tensor if needed
    if isinstance(X_real, np.ndarray):
        X_real_tensor = torch.from_numpy(X_real).float().to(device)
    else:
        X_real_tensor = X_real.float().to(device)

    if n_synthetic is None:
        n_synthetic = len(X_real)

    # Determine horizon length from generator
    L = X_real_tensor.shape[1]
    n_features = X_real_tensor.shape[2] if X_real_tensor.ndim == 3 else 1

    # Generate in batches
    Y_synthetic_list = []

    with torch.no_grad():
        for i in range(0, n_synthetic, batch_size):
            batch_size_curr = min(batch_size, n_synthetic - i)

            # Sample random noise
            noise = torch.randn(batch_size_curr, L, device=device)

            # Get corresponding X samples (cycle through real data if n_synthetic > len(X_real))
            idx = i % len(X_real)
            idx_end = min(idx + batch_size_curr, len(X_real))
            X_batch = X_real_tensor[idx:idx_end]

            # Pad if batch is smaller than batch_size_curr
            if len(X_batch) < batch_size_curr:
                pad_size = batch_size_curr - len(X_batch)
                X_batch = torch.cat([X_batch, X_real_tensor[:pad_size]], dim=0)

            # Generate
            try:
                Y_batch = generator(X_batch, noise)
            except TypeError:
                # If generator doesn't take noise, try without it
                Y_batch = generator(X_batch)

            Y_synthetic_list.append(Y_batch.cpu().numpy())

    Y_synthetic = np.vstack(Y_synthetic_list)[:n_synthetic]
    return Y_synthetic


# ============================================================================
# PART 3: TABLE GENERATION
# ============================================================================

def create_classification_metrics_table(discrimination_results, out_path):
    """
    Create table with classification metrics for real vs synthetic detection.

    Args:
        discrimination_results: dict with model names as keys and discrimination_score dicts as values
        out_path: str, path to save CSV file

    Returns:
        pd.DataFrame
    """
    rows = []
    for model, scores in discrimination_results.items():
        rows.append({
            'Model': model,
            'Accuracy': f"{scores['accuracy']:.6f}",
            'F1': f"{scores['f1']:.6f}",
            'Precision': f"{scores['precision']:.6f}",
            'Recall': f"{scores['recall']:.6f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def create_synthetic_quality_table(synthetic_quality_results, out_path):
    """
    Create table with synthetic data quality metrics.

    Args:
        synthetic_quality_results: dict with GAN type as key and metrics dict as value
        out_path: str, path to save CSV file

    Returns:
        pd.DataFrame
    """
    rows = []
    for gan_type, metrics in synthetic_quality_results.items():
        rows.append({
            'GAN_Type': gan_type,
            'Frechet_Distance': f"{metrics['frechet_distance']:.6f}",
            'Variance_Diff_Abs': f"{metrics['variance_difference']['abs_diff']:.6f}",
            'Variance_Diff_Rel_%': f"{metrics['variance_difference']['rel_diff']:.2f}",
            'Discrimination_Accuracy': f"{metrics['discrimination_score']['accuracy']:.6f}",
            'Discrimination_F1': f"{metrics['discrimination_score']['f1']:.6f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def create_data_augmentation_table(augmentation_results, out_path):
    """
    Create table showing effectiveness of data augmentation.

    Args:
        augmentation_results: dict with model names as keys and dicts containing
                             'real_only' and 'real_plus_synthetic' metrics
        out_path: str, path to save CSV file

    Returns:
        pd.DataFrame
    """
    rows = []
    for model, results in augmentation_results.items():
        real_only_rmse = results['real_only']['rmse']
        real_plus_rmse = results['real_plus_synthetic']['rmse']
        improvement = (real_only_rmse - real_plus_rmse) / (real_only_rmse + 1e-8) * 100

        rows.append({
            'Model': model,
            'Real_Only_RMSE': f"{real_only_rmse:.6f}",
            'Real_Plus_Synthetic_RMSE': f"{real_plus_rmse:.6f}",
            'Improvement_%': f"{improvement:.2f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


# ============================================================================
# PART 4: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_real_vs_synthetic_sequences(real_seqs, synthetic_seqs, out_path, n_samples=5):
    """
    Create line chart overlay showing real vs synthetic sequences.

    Args:
        real_seqs: np.ndarray of shape (n_samples, sequence_length, features) or (n_samples, sequence_length)
        synthetic_seqs: np.ndarray of same shape
        out_path: str, path to save visualizations (without extension)
        n_samples: int, number of sequences to plot

    Returns:
        dict: {'static': path_to_png, 'interactive': path_to_html}
    """
    out_path = _ensure_path(out_path)

    # Ensure 2D
    if real_seqs.ndim == 3:
        real_seqs = real_seqs.squeeze()
    if synthetic_seqs.ndim == 3:
        synthetic_seqs = synthetic_seqs.squeeze()

    n_samples = min(n_samples, len(real_seqs), len(synthetic_seqs))

    # Static matplotlib plot
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(real_seqs[i], label='Real', color=_PALETTE[0], linewidth=2, alpha=0.8)
        ax.plot(synthetic_seqs[i], label='Synthetic', color=_PALETTE[1], linewidth=2, alpha=0.8)
        ax.set_title(f'Sequence {i+1}')
        ax.legend(loc='best')
        _style_axes(ax)

    static_path = _finalise_static(fig, out_path)

    # Interactive plotly plot
    interactive_data = []
    for i in range(n_samples):
        real_seq = real_seqs[i]
        synthetic_seq = synthetic_seqs[i]
        for j, (r, s) in enumerate(zip(real_seq, synthetic_seq)):
            interactive_data.append({
                'Timestep': j,
                'Value': float(r),
                'Type': f'Real (Seq {i+1})'
            })
            interactive_data.append({
                'Timestep': j,
                'Value': float(s),
                'Type': f'Synthetic (Seq {i+1})'
            })

    if _HAS_PLOTLY and interactive_data:
        df_interactive = pd.DataFrame(interactive_data)
        fig_i = px.line(df_interactive, x='Timestep', y='Value', color='Type',
                       title='Real vs Synthetic Sequences',
                       template='plotly_white')
        fig_i.update_layout(height=200 * n_samples)
        interactive_path = _write_interactive(fig_i, out_path)
    else:
        interactive_path = None

    return {
        'static': static_path,
        'interactive': interactive_path
    }


def plot_real_vs_synthetic_distributions(real_seqs, synthetic_seqs, out_path, bins=30):
    """
    Create histogram comparing value distributions.

    Args:
        real_seqs: np.ndarray of shape (n_samples, sequence_length, features) or (n_samples, sequence_length)
        synthetic_seqs: np.ndarray of same shape
        out_path: str, path to save visualizations
        bins: int, number of histogram bins

    Returns:
        dict: {'static': path_to_png, 'interactive': path_to_html}
    """
    out_path = _ensure_path(out_path)
    real_flat = real_seqs.flatten()
    synthetic_flat = synthetic_seqs.flatten()

    # Static matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(real_flat, bins=bins, alpha=0.6, label='Real', color=_PALETTE[0], density=True)
    ax.hist(synthetic_flat, bins=bins, alpha=0.6, label='Synthetic', color=_PALETTE[1], density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison: Real vs Synthetic')
    ax.legend()
    _style_axes(ax)

    static_path = _finalise_static(fig, out_path)

    # Interactive plotly plot
    if _HAS_PLOTLY:
        fig_i = go.Figure()
        fig_i.add_trace(go.Histogram(x=real_flat, name='Real', opacity=0.7,
                                     marker=dict(color=_PALETTE[0]), nbinsx=bins))
        fig_i.add_trace(go.Histogram(x=synthetic_flat, name='Synthetic', opacity=0.7,
                                     marker=dict(color=_PALETTE[1]), nbinsx=bins))
        fig_i.update_layout(barmode='overlay', title='Distribution Comparison: Real vs Synthetic',
                           xaxis_title='Value', yaxis_title='Count', template='plotly_white')
        interactive_path = _write_interactive(fig_i, out_path)
    else:
        interactive_path = None

    return {
        'static': static_path,
        'interactive': interactive_path
    }


def plot_data_augmentation_comparison(augmentation_results, out_path):
    """
    Create bar chart comparing model performance with and without data augmentation.

    Args:
        augmentation_results: dict with model names as keys and dicts containing
                             'real_only' and 'real_plus_synthetic' metrics
        out_path: str, path to save visualizations

    Returns:
        dict: {'static': path_to_png, 'interactive': path_to_html}
    """
    out_path = _ensure_path(out_path)
    models = []
    real_only_rmse = []
    real_plus_rmse = []

    for model, results in augmentation_results.items():
        models.append(model)
        real_only_rmse.append(results['real_only']['rmse'])
        real_plus_rmse.append(results['real_plus_synthetic']['rmse'])

    # Static matplotlib plot
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, real_only_rmse, width, label='Real Only', color=_PALETTE[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, real_plus_rmse, width, label='Real + Synthetic', color=_PALETTE[1], alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Data Augmentation Effectiveness')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    _style_axes(ax)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    static_path = _finalise_static(fig, out_path)

    # Interactive plotly plot
    if _HAS_PLOTLY:
        df_data = pd.DataFrame({
            'Model': models + models,
            'RMSE': real_only_rmse + real_plus_rmse,
            'Type': ['Real Only'] * len(models) + ['Real + Synthetic'] * len(models)
        })

        fig_i = px.bar(df_data, x='Model', y='RMSE', color='Type',
                      title='Data Augmentation Effectiveness',
                      barmode='group', template='plotly_white')
        interactive_path = _write_interactive(fig_i, out_path)
    else:
        interactive_path = None

    return {
        'static': static_path,
        'interactive': interactive_path
    }
