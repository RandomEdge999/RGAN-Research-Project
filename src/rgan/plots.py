import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_single_train_test(curve_epoch, train_rmse, test_rmse, title, out_path, ylabel="RMSE"):
    plt.figure(figsize=(8,4.5))
    plt.plot(curve_epoch, train_rmse, label="Train RMSE")
    plt.plot(curve_epoch, test_rmse, label="Test RMSE")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_constant_train_test(train_value, test_value, title, out_path, ylabel="RMSE"):
    steps = [0, 1]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, [train_value, train_value], label=f"Train {ylabel}")
    plt.plot(steps, [test_value, test_value], label=f"Test {ylabel}")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.xticks(steps, ["Start", "End"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_compare_models_bars(train_errors: dict, test_errors: dict, out_test_path: str, out_train_path: str):
    models = list(test_errors.keys())
    plt.figure(figsize=(7,4))
    plt.bar(models, [test_errors[m] for m in models])
    plt.title("Test Error by Model")
    plt.xlabel("Models")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_test_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7,4))
    plt.bar(models, [train_errors[m] for m in models])
    plt.title("Train Error by Model")
    plt.xlabel("Models")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_train_path, dpi=150)
    plt.close()

def plot_classical_curves(sizes, ets_curve, arima_curve, out_path):
    if sizes is None:
        return None
    plt.figure(figsize=(8,4.5))
    plt.plot(sizes, ets_curve, marker="o", label="ETS (RMSE)")
    plt.plot(sizes, arima_curve, marker="o", label="ARIMA (RMSE)")
    plt.title("Classical Models: Error vs Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_learning_curves(sizes, curves: dict, out_path, ylabel="RMSE"):
    if sizes is None or len(sizes) == 0:
        return None
    plt.figure(figsize=(8, 4.5))
    for label, values in curves.items():
        plt.plot(sizes, values, marker="o", label=label)
    plt.title("Model RMSE vs Training Sample Size")
    plt.xlabel("Number of Training Windows")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def create_error_metrics_table(model_results: dict, out_path: str = None):
    data = []
    for model_name, results in model_results.items():
        if 'train' in results and 'test' in results:
            data.append({
                'Model': f"{model_name} (Train)",
                'RMSE': f"{results['train']['rmse']:.6f}",
                'MSE': f"{results['train']['mse']:.6f}",
                'BIAS': f"{results['train']['bias']:.6f}",
                'MAE': f"{results['train']['mae']:.6f}"
            })
            data.append({
                'Model': f"{model_name} (Test)",
                'RMSE': f"{results['test']['rmse']:.6f}",
                'MSE': f"{results['test']['mse']:.6f}",
                'BIAS': f"{results['test']['bias']:.6f}",
                'MAE': f"{results['test']['mae']:.6f}"
            })
    df = pd.DataFrame(data)
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Error metrics table saved to: {out_path}")
    return df

def plot_naive_bayes_comparison(naive_baseline_stats, naive_bayes_stats, out_path):
    models = ['Naive Baseline', 'Naive Bayes']
    rmse_values = [naive_baseline_stats['rmse'], naive_bayes_stats['rmse']]
    mse_values = [naive_baseline_stats['mse'], naive_bayes_stats['mse']]
    bias_values = [naive_baseline_stats['bias'], naive_bayes_stats['bias']]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.bar(models, rmse_values, color=['skyblue', 'lightcoral'])
    ax1.set_title('RMSE Comparison')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(models, mse_values, color=['skyblue', 'lightcoral'])
    ax2.set_title('MSE Comparison')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=45)

    ax3.bar(models, bias_values, color=['skyblue', 'lightcoral'])
    ax3.set_title('BIAS Comparison')
    ax3.set_ylabel('BIAS')
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


