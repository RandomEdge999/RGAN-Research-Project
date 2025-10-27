import matplotlib.pyplot as plt


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
