import copy
import dataclasses
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch, log_info, log_step, log_warn, log_error, log_exception, log_debug, log_shape
from .metrics import error_stats


class _IdentityScaler:
    """No-op GradScaler replacement when AMP is unavailable."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def scale(self, outputs: Any) -> Any:
        return outputs

    def unscale_(self, optimizer: Any) -> Any:
        return optimizer

    def step(self, optimizer: Any) -> None:
        optimizer.step()

    def update(self) -> None:
        pass


class _AmpAdapters:
    """Resolve the best available AMP APIs with graceful fallback handling."""

    def __init__(self) -> None:
        self._autocast_impl = None
        self._grad_scaler_cls = None
        self._resolve()

    def _resolve(self) -> None:
        torch_amp = getattr(torch, "amp", None)
        cuda_amp = getattr(torch.cuda, "amp", None)
        self._autocast_impl = getattr(torch_amp, "autocast", None) or getattr(
            cuda_amp, "autocast", None
        )
        self._grad_scaler_cls = getattr(torch_amp, "GradScaler", None) or getattr(
            cuda_amp, "GradScaler", None
        )

    @property
    def available(self) -> bool:
        return self._autocast_impl is not None and self._grad_scaler_cls is not None

    def autocast(self, device_type: str, enabled: bool = True) -> Any:
        if not enabled or self._autocast_impl is None:
            return nullcontext()
        try:
            return self._autocast_impl(device_type=device_type, enabled=True)
        except TypeError:  # Older torch.cuda.amp.autocast lacks device_type
            return self._autocast_impl(enabled=True)

    def make_scaler(self, enabled: bool = True) -> Any:
        if self._grad_scaler_cls is None:
            return _IdentityScaler(enabled=False)
        return self._grad_scaler_cls(enabled=enabled)


AMP = _AmpAdapters()
_DEFAULT_PRELOAD_LIMIT_BYTES = 1024 ** 3


def _select_eval_generator(
    generator: nn.Module,
    generator_ema: Optional[nn.Module] = None,
) -> nn.Module:
    """Return the generator checkpoint that should be used for evaluation."""
    return generator_ema if generator_ema is not None else generator


def _tensor_nbytes(arr: Union[np.ndarray, torch.Tensor]) -> int:
    """Return the byte size of a numpy array or torch tensor."""
    if torch.is_tensor(arr):
        return int(arr.element_size() * arr.numel())
    return int(arr.nbytes)


def _estimate_split_bytes(data_splits: Dict[str, Union[np.ndarray, torch.Tensor]]) -> int:
    """Estimate the total footprint of the provided data split tensors."""
    total = 0
    for key in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"):
        if key in data_splits:
            total += _tensor_nbytes(data_splits[key])
    return total


def _should_preload_to_device(
    preload_to_device: str,
    device: torch.device,
    total_bytes: int,
    limit_bytes: int = _DEFAULT_PRELOAD_LIMIT_BYTES,
) -> bool:
    """Decide whether dataset tensors should be moved to the target device once."""
    if device.type != "cuda":
        return False
    if preload_to_device == "always":
        return True
    if preload_to_device == "auto":
        return total_bytes <= limit_bytes
    return False


def _should_apply_critic_regularizer(critic_step: int, interval: int) -> bool:
    """Return True when critic regularization should fire on this step."""
    if interval <= 1:
        return True
    return critic_step % interval == 0


def _prepare_batch_tensor(
    batch: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Move an array/tensor batch to the target device if needed."""
    if torch.is_tensor(batch):
        if batch.device == device:
            return batch
        return batch.to(device=device, non_blocking=True)
    return torch.from_numpy(batch).to(device=device, non_blocking=True)


def _preload_splits_to_device(
    data_splits: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Preload all data splits to the target device."""
    preloaded: Dict[str, torch.Tensor] = {}
    for key, value in data_splits.items():
        preloaded[key] = torch.from_numpy(value).to(device=device)
    return preloaded


def compute_metrics(
    G: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 512,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run the generator in evaluation mode and compute error statistics.

    Args:
        G: The Generator module.
        X: Input features (numpy array).
        Y: Target values (numpy array).
        batch_size: Batch size for inference.

    Returns:
        A tuple containing:
        - A dictionary of error metrics (RMSE, MAE, etc.).
        - The predicted values as a numpy array.
    """
    if torch.is_tensor(Y):
        Y_ref = Y.detach().cpu().numpy()
    else:
        Y_ref = Y
    Yp = predict_generator_forecasts(G, X, batch_size=batch_size, stochastic=False)
    stats = error_stats(Y_ref.reshape(-1), Yp.reshape(-1))
    return stats, Yp


def predict_generator_forecasts(
    G: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    batch_size: int = 512,
    stochastic: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Run generator inference without needing reference targets."""
    G.eval()
    device = next(G.parameters()).device
    n_samples = len(X)
    bs = max(1, int(batch_size))
    noise_dim = int(getattr(G, "noise_dim", 0) or 0)
    rng = np.random.default_rng(seed) if seed is not None else None
    preds: List[np.ndarray] = []

    while True:
        try:
            preds.clear()
            idx = 0
            with torch.inference_mode():
                while idx < n_samples:
                    end = min(idx + bs, n_samples)
                    Xb = _prepare_batch_tensor(X[idx:end], device)
                    z = None
                    if stochastic and noise_dim > 0:
                        z_np = (rng.standard_normal((Xb.size(0), Xb.size(1), noise_dim)).astype(np.float32)
                                if rng is not None else np.random.randn(Xb.size(0), Xb.size(1), noise_dim).astype(np.float32))
                        z = torch.from_numpy(z_np).to(device=device, non_blocking=True)
                    with AMP.autocast(
                        device.type, enabled=(device.type == "cuda" and AMP.available)
                    ):
                        Yb = G(Xb, z) if z is not None else G(Xb)
                    preds.append(Yb.detach().cpu().float().numpy())
                    idx = end
                if device.type == "cuda":
                    torch.cuda.synchronize()
            return np.concatenate(preds, axis=0)
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)


def _gradient_penalty(
    D: nn.Module,
    real_inputs: torch.Tensor,
    fake_inputs: torch.Tensor,
    device: torch.device,
    gp_lambda: float,
) -> torch.Tensor:
    """Computes the gradient penalty for WGAN-GP.

    Args:
        D: The Discriminator module.
        real_inputs: Real data samples.
        fake_inputs: Generated data samples.
        device: The computation device.
        gp_lambda: The gradient penalty coefficient.

    Returns:
        The computed gradient penalty scalar tensor.
    """
    # Ensure fp32 precision for gradient penalty — inputs may carry fp16 from
    # AMP autocast, and the norm/subtraction (grad.norm - 1.0) loses precision in fp16.
    real_inputs = real_inputs.float()
    fake_inputs = fake_inputs.float()
    batch_size = real_inputs.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_inputs)
    interpolated = epsilon * real_inputs + (1 - epsilon) * fake_inputs
    interpolated.requires_grad_(True)

    # Disable CuDNN for this forward pass to allow double backward (required for gradient penalty)
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolated = D(interpolated)
    
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp_lambda * penalty


def _apply_instance_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Applies Gaussian instance noise to the input tensor."""
    if std <= 0:
        return tensor
    noise = torch.randn_like(tensor) * std
    return tensor + noise


def _init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Initializes the Exponential Moving Average shadow parameters."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _update_ema(
    shadow: Dict[str, torch.Tensor], model: nn.Module, decay: float
) -> None:
    """Updates the EMA shadow parameters."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def _swap_params_with_shadow(
    model: nn.Module, shadow: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Swaps the model parameters with the shadow parameters (for evaluation)."""
    backup = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in shadow:
                backup[name] = param.detach().clone()
                param.copy_(shadow[name])
    return backup


def _save_checkpoint(
    path: Path,
    epoch: int,
    G: nn.Module,
    D: nn.Module,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    scaler_G: Any,
    scaler_D: Any,
    best_val: float,
    best_state: Optional[Dict[str, torch.Tensor]],
    bad_epochs: int,
    history: Dict[str, List[float]],
    ema_shadow: Optional[Dict[str, torch.Tensor]],
    current_batch_size: int,
    config: TrainConfig,
) -> None:
    """Save a complete training checkpoint to disk."""
    ckpt = {
        "epoch": epoch,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "optG_state_dict": optG.state_dict(),
        "optD_state_dict": optD.state_dict(),
        "best_val": best_val,
        "best_state": best_state,
        "bad_epochs": bad_epochs,
        "history": history,
        "ema_shadow": ema_shadow,
        "current_batch_size": current_batch_size,
        "config_dict": dataclasses.asdict(config),
    }
    # Save scaler state only if they have real state (not _IdentityScaler)
    if hasattr(scaler_G, "state_dict") and not isinstance(scaler_G, _IdentityScaler):
        ckpt["scaler_G_state_dict"] = scaler_G.state_dict()
    if hasattr(scaler_D, "state_dict") and not isinstance(scaler_D, _IdentityScaler):
        ckpt["scaler_D_state_dict"] = scaler_D.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename for atomicity
    tmp_path = path.with_suffix(".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.rename(path)


def _load_checkpoint(
    path: Path,
    G: nn.Module,
    D: nn.Module,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    scaler_G: Any,
    scaler_D: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a training checkpoint and restore all state.

    Returns a dict with: epoch, best_val, best_state, bad_epochs, history,
    ema_shadow, current_batch_size.
    """
    console = get_console()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G_state_dict"])
    D.load_state_dict(ckpt["D_state_dict"])
    optG.load_state_dict(ckpt["optG_state_dict"])
    optD.load_state_dict(ckpt["optD_state_dict"])

    if "scaler_G_state_dict" in ckpt and not isinstance(scaler_G, _IdentityScaler):
        scaler_G.load_state_dict(ckpt["scaler_G_state_dict"])
    if "scaler_D_state_dict" in ckpt and not isinstance(scaler_D, _IdentityScaler):
        scaler_D.load_state_dict(ckpt["scaler_D_state_dict"])

    console.log(
        f"Resumed from checkpoint: epoch {ckpt['epoch']}, "
        f"best_val={ckpt['best_val']:.6f}"
    )
    return {
        "epoch": ckpt["epoch"],
        "best_val": ckpt["best_val"],
        "best_state": ckpt["best_state"],
        "bad_epochs": ckpt["bad_epochs"],
        "history": ckpt["history"],
        "ema_shadow": ckpt.get("ema_shadow"),
        "current_batch_size": ckpt.get("current_batch_size", 64),
        "config_dict": ckpt.get("config_dict", {}),
    }


def train_rgan_torch(
    config: TrainConfig,
    models: Tuple[nn.Module, nn.Module],
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "rgan",
) -> Dict[str, Any]:
    """Train the R-GAN with stability improvements and detailed telemetry.

    Args:
        config: The training configuration object.
        models: A tuple (Generator, Discriminator).
        data_splits: Dictionary containing train/val/test splits (X and Y).
        results_dir: Directory to save results (unused in this function but kept for interface).
        tag: Experiment tag.

    Returns:
        A dictionary containing the trained models, history, and evaluation statistics.
    """
    G, D = models
    console = get_console()
    log_info(f"train_rgan_torch called: epochs={config.epochs}, batch_size={config.batch_size}, device={config.device}")
    log_step(f"GAN variant: {config.gan_variant}, lr_g={config.lr_g}, lr_d={config.lr_d}")
    log_step(f"lambda_reg={config.lambda_reg}, adv_weight={config.adv_weight}, patience={config.patience}")
    log_step(f"AMP requested: {config.amp}, eval_every={config.eval_every}")
    log_step(f"Generator params: {sum(p.numel() for p in G.parameters())}")
    log_step(f"Discriminator params: {sum(p.numel() for p in D.parameters())}")

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        hint = (
            "Torch reports no accessible CUDA device. Ensure you installed a CUDA-enabled PyTorch build "
            "(e.g., pip install 'torch==2.2.2+cu121') and that NVIDIA drivers are available."
        )
        if config.strict_device:
            raise RuntimeError(
                f"CUDA device {config.device} was requested but is unavailable. {hint}"
            )
        console.log(
            f"WARNING: CUDA unavailable — falling back to CPU despite requested device {config.device}. {hint}"
        )
        requested_device = torch.device("cpu")

    if requested_device.type == "cuda":
        torch.cuda.set_device(requested_device)
        device_name = torch.cuda.get_device_name(requested_device)
        console.log(f"Using CUDA device: {device_name} ({requested_device})")
    else:
        console.log("WARNING: Using CPU for training (no CUDA device active).")

    device = requested_device
    G.to(device)
    D.to(device)

    compile_requested = config.compile_mode == "reduce-overhead"
    G_train = G
    D_train = D
    compile_enabled = False
    if compile_requested:
        if config.gan_variant.lower() == "wgan-gp":
            log_warn("torch.compile disabled for WGAN-GP: gradient penalty requires double backward, "
                     "which aot_autograd does not support.")
        else:
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:
                log_warn("torch.compile requested but unavailable in this PyTorch build; falling back to eager mode.")
            else:
                try:
                    G_train = compile_fn(G, mode="reduce-overhead")
                    D_train = compile_fn(D, mode="reduce-overhead")
                    compile_enabled = True
                    log_step("torch.compile enabled for RGAN training modules (mode=reduce-overhead).")
                except Exception as exc:
                    G_train = G
                    D_train = D
                    log_warn(f"torch.compile setup failed; continuing in eager mode. Reason: {exc}")

    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]
    Xte, Yte = data_splits["Xte"], data_splits["Yte"]
    log_shape("Xtr", Xtr)
    log_shape("Ytr", Ytr)
    log_shape("Xval", Xval)
    log_shape("Yval", Yval)
    log_shape("Xte", Xte)
    log_shape("Yte", Yte)

    split_bytes = _estimate_split_bytes(data_splits)
    preload_mode = config.preload_to_device
    preload_selected = _should_preload_to_device(preload_mode, device, split_bytes)
    preloaded_splits: Optional[Dict[str, torch.Tensor]] = None
    if preload_selected:
        try:
            preloaded_splits = _preload_splits_to_device(data_splits, device)
            log_step(
                f"Preloaded data splits to {device} ({split_bytes / (1024 ** 2):.1f} MB, mode={preload_mode})."
            )
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if preload_mode == "always":
                raise
            preloaded_splits = None
            log_warn(
                "Automatic GPU preloading ran out of memory; falling back to DataLoader-based eager batching."
            )
    else:
        reason = "non-CUDA device" if device.type != "cuda" else f"dataset exceeds {(_DEFAULT_PRELOAD_LIMIT_BYTES / (1024 ** 3)):.0f} GB auto threshold"
        if preload_mode != "never":
            log_step(f"Skipping dataset preloading ({reason}, mode={preload_mode}).")

    log_step(f"Creating Adam optimizers: lr_g={config.lr_g}, lr_d={config.lr_d}, weight_decay={config.weight_decay}")
    optG = torch.optim.Adam(G.parameters(), lr=config.lr_g, weight_decay=config.weight_decay)
    optD = torch.optim.Adam(D.parameters(), lr=config.lr_d, weight_decay=config.weight_decay)
    mse = nn.MSELoss()

    use_logits = config.use_logits

    gan_variant = config.gan_variant.lower()
    if gan_variant not in {"standard", "wgan", "wgan-gp"}:
        raise ValueError(f"Unsupported GAN variant '{config.gan_variant}'.")
    if gan_variant in {"wgan", "wgan-gp"}:
        use_logits = True  # Wasserstein training expects raw scores

    bce_loss = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    d_steps = int(config.d_steps)
    if d_steps < 1:
        raise ValueError("d_steps must be >= 1.")
    g_steps = max(1, config.g_steps)
    eval_batch_size = max(1, int(config.eval_batch_size))
    config.d_steps = d_steps
    config.g_steps = g_steps
    config.eval_batch_size = eval_batch_size
    warmup_epochs = max(0, config.supervised_warmup_epochs)
    patience = config.patience
    
    # Regularization scheduling
    lambda_reg_start = config.lambda_reg_start if config.lambda_reg_start is not None else config.lambda_reg
    lambda_reg_end = config.lambda_reg_end if config.lambda_reg_end is not None else config.lambda_reg
    lambda_reg_warmup = max(1, config.lambda_reg_warmup_epochs)

    adv_weight = config.adv_weight
    instance_noise_std = config.instance_noise_std
    instance_noise_decay = config.instance_noise_decay
    gp_lambda = config.wgan_gp_lambda
    critic_reg_interval = max(1, int(config.critic_reg_interval))
    ema_decay = config.ema_decay
    track_logits = config.track_discriminator_outputs

    disc_activation = config.d_activation.lower() if config.d_activation else ""
    if gan_variant in {"wgan", "wgan-gp"}:
        disc_activation = ""  # Force linear outputs for Wasserstein critics
    elif not disc_activation and not use_logits:
        disc_activation = "sigmoid"

    def _disc_output_to_prob(tensor: torch.Tensor) -> torch.Tensor:
        if use_logits:
            return tensor
        if disc_activation == "tanh":
            prob = 0.5 * (tensor + 1.0)
        elif disc_activation in {"sigmoid"}:
            prob = tensor
        else:
            prob = torch.sigmoid(tensor)
        
        if prob.is_floating_point():
            eps = torch.finfo(prob.dtype).eps
        else:
            eps = 1e-6
        return torch.clamp(prob, eps, 1.0 - eps)

    def _safe_bce(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if use_logits:
            # Inputs already represent logits; allow AMP to manage dtype.
            return bce_loss(inputs, targets if targets.dtype == inputs.dtype else targets.to(inputs.dtype))

        # Avoid unnecessary casting if already float32
        inputs_fp32 = inputs if inputs.dtype == torch.float32 else inputs.float()
        targets_fp32 = targets if targets.dtype == torch.float32 else targets.float()
        
        if device.type == "cuda":
            with AMP.autocast(device.type, enabled=False):
                return bce_loss(inputs_fp32, targets_fp32)
        return bce_loss(inputs_fp32, targets_fp32)

    current_batch_size = max(1, config.batch_size)
    amp_requested = config.amp
    amp_available = AMP.available
    amp_enabled = amp_requested and (device.type == "cuda") and amp_available
    log_step(f"AMP: requested={amp_requested}, available={amp_available}, enabled={amp_enabled}")
    
    if amp_requested and not amp_enabled:
        if device.type != "cuda":
            get_console().print(
                "WARNING: AMP requested but no CUDA device is active; running in full precision."
            )
        elif not amp_available:
            get_console().print(
                "WARNING: AMP requested but unsupported by this PyTorch build; running in full precision."
            )

    scaler_G = AMP.make_scaler(enabled=amp_enabled)
    # WGAN-GP gradient penalty uses create_graph=True for second-order gradients.
    # GradScaler inflates the loss, and that inflation leaks into the GP's
    # second-order computation, producing incorrect gradients.  Disable scaling
    # for D when using gradient penalty; G can still benefit from AMP scaling.
    scaler_D = AMP.make_scaler(enabled=(amp_enabled and gan_variant != "wgan-gp"))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    start_epoch = 1

    use_ema = 0.0 < ema_decay < 1.0
    ema_shadow: Optional[Dict[str, torch.Tensor]] = _init_ema(G) if use_ema else None

    noise_dim = getattr(G, "noise_dim", 0)
    lambda_diversity = getattr(config, "lambda_diversity", 0.0)
    if noise_dim > 0:
        log_step(f"Noise dim: {noise_dim}, lambda_diversity={lambda_diversity}")

    hist: Dict[str, List[float]] = {
        "epoch": [],
        "train_rmse": [],
        "test_rmse": [],
        "val_rmse": [],
        "D_loss": [],
        "G_loss": [],
        "G_adv": [],
        "G_reg": [],
        "G_div": [],
        "grad_norm_G": [],
        "grad_norm_D": [],
        "batch_size": [],
    }
    if track_logits:
        hist["D_real_mean"] = []
        hist["D_fake_mean"] = []

    log_step(f"Training state: batch_size={current_batch_size}, warmup_epochs={warmup_epochs}, d_steps={d_steps}, g_steps={g_steps}")
    log_step(f"Instance noise: std={instance_noise_std}, decay={instance_noise_decay}")
    log_step(f"EMA: decay={ema_decay}, enabled={use_ema}")
    log_step(f"Lambda reg: start={lambda_reg_start}, end={lambda_reg_end}, warmup={lambda_reg_warmup}")
    log_step(
        f"Execution path: preload_to_device={preload_mode}, preloaded={preloaded_splits is not None}, "
        f"compile_mode={config.compile_mode}, compile_enabled={compile_enabled}, critic_reg_interval={critic_reg_interval}"
    )
    lambda_reg = lambda_reg_end

    # --- Checkpoint resume ---
    resumed_from_epoch = 0
    if config.resume_from and os.path.isfile(config.resume_from):
        log_step(f"Loading checkpoint from: {config.resume_from}")
        restored = _load_checkpoint(
            Path(config.resume_from), G, D, optG, optD, scaler_G, scaler_D, device
        )
        resumed_from_epoch = int(restored["epoch"])
        if resumed_from_epoch > config.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {resumed_from_epoch}, "
                f"but --epochs={config.epochs}. Increase --epochs to continue."
            )
        start_epoch = restored["epoch"] + 1
        best_val = restored["best_val"]
        best_state = restored["best_state"]
        bad_epochs = restored["bad_epochs"]
        hist = restored["history"]
        if restored["ema_shadow"] is not None:
            ema_shadow = restored["ema_shadow"]
        current_batch_size = restored["current_batch_size"]
        remaining_epochs = max(0, config.epochs - resumed_from_epoch)
        log_step(
            f"Resume target semantics: checkpoint_epoch={resumed_from_epoch}, "
            f"target_epochs={config.epochs}, remaining_epochs={remaining_epochs}"
        )
        if remaining_epochs == 0:
            log_step("Checkpoint already reached target epochs; skipping RGAN epoch loop and materializing final outputs.")

    num_workers = config.num_workers
    prefetch_factor = max(1, config.prefetch_factor)
    persistent_workers = bool(config.persistent_workers and num_workers > 0)
    pin_memory = bool(config.pin_memory and device.type == "cuda")

    log_step(
        f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}"
    )
    train_ds = None
    if preloaded_splits is None:
        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
        log_step(f"TensorDataset created: {len(train_ds)} samples")
    else:
        log_step(f"Using preloaded tensor batches for {len(Xtr)} training samples.")

    def _effective_batch_size(bs: int) -> int:
        return max(1, min(int(bs), len(Xtr)))

    def _drop_last_for_compile(bs: int) -> bool:
        return bool(compile_enabled and len(Xtr) > bs and (len(Xtr) % bs) != 0)

    def make_loader(bs: int) -> DataLoader:
        if train_ds is None:
            raise RuntimeError("DataLoader requested even though training tensors were preloaded to device.")
        effective_bs = _effective_batch_size(bs)
        kwargs = dict(
            batch_size=effective_bs,
            shuffle=True,
            drop_last=_drop_last_for_compile(effective_bs),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(train_ds, **kwargs)
    
    def _iter_preloaded_batches(bs: int):
        effective_bs = _effective_batch_size(bs)
        drop_last = _drop_last_for_compile(effective_bs)
        order = torch.randperm(len(Xtr), device=device)
        if drop_last:
            usable = (len(order) // effective_bs) * effective_bs
            order = order[:usable]
        for start in range(0, len(order), effective_bs):
            idx = order[start:start + effective_bs]
            if idx.numel() == 0:
                continue
            if drop_last and idx.numel() < effective_bs:
                break
            yield (
                preloaded_splits["Xtr"].index_select(0, idx),
                preloaded_splits["Ytr"].index_select(0, idx),
            )

    current_batch_size = _effective_batch_size(current_batch_size)
    train_loader = make_loader(current_batch_size) if train_ds is not None else None

    with epoch_progress(config.epochs, description="R-GAN (Torch)") as (progress, task_id):
        if start_epoch > 1 and progress is not None:
            progress.update(task_id, completed=start_epoch - 1)
        critic_step = 0
        for epoch in range(start_epoch, config.epochs + 1):
            supervised_only = epoch <= warmup_epochs
            lambda_reg = lambda_reg_end
            if lambda_reg_warmup > 1:
                progress_ratio = min(1.0, max(0.0, (epoch - 1) / (lambda_reg_warmup - 1)))
                lambda_reg = lambda_reg_start + (lambda_reg_end - lambda_reg_start) * progress_ratio

            epoch_D_losses: List[float] = []
            epoch_G_losses: List[float] = []
            epoch_adv: List[float] = []
            epoch_reg: List[float] = []
            epoch_div: List[float] = []
            epoch_grad_G: List[float] = []
            epoch_grad_D: List[float] = []
            epoch_D_real: List[float] = []
            epoch_D_fake: List[float] = []
            
            noise_std_epoch = instance_noise_std * (instance_noise_decay ** max(0, epoch - 1))

            G.train()
            D.train()
            if G_train is not G:
                G_train.train()
            if D_train is not D:
                D_train.train()

            while True:
                try:
                    batch_iter = _iter_preloaded_batches(current_batch_size) if preloaded_splits is not None else train_loader
                    for Xb, Yb in batch_iter:
                        if preloaded_splits is None:
                            Xb = Xb.to(device, non_blocking=True)
                            Yb = Yb.to(device, non_blocking=True)
                        target_series = Xb[..., :1]

                        if not supervised_only:
                            for _ in range(d_steps):
                                critic_step += 1
                                optD.zero_grad(set_to_none=True)
                                with torch.no_grad():
                                    with AMP.autocast(device.type, enabled=amp_enabled):
                                        z_d = torch.randn(Xb.size(0), Xb.size(1), noise_dim, device=device) if noise_dim > 0 else None
                                        Y_fake_detached = G_train(Xb, z_d)
                                if Y_fake_detached.dtype != target_series.dtype:
                                    Y_fake_detached = Y_fake_detached.to(dtype=target_series.dtype)

                                with AMP.autocast(device.type, enabled=amp_enabled):
                                    real_pairs = torch.cat([target_series, Yb], dim=1)
                                    fake_pairs = torch.cat([target_series, Y_fake_detached], dim=1)
                                    if noise_std_epoch > 0:
                                        real_pairs = _apply_instance_noise(real_pairs, noise_std_epoch)
                                        fake_pairs = _apply_instance_noise(fake_pairs, noise_std_epoch)

                                    D_real = D_train(real_pairs)
                                    D_fake = D_train(fake_pairs)

                                if gan_variant in {"wgan", "wgan-gp"}:
                                    D_loss_main = -(D_real.mean() - D_fake.mean())
                                    if gan_variant == "wgan-gp":
                                        if _should_apply_critic_regularizer(critic_step, critic_reg_interval):
                                            gp = _gradient_penalty(
                                                D_train,
                                                real_pairs,
                                                fake_pairs,
                                                device,
                                                gp_lambda,
                                            )
                                            D_loss = D_loss_main + (gp * critic_reg_interval)
                                        else:
                                            D_loss = D_loss_main
                                    else:
                                        D_loss = D_loss_main
                                else:
                                    real_targets = torch.full_like(D_real, config.label_smooth)
                                    fake_targets = torch.zeros_like(D_fake)
                                    D_real_prob = _disc_output_to_prob(D_real)
                                    D_fake_prob = _disc_output_to_prob(D_fake)

                                    real_targets_eval = real_targets.to(D_real.dtype) if use_logits else real_targets
                                    fake_targets_eval = fake_targets.to(D_fake.dtype) if use_logits else fake_targets

                                    D_loss_main = 0.5 * (
                                        _safe_bce(D_real_prob, real_targets_eval)
                                        + _safe_bce(D_fake_prob, fake_targets_eval)
                                    )
                                    D_loss = D_loss_main

                                scaler_D.scale(D_loss).backward()
                                scaler_D.unscale_(optD)
                                clip_grad_value_(D.parameters(), config.grad_clip)

                                total_norm = torch.nn.utils.clip_grad_norm_(D.parameters(), float('inf'))
                                epoch_grad_D.append(total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm))

                                scaler_D.step(optD)
                                scaler_D.update()

                                if gan_variant == "wgan":
                                    clip_val = float(config.wgan_clip_value)
                                    for p in D.parameters():
                                        if p.requires_grad:
                                            p.data.clamp_(-clip_val, clip_val)

                                epoch_D_losses.append(D_loss_main.detach().item())
                                if track_logits:
                                    if gan_variant in {"wgan", "wgan-gp"}:
                                        epoch_D_real.append(D_real.mean().detach().item())
                                        epoch_D_fake.append(D_fake.mean().detach().item())
                                    else:
                                        real_prob = torch.sigmoid(D_real) if use_logits else _disc_output_to_prob(D_real)
                                        fake_prob = torch.sigmoid(D_fake) if use_logits else _disc_output_to_prob(D_fake)
                                        epoch_D_real.append(real_prob.mean().detach().item())
                                        epoch_D_fake.append(fake_prob.mean().detach().item())

                        for _ in range(g_steps):
                            optG.zero_grad(set_to_none=True)
                            with AMP.autocast(device.type, enabled=amp_enabled):
                                z_g = torch.randn(Xb.size(0), Xb.size(1), noise_dim, device=device) if noise_dim > 0 else None
                                Y_fake = G_train(Xb, z_g)
                                cat_target = target_series
                                if cat_target.dtype != Y_fake.dtype:
                                    cat_target = cat_target.to(dtype=Y_fake.dtype)
                                fake_pairs = torch.cat([cat_target, Y_fake], dim=1)
                                if noise_std_epoch > 0 and not supervised_only:
                                    fake_pairs = _apply_instance_noise(fake_pairs, noise_std_epoch)

                                if supervised_only:
                                    adv_loss = torch.tensor(0.0, device=device)
                                elif gan_variant in {"wgan", "wgan-gp"}:
                                    adv_loss = -D_train(fake_pairs).mean()
                                else:
                                    logits = D_train(fake_pairs)
                                    labels = torch.ones_like(logits)
                                    if use_logits:
                                        adv_inputs = logits
                                        labels_eval = labels
                                    else:
                                        adv_inputs = _disc_output_to_prob(logits)
                                        labels_eval = labels
                                    adv_loss = _safe_bce(adv_inputs, labels_eval)

                                target_for_reg = Yb
                                if target_for_reg.dtype != Y_fake.dtype:
                                    target_for_reg = target_for_reg.to(dtype=Y_fake.dtype)
                                reg_loss = mse(Y_fake, target_for_reg)

                                # Diversity loss (MSGAN-style): penalize same output for different z
                                div_loss = torch.tensor(0.0, device=device)
                                if noise_dim > 0 and lambda_diversity > 0 and not supervised_only:
                                    z_g2 = torch.randn_like(z_g)
                                    Y_fake2 = G_train(Xb, z_g2)
                                    z_dist = (z_g - z_g2).flatten(1).norm(dim=1, keepdim=True) + 1e-8
                                    y_dist = (Y_fake - Y_fake2).flatten(1).norm(dim=1, keepdim=True)
                                    div_loss = -(y_dist / z_dist).mean()

                                total_loss = adv_weight * adv_loss + lambda_reg * reg_loss + lambda_diversity * div_loss

                            scaler_G.scale(total_loss).backward()
                            scaler_G.unscale_(optG)
                            clip_grad_value_(G.parameters(), config.grad_clip)

                            total_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), float('inf'))
                            epoch_grad_G.append(total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm))

                            scaler_G.step(optG)
                            scaler_G.update()

                            epoch_G_losses.append(total_loss.detach().item())
                            epoch_adv.append(adv_loss.detach().item())
                            epoch_reg.append(reg_loss.detach().item())
                            epoch_div.append(div_loss.detach().item())

                            if use_ema and ema_shadow is not None:
                                _update_ema(ema_shadow, G, ema_decay)

                        del Xb, Yb, target_series

                    break
                except torch.cuda.OutOfMemoryError:
                    log_error(f"CUDA OOM at epoch {epoch}, current batch_size={current_batch_size}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    optD.zero_grad(set_to_none=True)
                    optG.zero_grad(set_to_none=True)
                    if current_batch_size == 1:
                        log_error("Batch size already 1, cannot reduce further. Raising OOM.")
                        raise
                    current_batch_size = _effective_batch_size(max(1, current_batch_size // 2))
                    log_warn(f"Reducing batch size to {current_batch_size} and retrying epoch {epoch}")
                    if train_ds is not None:
                        train_loader = make_loader(current_batch_size)
                    continue

            hist["batch_size"].append(current_batch_size)
            hist["grad_norm_G"].append(float(np.mean(epoch_grad_G) if epoch_grad_G else 0.0))
            hist["grad_norm_D"].append(float(np.mean(epoch_grad_D) if epoch_grad_D else 0.0))
            hist["D_loss"].append(float(np.mean(epoch_D_losses) if epoch_D_losses else 0.0))
            hist["G_loss"].append(float(np.mean(epoch_G_losses) if epoch_G_losses else 0.0))
            hist["G_adv"].append(float(np.mean(epoch_adv) if epoch_adv else 0.0))
            hist["G_reg"].append(float(np.mean(epoch_reg) if epoch_reg else 0.0))
            hist["G_div"].append(float(np.mean(epoch_div) if epoch_div else 0.0))
            if track_logits:
                hist["D_real_mean"].append(float(np.mean(epoch_D_real) if epoch_D_real else 0.0))
                hist["D_fake_mean"].append(float(np.mean(epoch_D_fake) if epoch_D_fake else 0.0))

            eval_bs = max(1, min(config.eval_batch_size, current_batch_size))
            eval_every = max(1, getattr(config, "eval_every", 1))
            is_eval_epoch = (epoch % eval_every == 0) or (epoch == config.epochs)

            if is_eval_epoch:
                # Only evaluate on val set during training for speed;
                # train/test metrics are computed at the end.
                eval_Xval = preloaded_splits["Xval"] if preloaded_splits is not None else Xval
                eval_Yval = preloaded_splits["Yval"] if preloaded_splits is not None else Yval
                if use_ema and ema_shadow is not None:
                    backup = _swap_params_with_shadow(G, ema_shadow)
                    va_stats, _ = compute_metrics(G, eval_Xval, eval_Yval, batch_size=eval_bs)
                    _swap_params_with_shadow(G, backup)
                else:
                    va_stats, _ = compute_metrics(G, eval_Xval, eval_Yval, batch_size=eval_bs)

                hist["epoch"].append(epoch)
                hist["val_rmse"].append(va_stats["rmse"])
                # Placeholder for train/test — filled at end
                hist["train_rmse"].append(float("nan"))
                hist["test_rmse"].append(float("nan"))

                update_epoch(progress, task_id, epoch, config.epochs, {
                    "D": hist["D_loss"][-1],
                    "G": hist["G_loss"][-1],
                    "Val": va_stats["rmse"],
                })

                va_rmse = va_stats["rmse"]
                if np.isnan(va_rmse):
                    log_warn(f"Validation RMSE is NaN at epoch {epoch}. bad_epochs={bad_epochs+1}/{patience}")
                    bad_epochs += 1
                elif va_rmse < best_val - 1e-7:
                    log_step(f"Epoch {epoch}: new best val RMSE={va_rmse:.6f} (prev={best_val:.6f})")
                    best_val = va_rmse
                    best_state = copy.deepcopy(G.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    log_debug(f"Epoch {epoch}: no improvement. val_rmse={va_rmse:.6f}, best={best_val:.6f}, bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        log_info(f"Early stopping at epoch {epoch}. best_val={best_val:.6f}")
                        if config.checkpoint_dir:
                            ckpt_dir = Path(config.checkpoint_dir)
                            _save_checkpoint(
                                ckpt_dir / "checkpoint_latest.pt", epoch,
                                G, D, optG, optD, scaler_G, scaler_D,
                                best_val, best_state, bad_epochs, hist,
                                ema_shadow, current_batch_size, config,
                            )
                        break
            else:
                # Non-eval epoch: just log losses
                update_epoch(progress, task_id, epoch, config.epochs, {
                    "D": hist["D_loss"][-1],
                    "G": hist["G_loss"][-1],
                })

            # --- Periodic checkpoint save ---
            if (
                config.checkpoint_dir
                and config.checkpoint_every > 0
                and epoch % config.checkpoint_every == 0
            ):
                ckpt_dir = Path(config.checkpoint_dir)
                ckpt_path = ckpt_dir / "checkpoint_latest.pt"
                log_step(f"Saving periodic checkpoint at epoch {epoch} to {ckpt_path}")
                _save_checkpoint(
                    ckpt_path, epoch,
                    G, D, optG, optD, scaler_G, scaler_D,
                    best_val, best_state, bad_epochs, hist,
                    ema_shadow, current_batch_size, config,
                )

    if best_state is not None:
        log_step(f"Loading best model state (best_val={best_val:.6f})")
        G.load_state_dict(best_state)
    else:
        log_warn("No best state was saved (training may not have improved from initial state)")

    if use_ema and ema_shadow is not None:
        log_step("Creating EMA model copy for final evaluation")
        G_ema = copy.deepcopy(G)
        _swap_params_with_shadow(G_ema, ema_shadow)
    else:
        G_ema = None

    eval_bs = max(1, min(config.eval_batch_size, current_batch_size))
    eval_model = _select_eval_generator(G, G_ema)
    eval_Xtr = preloaded_splits["Xtr"] if preloaded_splits is not None else Xtr
    eval_Ytr = preloaded_splits["Ytr"] if preloaded_splits is not None else Ytr
    eval_Xte = preloaded_splits["Xte"] if preloaded_splits is not None else Xte
    eval_Yte = preloaded_splits["Yte"] if preloaded_splits is not None else Yte
    log_step(f"Computing final train metrics (eval_bs={eval_bs}, samples={len(Xtr)})")
    train_stats, train_pred = compute_metrics(eval_model, eval_Xtr, eval_Ytr, batch_size=eval_bs)
    log_step(f"Final train RMSE={train_stats.get('rmse', 'N/A'):.6f}")
    log_step(f"Computing final test metrics (samples={len(Xte)})")
    test_stats, Y_pred = compute_metrics(eval_model, eval_Xte, eval_Yte, batch_size=eval_bs)
    log_step(f"Final test RMSE={test_stats.get('rmse', 'N/A'):.6f}")

    return {
        "G": G,
        "G_ema": G_ema,
        "D": D,
        "pipeline": "joint",
        "history": hist,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "pred_train": train_pred,
        "pred_test": Y_pred,
        "lambda_reg_final": lambda_reg,
        "resumed_from_epoch": resumed_from_epoch,
    }


# ---------------------------------------------------------------------------
# Two-Stage Pipeline (Paper Algorithm 1)
# ---------------------------------------------------------------------------


def _predict_batched(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run model inference in batches and return numpy predictions."""
    model.eval()
    n = len(X)
    parts: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = _prepare_batch_tensor(X[start:end], device)
            Yb = model(Xb)
            parts.append(Yb.detach().cpu().float().numpy())
    return np.concatenate(parts, axis=0)


def compute_hybrid_metrics(
    regression_model: nn.Module,
    residual_generator: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 512,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute metrics using the hybrid forecast: f̂(X) + G(z).

    When deterministic=True, uses f̂(X) only (G(z) has E[G(z)]≈0 when trained
    well, so omitting it gives the point forecast). When deterministic=False,
    adds a single stochastic residual sample.
    """
    Y_pred = predict_hybrid_forecasts(
        regression_model,
        residual_generator,
        X,
        batch_size=batch_size,
        deterministic=deterministic,
        seed=seed,
    )

    if torch.is_tensor(Y):
        Y_ref = Y.detach().cpu().numpy()
    else:
        Y_ref = Y
    stats = error_stats(Y_ref.reshape(-1), Y_pred.reshape(-1))
    return stats, Y_pred


def predict_hybrid_forecasts(
    regression_model: nn.Module,
    residual_generator: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    batch_size: int = 512,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Run two-stage hybrid inference, optionally sampling residual noise."""
    device = next(regression_model.parameters()).device
    Y_reg = _predict_batched(regression_model, X, device, batch_size)
    if deterministic:
        return Y_reg

    residual_generator.eval()
    noise_dim = residual_generator.noise_dim
    H = residual_generator.H
    n = len(X)
    rng = np.random.default_rng(seed) if seed is not None else None
    residual_parts: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bs = end - start
            z_np = (rng.standard_normal((bs, H, noise_dim)).astype(np.float32)
                    if rng is not None else np.random.randn(bs, H, noise_dim).astype(np.float32))
            z = torch.from_numpy(z_np).to(device=device, non_blocking=True)
            r = residual_generator(z)
            residual_parts.append(r.detach().cpu().float().numpy())
    residuals = np.concatenate(residual_parts, axis=0)
    return Y_reg + residuals


def _score_residual_wgan(
    critic: nn.Module,
    residual_generator: nn.Module,
    residuals: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    batch_size: int = 512,
    seed: Optional[int] = None,
) -> float:
    """Estimate validation Wasserstein gap on held-out residuals."""
    critic.eval()
    residual_generator.eval()
    noise_dim = residual_generator.noise_dim
    horizon = residual_generator.H
    rng = np.random.default_rng(seed) if seed is not None else None
    real_scores: List[float] = []
    fake_scores: List[float] = []
    n = len(residuals)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Rb = _prepare_batch_tensor(residuals[start:end], device).float()
            bs = end - start
            z_np = (rng.standard_normal((bs, horizon, noise_dim)).astype(np.float32)
                    if rng is not None else np.random.randn(bs, horizon, noise_dim).astype(np.float32))
            z = torch.from_numpy(z_np).to(device=device, non_blocking=True)
            R_fake = residual_generator(z)
            real_scores.append(float(critic(Rb).mean().detach().item()))
            fake_scores.append(float(critic(R_fake).mean().detach().item()))
    return float(np.mean(real_scores) - np.mean(fake_scores))


def train_two_stage(
    config: TrainConfig,
    models: Tuple[nn.Module, nn.Module, nn.Module],
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "rgan_two_stage",
) -> Dict[str, Any]:
    """Train the two-stage Regression-WGAN pipeline (Paper Algorithm 1).

    Stage 1: Train regression model f̂(X) on MSE loss.
    Stage 2: Compute residuals, train WGAN G(z)/D(r) on residuals.

    Args:
        config: Training configuration.
        models: Tuple of (RegressionModel, ResidualGenerator, Discriminator).
        data_splits: Dict with Xtr, Ytr, Xval, Yval, Xte, Yte.
        results_dir: Directory for saving results.
        tag: Experiment tag.

    Returns:
        Dict with trained models, history, and evaluation stats.
    """
    F_hat, G, D = models
    console = get_console()

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        if config.strict_device:
            raise RuntimeError(f"CUDA device {config.device} requested but unavailable.")
        requested_device = torch.device("cpu")
    device = requested_device

    F_hat.to(device)
    G.to(device)
    D.to(device)

    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]
    Xte, Yte = data_splits["Xte"], data_splits["Yte"]

    log_info("=" * 60)
    log_info("TWO-STAGE PIPELINE (Paper Algorithm 1)")
    log_info("=" * 60)

    # ------------------------------------------------------------------
    # STAGE 1: Train Regression Model f̂ (Algorithm Steps 3-6)
    # ------------------------------------------------------------------
    log_info("STAGE 1: Training Regression Model f̂(X)")
    log_step(f"  epochs={config.regression_epochs}, lr={config.regression_lr}, patience={config.regression_patience}")

    opt_f = torch.optim.Adam(F_hat.parameters(), lr=config.regression_lr)
    mse_loss = nn.MSELoss()
    best_val_rmse = float("inf")
    best_f_state = None
    bad_epochs = 0
    reg_history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_rmse": []}

    # Create DataLoader for Stage 1
    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(Ytr).float(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=0, drop_last=False,
    )

    eval_bs = max(1, config.eval_batch_size)

    with epoch_progress(config.regression_epochs, description="Stage 1: Regression") as (progress, task_id):
        for epoch in range(1, config.regression_epochs + 1):
            F_hat.train()
            epoch_losses: List[float] = []
            for Xb, Yb in train_loader:
                Xb = Xb.to(device, non_blocking=True)
                Yb = Yb.to(device, non_blocking=True)
                opt_f.zero_grad(set_to_none=True)
                Y_pred = F_hat(Xb)
                if Y_pred.dtype != Yb.dtype:
                    Yb = Yb.to(dtype=Y_pred.dtype)
                loss = mse_loss(Y_pred, Yb)
                loss.backward()
                clip_grad_value_(F_hat.parameters(), config.grad_clip)
                opt_f.step()
                epoch_losses.append(loss.detach().item())

            mean_loss = float(np.mean(epoch_losses))
            reg_history["train_loss"].append(mean_loss)

            # Validate
            val_stats, _ = compute_metrics(F_hat, Xval, Yval, batch_size=eval_bs)
            val_rmse = val_stats["rmse"]
            reg_history["epoch"].append(epoch)
            reg_history["val_rmse"].append(val_rmse)

            update_epoch(progress, task_id, epoch, config.regression_epochs, {
                "Loss": mean_loss, "Val": val_rmse,
            })

            if np.isnan(val_rmse):
                bad_epochs += 1
            elif val_rmse < best_val_rmse - 1e-7:
                log_step(f"  Epoch {epoch}: new best val RMSE={val_rmse:.6f}")
                best_val_rmse = val_rmse
                best_f_state = copy.deepcopy(F_hat.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= config.regression_patience:
                    log_info(f"  Early stopping at epoch {epoch}, best_val={best_val_rmse:.6f}")
                    break

    if best_f_state is not None:
        F_hat.load_state_dict(best_f_state)
    log_info(f"Stage 1 complete. Best val RMSE={best_val_rmse:.6f}")

    # ------------------------------------------------------------------
    # Compute Residuals (Algorithm Step 6): r_t = Y - f̂(X)
    # ------------------------------------------------------------------
    log_info("Computing residuals r_t = Y - f̂(X)")
    Y_hat_train = _predict_batched(F_hat, Xtr, device, eval_bs)
    R_train = Ytr - Y_hat_train  # (N, H, 1) residuals

    Y_hat_val = _predict_batched(F_hat, Xval, device, eval_bs)
    R_val = Yval - Y_hat_val

    log_step(f"  Residuals — train mean={R_train.mean():.6f}, std={R_train.std():.6f}")
    log_step(f"  Residuals — val   mean={R_val.mean():.6f}, std={R_val.std():.6f}")

    # ------------------------------------------------------------------
    # STAGE 2: Train WGAN on Residuals (Algorithm Steps 7-20)
    # ------------------------------------------------------------------
    log_info("STAGE 2: Training WGAN on Residuals")
    log_step(f"  epochs={config.epochs}, d_steps={config.d_steps}, gp_lambda={config.wgan_gp_lambda}")

    noise_dim = G.noise_dim
    H = G.H
    optG = torch.optim.Adam(G.parameters(), lr=config.lr_g, weight_decay=config.weight_decay)
    optD = torch.optim.Adam(D.parameters(), lr=config.lr_d, weight_decay=config.weight_decay)

    d_steps = max(1, config.d_steps)
    gp_lambda = config.wgan_gp_lambda

    residual_ds = TensorDataset(torch.from_numpy(R_train).float())
    residual_loader = DataLoader(
        residual_ds, batch_size=config.batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=0, drop_last=True,
    )

    wgan_history: Dict[str, List[float]] = {
        "epoch": [], "D_loss": [], "G_loss": [],
        "D_real_mean": [], "D_fake_mean": [],
        "val_wasserstein": [],
    }
    best_val_gap = float("-inf")
    best_g_state = copy.deepcopy(G.state_dict())
    best_d_state = copy.deepcopy(D.state_dict())
    bad_wgan_epochs = 0
    val_eval_seed = int(config.seed) + 1729

    with epoch_progress(config.epochs, description="Stage 2: WGAN Residuals") as (progress, task_id):
        for epoch in range(1, config.epochs + 1):
            G.train()
            D.train()
            epoch_D_losses: List[float] = []
            epoch_G_losses: List[float] = []
            epoch_D_real: List[float] = []
            epoch_D_fake: List[float] = []

            for (Rb,) in residual_loader:
                Rb = Rb.to(device, non_blocking=True)
                bs = Rb.size(0)

                # --- Critic update (k steps) ---
                for _ in range(d_steps):
                    optD.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        z_d = torch.randn(bs, H, noise_dim, device=device)
                        R_fake = G(z_d)

                    D_real = D(Rb)
                    D_fake = D(R_fake)
                    D_loss_main = -(D_real.mean() - D_fake.mean())

                    # Gradient penalty
                    gp = _gradient_penalty(D, Rb, R_fake, device, gp_lambda)
                    D_loss = D_loss_main + gp

                    D_loss.backward()
                    clip_grad_value_(D.parameters(), config.grad_clip)
                    optD.step()

                    epoch_D_losses.append(D_loss_main.detach().item())
                    epoch_D_real.append(D_real.mean().detach().item())
                    epoch_D_fake.append(D_fake.mean().detach().item())

                # --- Generator update ---
                optG.zero_grad(set_to_none=True)
                z_g = torch.randn(bs, H, noise_dim, device=device)
                R_fake = G(z_g)
                G_loss = -D(R_fake).mean()

                G_loss.backward()
                clip_grad_value_(G.parameters(), config.grad_clip)
                optG.step()

                epoch_G_losses.append(G_loss.detach().item())

            wgan_history["epoch"].append(epoch)
            wgan_history["D_loss"].append(float(np.mean(epoch_D_losses)))
            wgan_history["G_loss"].append(float(np.mean(epoch_G_losses)))
            wgan_history["D_real_mean"].append(float(np.mean(epoch_D_real)))
            wgan_history["D_fake_mean"].append(float(np.mean(epoch_D_fake)))
            val_gap = _score_residual_wgan(
                D,
                G,
                R_val,
                device,
                batch_size=eval_bs,
                seed=val_eval_seed,
            ) if len(R_val) else float("nan")
            wgan_history["val_wasserstein"].append(val_gap)

            update_epoch(progress, task_id, epoch, config.epochs, {
                "D": wgan_history["D_loss"][-1],
                "G": wgan_history["G_loss"][-1],
                "ValGap": val_gap,
            })

            if np.isnan(val_gap):
                bad_wgan_epochs += 1
            elif val_gap > best_val_gap + 1e-7:
                log_step(f"  Epoch {epoch}: new best residual val gap={val_gap:.6f}")
                best_val_gap = val_gap
                best_g_state = copy.deepcopy(G.state_dict())
                best_d_state = copy.deepcopy(D.state_dict())
                bad_wgan_epochs = 0
            else:
                bad_wgan_epochs += 1
                if bad_wgan_epochs >= config.patience:
                    log_info(
                        f"  Residual WGAN early stopping at epoch {epoch}, best_val_gap={best_val_gap:.6f}"
                    )
                    break

    log_info("Stage 2 complete.")
    G.load_state_dict(best_g_state)
    D.load_state_dict(best_d_state)

    # ------------------------------------------------------------------
    # Final Evaluation — Hybrid Forecast: x̂* = f̂(X) + G(z)
    # ------------------------------------------------------------------
    log_info("Computing final metrics with hybrid forecast f̂(X) + G(z)")

    train_stats, train_pred = compute_hybrid_metrics(
        F_hat, G, Xtr, Ytr, batch_size=eval_bs, deterministic=True,
    )
    test_stats, test_pred = compute_hybrid_metrics(
        F_hat, G, Xte, Yte, batch_size=eval_bs, deterministic=True,
    )
    log_step(f"Final train RMSE={train_stats['rmse']:.6f}")
    log_step(f"Final test  RMSE={test_stats['rmse']:.6f}")

    return {
        "F_hat": F_hat,
        "G": G,
        "G_ema": None,
        "D": D,
        "history": {
            "regression": reg_history,
            "wgan": wgan_history,
        },
        "train_stats": train_stats,
        "test_stats": test_stats,
        "pred_train": train_pred,
        "pred_test": test_pred,
        "pipeline": "two_stage",
    }
