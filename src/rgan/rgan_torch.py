import copy
from contextlib import nullcontext
from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch
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


def compute_metrics(
    G: nn.Module, X: np.ndarray, Y: np.ndarray, batch_size: int = 512
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
    G.eval()
    # Determine device from model parameters
    device = next(G.parameters()).device

    n_samples = len(X)
    bs = max(1, int(batch_size))
    Yp = np.empty_like(Y)

    while True:
        try:
            idx = 0
            with torch.inference_mode():
                while idx < n_samples:
                    end = min(idx + bs, n_samples)
                    Xb = torch.from_numpy(X[idx:end]).to(device=device)
                    with AMP.autocast(
                        device.type, enabled=(device.type == "cuda" and AMP.available)
                    ):
                        Yb = G(Xb)
                    Yp[idx:end] = Yb.detach().cpu().numpy()
                    idx = end
                if device.type == "cuda":
                    torch.cuda.synchronize()
            break
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)

    stats = error_stats(Y.reshape(-1), Yp.reshape(-1))
    return stats, Yp


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
        console.print(
            f"[yellow]CUDA unavailable — falling back to CPU despite requested device {config.device}. {hint}[/yellow]"
        )
        requested_device = torch.device("cpu")

    if requested_device.type == "cuda":
        torch.cuda.set_device(requested_device)
        device_name = torch.cuda.get_device_name(requested_device)
        console.print(f"[green]Using CUDA device:[/green] {device_name} ({requested_device})")
    else:
        console.print("[yellow]Using CPU for training (no CUDA device active).[/yellow]")

    device = requested_device
    G.to(device)
    D.to(device)

    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]
    Xte, Yte = data_splits["Xte"], data_splits["Yte"]

    optG = torch.optim.Adam(G.parameters(), lr=config.lr_g)
    optD = torch.optim.Adam(D.parameters(), lr=config.lr_d)
    mse = nn.MSELoss()

    use_logits = config.use_logits

    gan_variant = config.gan_variant.lower()
    if gan_variant not in {"standard", "wgan", "wgan-gp"}:
        raise ValueError(f"Unsupported GAN variant '{config.gan_variant}'.")
    if gan_variant in {"wgan", "wgan-gp"}:
        use_logits = True  # Wasserstein training expects raw scores

    bce_loss = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    default_d_steps = 5 if gan_variant in {"wgan", "wgan-gp"} else 1
    d_steps = max(1, config.d_steps)
    g_steps = max(1, config.g_steps)
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
            with AMP.autocast("cuda", enabled=False):
                return bce_loss(inputs_fp32, targets_fp32)
        return bce_loss(inputs_fp32, targets_fp32)

    current_batch_size = max(1, config.batch_size)
    amp_requested = config.amp
    amp_available = AMP.available
    amp_enabled = amp_requested and (device.type == "cuda") and amp_available
    
    if amp_requested and not amp_enabled:
        if device.type != "cuda":
            get_console().print(
                "[yellow]AMP requested but no CUDA device is active; running in full precision.[/yellow]"
            )
        elif not amp_available:
            get_console().print(
                "[yellow]AMP requested but unsupported by this PyTorch build; running in full precision.[/yellow]"
            )

    scaler_G = AMP.make_scaler(enabled=amp_enabled)
    scaler_D = AMP.make_scaler(enabled=amp_enabled)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    use_ema = 0.0 < ema_decay < 1.0
    ema_shadow: Optional[Dict[str, torch.Tensor]] = _init_ema(G) if use_ema else None

    hist: Dict[str, List[float]] = {
        "epoch": [],
        "train_rmse": [],
        "test_rmse": [],
        "val_rmse": [],
        "D_loss": [],
        "G_loss": [],
        "G_adv": [],
        "G_reg": [],
        "grad_norm_G": [],
        "grad_norm_D": [],
        "batch_size": [],
    }
    if track_logits:
        hist["D_real_mean"] = []
        hist["D_fake_mean"] = []

    num_workers = config.num_workers
    # Optimization: Increase prefetch factor for high-end GPUs
    prefetch_factor = 4 
    persistent_workers = (num_workers > 0)
    pin_memory = (device.type == "cuda")

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))

    def make_loader(bs: int) -> DataLoader:
        kwargs = dict(
            batch_size=bs,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = max(1, prefetch_factor)
        return DataLoader(train_ds, **kwargs)

    train_loader = make_loader(current_batch_size)

    with epoch_progress(config.epochs, description="R-GAN (Torch)") as (progress, task_id):
        for epoch in range(1, config.epochs + 1):
            supervised_only = epoch <= warmup_epochs
            lambda_reg = lambda_reg_end
            if lambda_reg_warmup > 1:
                progress_ratio = min(1.0, max(0.0, (epoch - 1) / (lambda_reg_warmup - 1)))
                lambda_reg = lambda_reg_start + (lambda_reg_end - lambda_reg_start) * progress_ratio

            epoch_D_losses: List[float] = []
            epoch_G_losses: List[float] = []
            epoch_adv: List[float] = []
            epoch_reg: List[float] = []
            epoch_grad_G: List[float] = []
            epoch_grad_D: List[float] = []
            epoch_D_real: List[float] = []
            epoch_D_fake: List[float] = []
            
            noise_std_epoch = instance_noise_std * (instance_noise_decay ** max(0, epoch - 1))

            G.train()
            D.train()

            while True:
                try:
                    for Xb, Yb in train_loader:
                        Xb = Xb.to(device, non_blocking=True)
                        Yb = Yb.to(device, non_blocking=True)
                        target_series = Xb[..., :1]

                        if not supervised_only:
                            for _ in range(d_steps):
                                optD.zero_grad(set_to_none=True)
                                with torch.no_grad():
                                    with AMP.autocast("cuda", enabled=amp_enabled):
                                        Y_fake_detached = G(Xb)
                                if Y_fake_detached.dtype != target_series.dtype:
                                    Y_fake_detached = Y_fake_detached.to(dtype=target_series.dtype)

                                with AMP.autocast("cuda", enabled=amp_enabled):
                                    real_pairs = torch.cat([target_series, Yb], dim=1)
                                    fake_pairs = torch.cat([target_series, Y_fake_detached], dim=1)
                                    if noise_std_epoch > 0:
                                        real_pairs = _apply_instance_noise(real_pairs, noise_std_epoch)
                                        fake_pairs = _apply_instance_noise(fake_pairs, noise_std_epoch)

                                    D_real = D(real_pairs)
                                    D_fake = D(fake_pairs)

                                if gan_variant in {"wgan", "wgan-gp"}:
                                    D_loss_main = -(D_real.mean() - D_fake.mean())
                                    if gan_variant == "wgan-gp":
                                        gp = _gradient_penalty(D, real_pairs, fake_pairs, device, gp_lambda)
                                        D_loss = D_loss_main + gp
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

                                grad_sq = 0.0
                                for p in D.parameters():
                                    if p.grad is not None:
                                        grad_sq += p.grad.detach().pow(2).sum().item()
                                epoch_grad_D.append(float(np.sqrt(grad_sq)))

                                scaler_D.step(optD)
                                scaler_D.update()

                                if gan_variant == "wgan":
                                    clip_val = float(config.wgan_clip_value)
                                    for p in D.parameters():
                                        if p.requires_grad:
                                            p.data.clamp_(-clip_val, clip_val)

                                epoch_D_losses.append(float(D_loss_main.detach().cpu()))
                                if track_logits:
                                    if gan_variant in {"wgan", "wgan-gp"}:
                                        epoch_D_real.append(float(D_real.mean().detach().cpu()))
                                        epoch_D_fake.append(float(D_fake.mean().detach().cpu()))
                                    else:
                                        real_prob = torch.sigmoid(D_real) if use_logits else _disc_output_to_prob(D_real)
                                        fake_prob = torch.sigmoid(D_fake) if use_logits else _disc_output_to_prob(D_fake)
                                        epoch_D_real.append(float(real_prob.mean().detach().cpu()))
                                        epoch_D_fake.append(float(fake_prob.mean().detach().cpu()))

                        for _ in range(g_steps):
                            optG.zero_grad(set_to_none=True)
                            with AMP.autocast("cuda", enabled=amp_enabled):
                                Y_fake = G(Xb)
                                cat_target = target_series
                                if cat_target.dtype != Y_fake.dtype:
                                    cat_target = cat_target.to(dtype=Y_fake.dtype)
                                fake_pairs = torch.cat([cat_target, Y_fake], dim=1)
                                if noise_std_epoch > 0 and not supervised_only:
                                    fake_pairs = _apply_instance_noise(fake_pairs, noise_std_epoch)

                                if supervised_only:
                                    adv_loss = torch.tensor(0.0, device=device)
                                elif gan_variant in {"wgan", "wgan-gp"}:
                                    adv_loss = -D(fake_pairs).mean()
                                else:
                                    logits = D(fake_pairs)
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
                                total_loss = adv_weight * adv_loss + lambda_reg * reg_loss

                            scaler_G.scale(total_loss).backward()
                            scaler_G.unscale_(optG)
                            clip_grad_value_(G.parameters(), config.grad_clip)

                            grad_sq = 0.0
                            for p in G.parameters():
                                if p.grad is not None:
                                    grad_sq += p.grad.detach().pow(2).sum().item()
                            epoch_grad_G.append(float(np.sqrt(grad_sq)))

                            scaler_G.step(optG)
                            scaler_G.update()

                            epoch_G_losses.append(float(total_loss.detach().cpu()))
                            epoch_adv.append(float(adv_loss.detach().cpu()))
                            epoch_reg.append(float(reg_loss.detach().cpu()))

                            if use_ema and ema_shadow is not None:
                                _update_ema(ema_shadow, G, ema_decay)

                        del Xb, Yb, target_series

                    break
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    optD.zero_grad(set_to_none=True)
                    optG.zero_grad(set_to_none=True)
                    if current_batch_size == 1:
                        raise
                    current_batch_size = max(1, current_batch_size // 2)
                    console.log(f"[R-GAN Torch] CUDA OOM detected. Reducing batch size to {current_batch_size}.")
                    train_loader = make_loader(current_batch_size)
                    continue

            hist["batch_size"].append(current_batch_size)
            hist["grad_norm_G"].append(float(np.mean(epoch_grad_G) if epoch_grad_G else 0.0))
            hist["grad_norm_D"].append(float(np.mean(epoch_grad_D) if epoch_grad_D else 0.0))
            hist["D_loss"].append(float(np.mean(epoch_D_losses) if epoch_D_losses else 0.0))
            hist["G_loss"].append(float(np.mean(epoch_G_losses) if epoch_G_losses else 0.0))
            hist["G_adv"].append(float(np.mean(epoch_adv) if epoch_adv else 0.0))
            hist["G_reg"].append(float(np.mean(epoch_reg) if epoch_reg else 0.0))
            if track_logits:
                hist["D_real_mean"].append(float(np.mean(epoch_D_real) if epoch_D_real else 0.0))
                hist["D_fake_mean"].append(float(np.mean(epoch_D_fake) if epoch_D_fake else 0.0))

            # Use eval_batch_size from config if present, else default to batch_size
            # Since we don't have eval_batch_size in TrainConfig yet, we'll just use batch_size
            # or add it to TrainConfig. Let's assume batch_size for now or hardcode 512 cap.
            eval_bs = max(1, min(512, current_batch_size))

            if use_ema and ema_shadow is not None:
                backup = _swap_params_with_shadow(G, ema_shadow)
                tr_stats, _ = compute_metrics(G, Xtr, Ytr, batch_size=eval_bs)
                te_stats, _ = compute_metrics(G, Xte, Yte, batch_size=eval_bs)
                va_stats, _ = compute_metrics(G, Xval, Yval, batch_size=eval_bs)
                _swap_params_with_shadow(G, backup)
            else:
                tr_stats, _ = compute_metrics(G, Xtr, Ytr, batch_size=eval_bs)
                te_stats, _ = compute_metrics(G, Xte, Yte, batch_size=eval_bs)
                va_stats, _ = compute_metrics(G, Xval, Yval, batch_size=eval_bs)

            hist["epoch"].append(epoch)
            hist["train_rmse"].append(tr_stats["rmse"])
            hist["test_rmse"].append(te_stats["rmse"])
            hist["val_rmse"].append(va_stats["rmse"])

            update_epoch(progress, task_id, epoch, config.epochs, {
                "D": hist["D_loss"][-1],
                "G": hist["G_loss"][-1],
                "Train": tr_stats["rmse"],
                "Val": va_stats["rmse"],
                "Test": te_stats["rmse"],
            })

            if va_stats["rmse"] < best_val - 1e-7:
                best_val = va_stats["rmse"]
                best_state = copy.deepcopy(G.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    console.log(f"[R-GAN Torch] Early stopping at epoch {epoch}.")
                    break

    if best_state is not None:
        G.load_state_dict(best_state)

    if use_ema and ema_shadow is not None:
        G_ema = copy.deepcopy(G)
        _swap_params_with_shadow(G_ema, ema_shadow)
    else:
        G_ema = None

    eval_bs = max(1, min(512, current_batch_size))
    eval_model = G_ema or G
    train_stats, _ = compute_metrics(eval_model, Xtr, Ytr, batch_size=eval_bs)
    test_stats, Y_pred = compute_metrics(eval_model, Xte, Yte, batch_size=eval_bs)

    return {
        "G": G,
        "G_ema": G_ema,
        "D": D,
        "history": hist,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "pred_test": Y_pred,
        "lambda_reg_final": lambda_reg,
    }
