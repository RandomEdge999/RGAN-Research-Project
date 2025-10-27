import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

def step_discriminator(D, optD, bce, Xb, Yb, G, label_smooth=0.9, clip_value=5.0):
    with tf.GradientTape() as tape:
        Y_fake = G(Xb, training=True)
        real_pairs = tf.concat([Xb[..., :1], Yb], axis=1)
        fake_pairs = tf.concat([Xb[..., :1], Y_fake], axis=1)
        D_real = D(real_pairs, training=True)
        D_fake = D(fake_pairs, training=True)
        y_real = tf.ones_like(D_real) * label_smooth
        y_fake = tf.zeros_like(D_fake)
        loss_real = bce(y_real, D_real)
        loss_fake = bce(y_fake, D_fake)
        D_loss = 0.5 * (loss_real + loss_fake)
    grads = tape.gradient(D_loss, D.trainable_variables)
    grads = [tf.clip_by_value(g, -clip_value, clip_value) if g is not None else None for g in grads]
    optD.apply_gradients(zip(grads, D.trainable_variables))
    return D_loss

def step_generator(G, D, optG, bce, mse, Xb, Yb, lambda_reg=0.1, clip_value=5.0):
    with tf.GradientTape() as tape:
        Y_fake = G(Xb, training=True)
        fake_pairs = tf.concat([Xb[..., :1], Y_fake], axis=1)
        D_fake = D(fake_pairs, training=True)
        y_adv = tf.ones_like(D_fake)
        adv_loss = bce(y_adv, D_fake)
        reg_loss = mse(Yb, Y_fake)
        G_loss = adv_loss + lambda_reg * reg_loss
    grads = tape.gradient(G_loss, G.trainable_variables)
    grads = [tf.clip_by_value(g, -clip_value, clip_value) if g is not None else None for g in grads]
    optG.apply_gradients(zip(grads, G.trainable_variables))
    return G_loss, adv_loss, reg_loss

def compute_rmse_mae(G, X, Y) -> Tuple[float, float, np.ndarray]:
    Yp = G.predict(X, verbose=0)
    y_true = Y.reshape(-1)
    y_pred = Yp.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae, Yp

def train_rgan_keras(config: Dict, models, data_splits, results_dir: str, tag="rgan") -> Dict:
    G, D = models
    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]
    Xte,  Yte  = data_splits["Xte"],  data_splits["Yte"]

    optG = tf.keras.optimizers.Adam(learning_rate=config["lrG"])
    optD = tf.keras.optimizers.Adam(learning_rate=config["lrD"])
    bce = tf.keras.losses.BinaryCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()

    # Pre-build optimizer slot variables outside any tf.function to avoid tf.Variable creation errors across trials.
    try:
        optD.build(D.trainable_variables)
        optG.build(G.trainable_variables)
    except Exception:
        _zeroD = [tf.zeros_like(v) for v in D.trainable_variables]
        optD.apply_gradients(zip(_zeroD, D.trainable_variables))
        _zeroG = [tf.zeros_like(v) for v in G.trainable_variables]
        optG.apply_gradients(zip(_zeroG, G.trainable_variables))

    best_val = float("inf")
    best_weights = None
    patience, bad_epochs = config["patience"], 0
    steps = max(1, int(np.ceil(len(Xtr) / config["batch_size"])))

    hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": [],
            "D_loss": [], "G_loss": [], "G_adv": [], "G_reg": []}

    for epoch in range(1, config["epochs"] + 1):
        idx = np.random.permutation(len(Xtr))
        Xtr = Xtr[idx]; Ytr = Ytr[idx]
        D_losses, G_losses, G_advs, G_regs = [], [], [], []

        for s in range(steps):
            b0 = s * config["batch_size"]; b1 = min((s+1)*config["batch_size"], len(Xtr))
            Xb, Yb = Xtr[b0:b1], Ytr[b0:b1]
            Dl = step_discriminator(D, optD, bce, Xb, Yb, G, label_smooth=config["label_smooth"], clip_value=config["grad_clip"])
            Gl, Ga, Gr = step_generator(G, D, optG, bce, mse, Xb, Yb, lambda_reg=config["lambda_reg"], clip_value=config["grad_clip"])
            D_losses.append(float(Dl)); G_losses.append(float(Gl)); G_advs.append(float(Ga)); G_regs.append(float(Gr))

        tr_rmse, _, _ = compute_rmse_mae(G, Xtr, Ytr)
        te_rmse, _, _ = compute_rmse_mae(G, Xte, Yte)
        va_rmse, _, _ = compute_rmse_mae(G, Xval, Yval)

        hist["epoch"].append(epoch)
        hist["D_loss"].append(np.mean(D_losses)); hist["G_loss"].append(np.mean(G_losses))
        hist["G_adv"].append(np.mean(G_advs));   hist["G_reg"].append(np.mean(G_regs))
        hist["train_rmse"].append(tr_rmse); hist["test_rmse"].append(te_rmse); hist["val_rmse"].append(va_rmse)

        print(f"[R-GAN] Epoch {epoch:03d} | D {np.mean(D_losses):.4f} | G {np.mean(G_losses):.4f} | "
              f"Train {tr_rmse:.5f} | Val {va_rmse:.5f} | Test {te_rmse:.5f}")

        if va_rmse < best_val - 1e-7:
            best_val = va_rmse; best_weights = G.get_weights(); bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[R-GAN] Early stopping at epoch {epoch}."); break

    if best_weights is not None:
        G.set_weights(best_weights)

    train_rmse, train_mae, _ = compute_rmse_mae(G, Xtr, Ytr)
    test_rmse, test_mae, Y_pred = compute_rmse_mae(G, Xte, Yte)

    return {
        "G": G, "D": D, "history": hist,
        "train_rmse": train_rmse, "train_mae": train_mae,
        "test_rmse": test_rmse, "test_mae": test_mae,
        "pred_test": Y_pred
    }
