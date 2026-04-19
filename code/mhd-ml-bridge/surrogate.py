"""Brio-Wu MLP surrogate in JAX.

Scaffolding, not science. The point is to prove the stack runs end-to-end:
  classical solver -> training data -> JAX MLP -> predictions on a held-out set.

The surrogate is not expected to beat HLL. Week 5+ is where it gets physics-y.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from data import N_FEATURES, N_OUT, N_OUTPUTS, build_dataset

HIDDEN = (128, 128)
N_TRAIN = 96
N_VAL = 32
EPOCHS = 800
BATCH = 16
LR = 1e-3
SEED = 0


class Params(NamedTuple):
    weights: list  # list of (W, b) per layer
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray


def init_mlp(key: jax.Array, sizes: tuple[int, ...]) -> list:
    layers = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        # Glorot init
        scale = jnp.sqrt(2.0 / (n_in + n_out))
        W = jax.random.normal(k, (n_in, n_out)) * scale
        b = jnp.zeros(n_out)
        layers.append((W, b))
    return layers


def mlp_forward(weights: list, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for i, (W, b) in enumerate(weights):
        h = h @ W + b
        if i < len(weights) - 1:
            h = jax.nn.gelu(h)
    return h


def normalize(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return (x - mean) / (std + 1e-8)


def predict(params: Params, X: jnp.ndarray) -> jnp.ndarray:
    Xn = normalize(X, params.x_mean, params.x_std)
    Yn = mlp_forward(params.weights, Xn)
    return Yn * (params.y_std + 1e-8) + params.y_mean


def loss_fn(weights, X, Y, x_mean, x_std, y_mean, y_std):
    Xn = normalize(X, x_mean, x_std)
    Yn_true = normalize(Y, y_mean, y_std)
    Yn_pred = mlp_forward(weights, Xn)
    return jnp.mean((Yn_pred - Yn_true) ** 2)


def train():
    print("generating data...")
    X_train_np, Y_train_np = build_dataset(N_TRAIN, seed=SEED)
    X_val_np, Y_val_np = build_dataset(N_VAL, seed=SEED + 1)

    # Fit normalization stats on the training set only.
    x_mean = jnp.asarray(X_train_np.mean(axis=0))
    x_std = jnp.asarray(X_train_np.std(axis=0))
    y_mean = jnp.asarray(Y_train_np.mean(axis=0))
    y_std = jnp.asarray(Y_train_np.std(axis=0))

    X_train = jnp.asarray(X_train_np)
    Y_train = jnp.asarray(Y_train_np)
    X_val = jnp.asarray(X_val_np)
    Y_val = jnp.asarray(Y_val_np)

    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    sizes = (N_FEATURES, *HIDDEN, N_OUTPUTS)
    weights = init_mlp(init_key, sizes)

    opt = optax.adam(LR)
    opt_state = opt.init(weights)

    @jax.jit
    def step(weights, opt_state, xb, yb):
        l, grads = jax.value_and_grad(loss_fn)(weights, xb, yb, x_mean, x_std, y_mean, y_std)
        updates, opt_state = opt.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, l

    @jax.jit
    def eval_loss(weights, X, Y):
        return loss_fn(weights, X, Y, x_mean, x_std, y_mean, y_std)

    train_hist = []
    val_hist = []
    print(f"training for {EPOCHS} epochs, batch={BATCH}, N_train={N_TRAIN}")
    for epoch in range(EPOCHS):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, N_TRAIN)
        losses = []
        for start in range(0, N_TRAIN, BATCH):
            idx = perm[start : start + BATCH]
            weights, opt_state, l = step(weights, opt_state, X_train[idx], Y_train[idx])
            losses.append(float(l))
        train_loss = float(np.mean(losses))
        val_loss = float(eval_loss(weights, X_val, Y_val))
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:4d}  train={train_loss:.4e}  val={val_loss:.4e}")

    params = Params(weights, x_mean, x_std, y_mean, y_std)

    # Per-field MSE on validation (unnormalized).
    Y_pred = np.asarray(predict(params, X_val))
    Y_val_np_ = np.asarray(Y_val)
    field_names = ["rho", "u", "By", "p"]
    print("\nval MSE per field (unnormalized):")
    for i, name in enumerate(field_names):
        pred_f = Y_pred[:, i * N_OUT : (i + 1) * N_OUT]
        true_f = Y_val_np_[:, i * N_OUT : (i + 1) * N_OUT]
        mse = float(np.mean((pred_f - true_f) ** 2))
        rng = float(true_f.max() - true_f.min())
        print(f"  {name:>3s}: MSE={mse:.4e}   range={rng:.3f}   rMSE/range={np.sqrt(mse)/rng:.3f}")

    # Plots.
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(train_hist, label="train")
    ax.semilogy(val_hist, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized MSE")
    ax.set_title(f"Brio-Wu MLP surrogate: {HIDDEN} hidden, N_train={N_TRAIN}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    loss_path = os.path.join(out_dir, "training_curve.png")
    fig.savefig(loss_path, dpi=140)
    print(f"wrote {loss_path}")

    # Prediction-vs-truth panel for one validation example.
    dx = 1.0 / N_OUT
    x_grid = (np.arange(N_OUT) + 0.5) * dx
    idx0 = 0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, i, name in zip(axes.flat, range(4), field_names):
        true_f = Y_val_np_[idx0, i * N_OUT : (i + 1) * N_OUT]
        pred_f = Y_pred[idx0, i * N_OUT : (i + 1) * N_OUT]
        ax.plot(x_grid, true_f, "k-", lw=1.5, label="HLL (truth)")
        ax.plot(x_grid, pred_f, "r--", lw=1.2, label="MLP")
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("Val example 0: MLP prediction vs HLL truth at t=0.1")
    fig.tight_layout()
    pred_path = os.path.join(out_dir, "val_example.png")
    fig.savefig(pred_path, dpi=140)
    print(f"wrote {pred_path}")

    return params, train_hist, val_hist


if __name__ == "__main__":
    train()
