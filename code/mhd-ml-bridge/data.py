"""Generate training data for the Brio-Wu surrogate.

Sweeps a handful of parameters around the canonical Brio-Wu initial condition,
runs the 1D HLL MHD solver from ../mhd-1d/mhd.py, and returns (inputs, outputs)
arrays suitable for an MLP.

Inputs (per example, 7 scalars):
    rho_L, p_L, By_L, rho_R, p_R, By_R, Bx

Output (per example, flattened 4*N_out):
    [rho(x), u(x), By(x), p(x)]  each on a uniform grid of N_out cells
    at t = T_FINAL.

Fixed for this scaffold: gamma = 2.0, t_final = 0.1, domain = [0, 1].
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Import the solver from the sibling week-3 project.
_MHD_1D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mhd-1d"))
if _MHD_1D not in sys.path:
    sys.path.insert(0, _MHD_1D)

from mhd import (  # noqa: E402
    mhd_conservative_to_primitive,
    mhd_primitive_to_conservative,
    mhd_simulate,
)

GAMMA = 2.0
T_FINAL = 0.1
N_SIM = 128
N_OUT = 64
N_FEATURES = 7
N_OUTPUTS = 4 * N_OUT  # rho, u, By, p


def sample_params(rng: np.random.Generator) -> np.ndarray:
    """Draw a single 7-vector of Brio-Wu-ish parameters."""
    return np.array([
        rng.uniform(0.8, 1.2),    # rho_L
        rng.uniform(0.8, 1.2),    # p_L
        rng.uniform(0.8, 1.2),    # By_L
        rng.uniform(0.10, 0.15),  # rho_R
        rng.uniform(0.08, 0.12),  # p_R
        rng.uniform(-1.2, -0.8),  # By_R
        rng.uniform(0.60, 0.90),  # Bx
    ])


def run_one(params: np.ndarray) -> np.ndarray:
    """Run HLL MHD on a single parameter vector; return (4, N_OUT) primitives at t=T_FINAL."""
    rho_L, p_L, By_L, rho_R, p_R, By_R, Bx = params
    dx = 1.0 / N_SIM
    x = (np.arange(N_SIM) + 0.5) * dx
    left = x < 0.5
    rho = np.where(left, rho_L, rho_R)
    p = np.where(left, p_L, p_R)
    By = np.where(left, By_L, By_R)
    zero = np.zeros(N_SIM)
    U0 = mhd_primitive_to_conservative(rho, zero, zero, zero, By, zero, p, GAMMA, Bx)
    U = mhd_simulate(U0, dx, t_final=T_FINAL, gamma=GAMMA, Bx=Bx)
    rho_f, u_f, _, _, By_f, _, p_f = mhd_conservative_to_primitive(U, GAMMA, Bx)
    # Downsample N_SIM -> N_OUT by block-mean.
    ratio = N_SIM // N_OUT
    def bin_mean(a):
        return a.reshape(N_OUT, ratio).mean(axis=1)
    return np.stack([bin_mean(rho_f), bin_mean(u_f), bin_mean(By_f), bin_mean(p_f)])


def build_dataset(n_examples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_examples, N_FEATURES), dtype=np.float32)
    Y = np.zeros((n_examples, N_OUTPUTS), dtype=np.float32)
    for i in range(n_examples):
        p = sample_params(rng)
        out = run_one(p)  # (4, N_OUT)
        X[i] = p.astype(np.float32)
        Y[i] = out.reshape(-1).astype(np.float32)
    return X, Y


if __name__ == "__main__":
    X, Y = build_dataset(8, seed=0)
    print("X shape:", X.shape, "Y shape:", Y.shape)
    print("X[0]:", X[0])
    print("Y[0, :8]:", Y[0, :8])
