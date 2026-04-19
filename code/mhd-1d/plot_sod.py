"""Generate the Sod shock tube comparison plot at code/mhd-1d/figures/sod.png."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from euler import conservative_to_primitive, simulate, sod_initial
from riemann_euler import sample_grid


def main():
    gamma = 1.4
    N = 400
    t_final = 0.2

    x, U0 = sod_initial(N)
    dx = x[1] - x[0]
    U = simulate(U0, dx, t_final=t_final, gamma=gamma)
    rho, u, p = conservative_to_primitive(U, gamma)

    # Exact on a finer grid for the reference curve
    x_fine = np.linspace(0.0, 1.0, 2000)
    rho_ex, u_ex, p_ex = sample_grid(
        x_fine, t=t_final, x0=0.5,
        rho_l=1.0, u_l=0.0, p_l=1.0,
        rho_r=0.125, u_r=0.0, p_r=0.1,
        gamma=gamma,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (num, ex, name) in zip(
        axes,
        [(rho, rho_ex, "density"), (u, u_ex, "velocity"), (p, p_ex, "pressure")],
    ):
        ax.plot(x_fine, ex, "k-", lw=1.5, label="exact")
        ax.plot(x, num, "r.", ms=3.5, label=f"HLL, N={N}")
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.set_title(f"Sod: {name}, t={t_final}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "sod.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
