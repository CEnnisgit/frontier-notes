"""Generate the Brio-Wu comparison plot at code/mhd-1d/figures/brio-wu.png.

Plots rho, u, B_y, p across x at t=0.1 for the canonical Brio-Wu shock tube.
No analytical reference solution exists (MHD doesn't have an exact Riemann
solver in closed form); the plot is for visual comparison to published figures.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mhd import (
    BX_BRIO_WU, GAMMA_BRIO_WU,
    brio_wu_initial, mhd_conservative_to_primitive, mhd_simulate,
)


def main():
    N_coarse, N_fine = 400, 1600
    t_final = 0.1

    results = {}
    for N in (N_coarse, N_fine):
        x, U0, Bx = brio_wu_initial(N)
        dx = x[1] - x[0]
        U = mhd_simulate(U0, dx, t_final=t_final, gamma=GAMMA_BRIO_WU, Bx=Bx)
        rho, u, v, w, By, Bz, p = mhd_conservative_to_primitive(U, GAMMA_BRIO_WU, Bx)
        results[N] = (x, rho, u, By, p)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("density", 1), ("velocity u", 2), ("B_y", 3), ("pressure", 4)]
    panel_idx_map = {"density": 1, "velocity u": 2, "B_y": 3, "pressure": 4}

    for (ax, name) in zip(axes.flat, ["density", "velocity u", "B_y", "pressure"]):
        for N, style, label in [(N_fine, "k-", f"N={N_fine}"), (N_coarse, "r.", f"N={N_coarse}")]:
            x, rho, u, By, p = results[N]
            y = {"density": rho, "velocity u": u, "B_y": By, "pressure": p}[name]
            if style.endswith("."):
                ax.plot(x, y, style, ms=3, label=label)
            else:
                ax.plot(x, y, style, lw=1.2, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.set_title(f"Brio-Wu {name}, t={t_final}, gamma=2, B_x=0.75")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "brio-wu.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
