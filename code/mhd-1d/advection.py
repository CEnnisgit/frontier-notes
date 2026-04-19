"""1D linear advection: du/dt + c du/dx = 0, periodic BC.

Upwind finite-volume with forward Euler. First-order accurate in both time and
space for smooth solutions. Numerical diffusion coefficient for the scheme is
  nu_num = c * dx * (1 - CFL) / 2
(from the modified-equation analysis of first-order upwind).

Verification driver at the bottom runs a Gaussian pulse for one full period on
a periodic domain and prints a convergence table showing O(dx) L2 error scaling.
"""

from __future__ import annotations

import numpy as np


def advect_upwind(u0: np.ndarray, c: float, dx: float, dt: float, n_steps: int) -> np.ndarray:
    """Advance du/dt + c du/dx = 0 by n_steps with upwind FV + forward Euler.

    Periodic boundary conditions. CFL = |c| * dt / dx must be < 1 for stability.
    """
    cfl = c * dt / dx
    if abs(cfl) >= 1.0:
        raise ValueError(f"CFL = {cfl:.4f} must satisfy |CFL| < 1 for stability")

    u = u0.copy()
    for _ in range(n_steps):
        if c >= 0.0:
            flux = c * u
        else:
            flux = c * np.roll(u, -1)
        u = u - (dt / dx) * (flux - np.roll(flux, 1))
    return u


def run_gaussian_test(
    N: int = 200,
    c: float = 1.0,
    t_final: float = 1.0,
    domain: tuple[float, float] = (0.0, 1.0),
    sigma: float = 0.1,
    x0: float = 0.5,
    cfl_target: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Advect a Gaussian pulse on a periodic domain for exactly t_final.

    Default t_final = 1.0 with c = 1.0 and domain length = 1.0 ⇒ one full period,
    so the analytical final state equals the initial state.

    Returns
    -------
    x          : cell-center coordinates, shape (N,)
    u_initial  : initial pulse
    u_final    : numerical solution at t_final
    l2_error   : L2 error vs u_initial (or None if t_final is not a multiple of the period)
    """
    x_min, x_max = domain
    L = x_max - x_min
    dx = L / N
    x = x_min + (np.arange(N) + 0.5) * dx

    u_initial = np.exp(-((x - x0) / sigma) ** 2)

    dt_cfl = cfl_target * dx / abs(c)
    n_steps = int(np.ceil(t_final / dt_cfl))
    dt = t_final / n_steps  # land exactly on t_final

    u_final = advect_upwind(u_initial, c, dx, dt, n_steps)

    period = L / abs(c)
    n_periods = t_final / period
    if abs(n_periods - round(n_periods)) < 1e-10:
        l2_error = float(np.sqrt(np.mean((u_final - u_initial) ** 2)))
    else:
        l2_error = None

    return x, u_initial, u_final, l2_error


def convergence_study(Ns=(50, 100, 200, 400, 800)) -> tuple[list[int], list[float], list[float]]:
    """Run run_gaussian_test at increasing resolution and estimate the convergence order."""
    errors = []
    for N in Ns:
        _, _, _, err = run_gaussian_test(N=N)
        assert err is not None
        errors.append(err)

    orders = []
    for i in range(1, len(Ns)):
        order = np.log(errors[i - 1] / errors[i]) / np.log(Ns[i] / Ns[i - 1])
        orders.append(order)

    return list(Ns), errors, orders


if __name__ == "__main__":
    x, u0, uf, err = run_gaussian_test(N=400)
    print(f"Single run (N=400): L2 error after one period = {err:.6f}\n")

    Ns, errs, orders = convergence_study()
    print("Convergence study:")
    print(f"  {'N':>5}  {'dx':>10}  {'L2 error':>12}  {'order':>8}")
    for i, (N, e) in enumerate(zip(Ns, errs)):
        order_str = f"{orders[i-1]:.3f}" if i > 0 else "—"
        print(f"  {N:>5d}  {1.0/N:>10.5f}  {e:>12.6f}  {order_str:>8}")

    avg_order = float(np.mean(orders))
    print(f"\nMean observed order: {avg_order:.3f}  (expected ~1.0 for upwind + Euler)")
