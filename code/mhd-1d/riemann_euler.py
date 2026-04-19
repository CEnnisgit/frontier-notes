"""Exact Riemann solver for 1D compressible Euler equations.

Implements the two-shock / two-rarefaction Newton iteration for the star-region
pressure, then samples the self-similar solution at any ξ = x/t.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., §4.2–4.5. Equation numbers in comments refer to that text.
"""

from __future__ import annotations

import numpy as np

GAMMA_DEFAULT = 1.4


def _sound_speed(rho: float, p: float, gamma: float) -> float:
    return float(np.sqrt(gamma * p / rho))


def _pressure_function(p: float, rho_k: float, p_k: float, gamma: float) -> tuple[float, float]:
    """Pressure function f_K and its derivative f'_K (Toro eq 4.6 / 4.7)."""
    a_k = _sound_speed(rho_k, p_k, gamma)
    if p > p_k:  # shock
        A_k = 2.0 / ((gamma + 1.0) * rho_k)
        B_k = (gamma - 1.0) / (gamma + 1.0) * p_k
        sq = np.sqrt(A_k / (p + B_k))
        f = (p - p_k) * sq
        fp = sq * (1.0 - 0.5 * (p - p_k) / (B_k + p))
    else:  # rarefaction
        f = (2.0 * a_k / (gamma - 1.0)) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fp = (1.0 / (rho_k * a_k)) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, fp


def _initial_pressure_guess(
    rho_l: float, u_l: float, p_l: float,
    rho_r: float, u_r: float, p_r: float,
    gamma: float,
) -> float:
    """Primitive-variable (PVRS) initial guess, Toro eq 4.46, clamped positive."""
    a_l = _sound_speed(rho_l, p_l, gamma)
    a_r = _sound_speed(rho_r, p_r, gamma)
    p_pv = 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r)
    return max(1e-8, p_pv)


def solve_star_state(
    rho_l: float, u_l: float, p_l: float,
    rho_r: float, u_r: float, p_r: float,
    gamma: float = GAMMA_DEFAULT,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Iteratively solve f_L(p) + f_R(p) + Δu = 0 for p_star; return (p_star, u_star)."""
    du = u_r - u_l
    p = _initial_pressure_guess(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)
    for _ in range(max_iter):
        f_l, fp_l = _pressure_function(p, rho_l, p_l, gamma)
        f_r, fp_r = _pressure_function(p, rho_r, p_r, gamma)
        f = f_l + f_r + du
        fp = fp_l + fp_r
        p_new = p - f / fp
        if p_new < 0.0:
            p_new = 0.5 * p
        if abs(p_new - p) / (0.5 * (p + p_new)) < tol:
            p = p_new
            break
        p = p_new
    else:
        raise RuntimeError(f"Riemann star-state iteration did not converge in {max_iter} steps")

    f_l, _ = _pressure_function(p, rho_l, p_l, gamma)
    f_r, _ = _pressure_function(p, rho_r, p_r, gamma)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    return p, u_star


def sample(
    xi: float,
    rho_l: float, u_l: float, p_l: float,
    rho_r: float, u_r: float, p_r: float,
    gamma: float = GAMMA_DEFAULT,
    p_star: float | None = None,
    u_star: float | None = None,
) -> tuple[float, float, float]:
    """Exact (rho, u, p) at self-similar coordinate ξ = (x - x0)/t.

    If p_star, u_star are passed (from a prior solve_star_state call), skip the
    Newton iteration — useful when sampling many points at the same t.
    """
    if p_star is None or u_star is None:
        p_star, u_star = solve_star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)

    a_l = _sound_speed(rho_l, p_l, gamma)
    a_r = _sound_speed(rho_r, p_r, gamma)

    if xi <= u_star:
        # Left of contact
        if p_star > p_l:
            s_shock = u_l - a_l * np.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * p_star / p_l + (gamma - 1.0) / (2.0 * gamma)
            )
            if xi < s_shock:
                return rho_l, u_l, p_l
            rho_star_l = rho_l * (
                (p_star / p_l + (gamma - 1.0) / (gamma + 1.0))
                / ((gamma - 1.0) / (gamma + 1.0) * p_star / p_l + 1.0)
            )
            return rho_star_l, u_star, p_star
        else:
            s_head = u_l - a_l
            a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
            s_tail = u_star - a_star_l
            if xi < s_head:
                return rho_l, u_l, p_l
            if xi > s_tail:
                rho_star_l = rho_l * (p_star / p_l) ** (1.0 / gamma)
                return rho_star_l, u_star, p_star
            # Inside left rarefaction fan
            c = 2.0 / (gamma + 1.0) + (gamma - 1.0) / ((gamma + 1.0) * a_l) * (u_l - xi)
            rho = rho_l * c ** (2.0 / (gamma - 1.0))
            u = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * u_l + xi)
            p = p_l * c ** (2.0 * gamma / (gamma - 1.0))
            return rho, u, p
    else:
        # Right of contact — mirror
        if p_star > p_r:
            s_shock = u_r + a_r * np.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * p_star / p_r + (gamma - 1.0) / (2.0 * gamma)
            )
            if xi > s_shock:
                return rho_r, u_r, p_r
            rho_star_r = rho_r * (
                (p_star / p_r + (gamma - 1.0) / (gamma + 1.0))
                / ((gamma - 1.0) / (gamma + 1.0) * p_star / p_r + 1.0)
            )
            return rho_star_r, u_star, p_star
        else:
            s_head = u_r + a_r
            a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
            s_tail = u_star + a_star_r
            if xi > s_head:
                return rho_r, u_r, p_r
            if xi < s_tail:
                rho_star_r = rho_r * (p_star / p_r) ** (1.0 / gamma)
                return rho_star_r, u_star, p_star
            c = 2.0 / (gamma + 1.0) - (gamma - 1.0) / ((gamma + 1.0) * a_r) * (u_r - xi)
            rho = rho_r * c ** (2.0 / (gamma - 1.0))
            u = 2.0 / (gamma + 1.0) * (-a_r + (gamma - 1.0) / 2.0 * u_r + xi)
            p = p_r * c ** (2.0 * gamma / (gamma - 1.0))
            return rho, u, p


def sample_grid(
    x: np.ndarray, t: float, x0: float,
    rho_l: float, u_l: float, p_l: float,
    rho_r: float, u_r: float, p_r: float,
    gamma: float = GAMMA_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the exact solution on a grid of cell centers x at time t."""
    p_star, u_star = solve_star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)
    for i, xi_val in enumerate(x):
        xi = (xi_val - x0) / t
        rho[i], u[i], p[i] = sample(
            xi, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma,
            p_star=p_star, u_star=u_star,
        )
    return rho, u, p


if __name__ == "__main__":
    # Sod test: solve star state and print summary
    p_star, u_star = solve_star_state(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gamma=1.4)
    print(f"Sod star state: p* = {p_star:.6f}, u* = {u_star:.6f}")
    print("  Reference (Toro Table 4.1): p* ~ 0.30313, u* ~ 0.92745")
