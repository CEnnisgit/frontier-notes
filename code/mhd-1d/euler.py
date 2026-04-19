"""1D compressible Euler equations with HLL flux.

State vector (conservative): U = [rho, rho*u, E]^T
where E = p/(gamma-1) + 0.5*rho*u^2 is total energy density.

Flux: F(U) = [rho*u, rho*u^2 + p, (E + p)*u]^T

Numerical scheme: HLL approximate Riemann flux at cell interfaces,
forward Euler in time, transmissive boundary conditions.

HLL is positivity-preserving for ideal gases and handles both shocks and
rarefactions, but smears contact discontinuities (that's HLLC's job — we
stick with HLL here for simplicity and because it extends to MHD cleanly).
"""

from __future__ import annotations

import numpy as np

GAMMA_DEFAULT = 1.4


def primitive_to_conservative(rho: np.ndarray, u: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    """Stack (rho, u, p) -> U = (rho, rho*u, E). Inputs may be scalars or 1D arrays."""
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, E])


def conservative_to_primitive(U: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U shape (3,) or (3, N). Returns (rho, u, p) with matching trailing shape."""
    rho = U[0]
    u = U[1] / rho
    E = U[2]
    p = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    return rho, u, p


def euler_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """F(U). Shape matches U."""
    rho, u, p = conservative_to_primitive(U, gamma)
    F = np.empty_like(U)
    F[0] = rho * u
    F[1] = rho * u * u + p
    F[2] = (U[2] + p) * u
    return F


def hll_flux(U_L: np.ndarray, U_R: np.ndarray, gamma: float) -> np.ndarray:
    """HLL numerical flux at interfaces between cells L and R.

    Wave speed estimates S_L, S_R use the min/max of the two states' (u +/- a)
    — the two-wave (Davis 1988) bound. More sophisticated estimates exist;
    this one is robust and enough for Sod.
    """
    rho_l, u_l, p_l = conservative_to_primitive(U_L, gamma)
    rho_r, u_r, p_r = conservative_to_primitive(U_R, gamma)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    s_l = np.minimum(u_l - a_l, u_r - a_r)
    s_r = np.maximum(u_l + a_l, u_r + a_r)

    F_L = euler_flux(U_L, gamma)
    F_R = euler_flux(U_R, gamma)

    flux = np.empty_like(U_L)
    # Broadcasting: s_l, s_r have shape (N,); U, F have shape (3, N)
    mask_left = s_l >= 0.0
    mask_right = s_r <= 0.0
    mask_mid = ~mask_left & ~mask_right

    flux[:, mask_left] = F_L[:, mask_left]
    flux[:, mask_right] = F_R[:, mask_right]

    if np.any(mask_mid):
        sl = s_l[mask_mid]
        sr = s_r[mask_mid]
        flux[:, mask_mid] = (
            sr * F_L[:, mask_mid] - sl * F_R[:, mask_mid]
            + sl * sr * (U_R[:, mask_mid] - U_L[:, mask_mid])
        ) / (sr - sl)
    return flux


def step_forward_euler(U: np.ndarray, gamma: float, dx: float, dt: float) -> np.ndarray:
    """One FE step with HLL fluxes and transmissive (zero-gradient) BCs."""
    N = U.shape[1]
    # N+1 interfaces; index k means between cell k-1 and cell k (after ghost padding)
    U_L = np.empty((3, N + 1))
    U_R = np.empty((3, N + 1))
    U_L[:, 1:-1] = U[:, :-1]
    U_R[:, 1:-1] = U[:, 1:]
    # Transmissive: ghost cell equals boundary cell
    U_L[:, 0] = U[:, 0]
    U_R[:, 0] = U[:, 0]
    U_L[:, -1] = U[:, -1]
    U_R[:, -1] = U[:, -1]

    F = hll_flux(U_L, U_R, gamma)
    return U - (dt / dx) * (F[:, 1:] - F[:, :-1])


def max_wave_speed(U: np.ndarray, gamma: float) -> float:
    rho, u, p = conservative_to_primitive(U, gamma)
    a = np.sqrt(gamma * p / rho)
    return float(np.max(np.abs(u) + a))


def simulate(
    U0: np.ndarray,
    dx: float,
    t_final: float,
    gamma: float = GAMMA_DEFAULT,
    cfl_target: float = 0.4,
) -> np.ndarray:
    """Integrate from U0 to t_final with adaptive dt (CFL-constrained)."""
    U = U0.copy()
    t = 0.0
    while t < t_final - 1e-14:
        dt = cfl_target * dx / max_wave_speed(U, gamma)
        if t + dt > t_final:
            dt = t_final - t
        U = step_forward_euler(U, gamma, dx, dt)
        t += dt
    return U


def sod_initial(N: int, domain: tuple[float, float] = (0.0, 1.0), gamma: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """Sod shock tube initial condition.

    Left state  (x < 0.5): rho=1.0,  u=0,  p=1.0
    Right state (x > 0.5): rho=0.125, u=0, p=0.1
    gamma = 1.4, t_final = 0.2 is the standard test.
    """
    dx = (domain[1] - domain[0]) / N
    x = domain[0] + (np.arange(N) + 0.5) * dx
    left = x < 0.5
    rho = np.where(left, 1.0, 0.125)
    u = np.zeros(N)
    p = np.where(left, 1.0, 0.1)
    U = primitive_to_conservative(rho, u, p, gamma)
    return x, U


if __name__ == "__main__":
    N = 400
    x, U0 = sod_initial(N)
    dx = x[1] - x[0]
    U = simulate(U0, dx, t_final=0.2, gamma=1.4)

    rho, u, p = conservative_to_primitive(U, 1.4)

    # Compare to exact solution
    from riemann_euler import sample_grid

    rho_ex, u_ex, p_ex = sample_grid(
        x, t=0.2, x0=0.5,
        rho_l=1.0, u_l=0.0, p_l=1.0,
        rho_r=0.125, u_r=0.0, p_r=0.1,
        gamma=1.4,
    )

    l1_rho = float(np.mean(np.abs(rho - rho_ex))) / float(np.mean(np.abs(rho_ex)))
    l1_u = float(np.mean(np.abs(u - u_ex)))  # absolute (u_ex has zero regions)
    l1_p = float(np.mean(np.abs(p - p_ex))) / float(np.mean(np.abs(p_ex)))

    print(f"Sod shock tube, N={N}, t=0.2")
    print(f"  relative L1 density:  {l1_rho*100:.3f}%")
    print(f"  absolute L1 velocity: {l1_u:.5f}")
    print(f"  relative L1 pressure: {l1_p*100:.3f}%")
