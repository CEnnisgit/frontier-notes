"""1D ideal magnetohydrodynamics (MHD) with HLL flux.

State vector (7 variables; B_x is a constant parameter, since div-B = 0 in 1D):
    U = [rho, rho*u, rho*v, rho*w, B_y, B_z, E]^T

Units are natural (mu_0 = 1). Total energy:
    E = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2

Flux (for the x-direction, with B_x constant):
    F[0] = rho*u
    F[1] = rho*u^2 + p_total - B_x^2
    F[2] = rho*u*v - B_x*B_y
    F[3] = rho*u*w - B_x*B_z
    F[4] = u*B_y - v*B_x
    F[5] = u*B_z - w*B_x
    F[6] = (E + p_total)*u - B_x*(u*B_x + v*B_y + w*B_z)
with p_total = p + 0.5*|B|^2.

Canonical 1D test: Brio & Wu, JCP 75 (1988) 400. Reproduced here at t=0.1
using gamma = 2.0 and B_x = 0.75.
"""

from __future__ import annotations

import numpy as np

GAMMA_BRIO_WU = 2.0
BX_BRIO_WU = 0.75


def mhd_primitive_to_conservative(
    rho: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
    By: np.ndarray, Bz: np.ndarray, p: np.ndarray,
    gamma: float, Bx: float,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    By = np.asarray(By, dtype=float)
    Bz = np.asarray(Bz, dtype=float)
    p = np.asarray(p, dtype=float)
    E = (
        p / (gamma - 1.0)
        + 0.5 * rho * (u * u + v * v + w * w)
        + 0.5 * (Bx * Bx + By * By + Bz * Bz)
    )
    return np.stack([rho, rho * u, rho * v, rho * w, By, Bz, E])


def mhd_conservative_to_primitive(U: np.ndarray, gamma: float, Bx: float) -> tuple:
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    w = U[3] / rho
    By = U[4]
    Bz = U[5]
    E = U[6]
    p = (gamma - 1.0) * (
        E - 0.5 * rho * (u * u + v * v + w * w) - 0.5 * (Bx * Bx + By * By + Bz * Bz)
    )
    return rho, u, v, w, By, Bz, p


def mhd_flux(U: np.ndarray, gamma: float, Bx: float) -> np.ndarray:
    rho, u, v, w, By, Bz, p = mhd_conservative_to_primitive(U, gamma, Bx)
    p_total = p + 0.5 * (Bx * Bx + By * By + Bz * Bz)
    E = U[6]
    F = np.empty_like(U)
    F[0] = rho * u
    F[1] = rho * u * u + p_total - Bx * Bx
    F[2] = rho * u * v - Bx * By
    F[3] = rho * u * w - Bx * Bz
    F[4] = u * By - v * Bx
    F[5] = u * Bz - w * Bx
    F[6] = (E + p_total) * u - Bx * (u * Bx + v * By + w * Bz)
    return F


def fast_magnetosonic_speed(
    rho: np.ndarray, p: np.ndarray, By: np.ndarray, Bz: np.ndarray,
    gamma: float, Bx: float,
) -> np.ndarray:
    """c_f such that c_f^2 = 0.5(a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4*a^2*b_x^2))."""
    a2 = gamma * p / rho
    b2 = (Bx * Bx + By * By + Bz * Bz) / rho
    bx2 = Bx * Bx / rho
    disc = (a2 + b2) ** 2 - 4.0 * a2 * bx2
    disc = np.maximum(disc, 0.0)
    return np.sqrt(0.5 * (a2 + b2 + np.sqrt(disc)))


def mhd_hll_flux(U_L: np.ndarray, U_R: np.ndarray, gamma: float, Bx: float) -> np.ndarray:
    rho_l, u_l, v_l, w_l, By_l, Bz_l, p_l = mhd_conservative_to_primitive(U_L, gamma, Bx)
    rho_r, u_r, v_r, w_r, By_r, Bz_r, p_r = mhd_conservative_to_primitive(U_R, gamma, Bx)
    cf_l = fast_magnetosonic_speed(rho_l, p_l, By_l, Bz_l, gamma, Bx)
    cf_r = fast_magnetosonic_speed(rho_r, p_r, By_r, Bz_r, gamma, Bx)
    s_l = np.minimum(u_l - cf_l, u_r - cf_r)
    s_r = np.maximum(u_l + cf_l, u_r + cf_r)

    F_L = mhd_flux(U_L, gamma, Bx)
    F_R = mhd_flux(U_R, gamma, Bx)

    flux = np.empty_like(U_L)
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


def mhd_step(U: np.ndarray, gamma: float, Bx: float, dx: float, dt: float) -> np.ndarray:
    N = U.shape[1]
    U_L = np.empty((7, N + 1))
    U_R = np.empty((7, N + 1))
    U_L[:, 1:-1] = U[:, :-1]
    U_R[:, 1:-1] = U[:, 1:]
    U_L[:, 0] = U[:, 0]
    U_R[:, 0] = U[:, 0]
    U_L[:, -1] = U[:, -1]
    U_R[:, -1] = U[:, -1]
    F = mhd_hll_flux(U_L, U_R, gamma, Bx)
    return U - (dt / dx) * (F[:, 1:] - F[:, :-1])


def mhd_max_wave_speed(U: np.ndarray, gamma: float, Bx: float) -> float:
    rho, u, v, w, By, Bz, p = mhd_conservative_to_primitive(U, gamma, Bx)
    cf = fast_magnetosonic_speed(rho, p, By, Bz, gamma, Bx)
    return float(np.max(np.abs(u) + cf))


def mhd_simulate(
    U0: np.ndarray, dx: float, t_final: float,
    gamma: float, Bx: float, cfl_target: float = 0.4,
) -> np.ndarray:
    U = U0.copy()
    t = 0.0
    while t < t_final - 1e-14:
        dt = cfl_target * dx / mhd_max_wave_speed(U, gamma, Bx)
        if t + dt > t_final:
            dt = t_final - t
        U = mhd_step(U, gamma, Bx, dx, dt)
        t += dt
    return U


def brio_wu_initial(N: int, domain: tuple[float, float] = (0.0, 1.0)) -> tuple[np.ndarray, np.ndarray, float]:
    """Brio & Wu 1988 shock tube ICs.

    Left  (x < 0.5): rho=1.0,   p=1.0, u=v=w=0, B_y=+1, B_z=0
    Right (x > 0.5): rho=0.125, p=0.1, u=v=w=0, B_y=-1, B_z=0
    B_x = 0.75 everywhere; gamma = 2.0; reference t = 0.1.
    """
    dx = (domain[1] - domain[0]) / N
    x = domain[0] + (np.arange(N) + 0.5) * dx
    left = x < 0.5
    rho = np.where(left, 1.0, 0.125)
    p = np.where(left, 1.0, 0.1)
    u = np.zeros(N)
    v = np.zeros(N)
    w = np.zeros(N)
    By = np.where(left, 1.0, -1.0)
    Bz = np.zeros(N)
    U = mhd_primitive_to_conservative(rho, u, v, w, By, Bz, p, GAMMA_BRIO_WU, BX_BRIO_WU)
    return x, U, BX_BRIO_WU


if __name__ == "__main__":
    N = 800
    x, U0, Bx = brio_wu_initial(N)
    dx = x[1] - x[0]
    U = mhd_simulate(U0, dx, t_final=0.1, gamma=GAMMA_BRIO_WU, Bx=Bx)
    rho, u, v, w, By, Bz, p = mhd_conservative_to_primitive(U, GAMMA_BRIO_WU, Bx)

    print(f"Brio-Wu, N={N}, t=0.1:")
    print(f"  min/max rho:  {rho.min():.4f} / {rho.max():.4f}")
    print(f"  min/max p:    {p.min():.4f} / {p.max():.4f}")
    print(f"  B_y spans:    {By.min():.4f} to {By.max():.4f}")

    # Qualitative landmarks from published references (Brio & Wu 1988, Fig. 2;
    # also Toth 2000, Stone+ Athena paper). Values are approximate — HLL with
    # no limiter is more diffusive than published results with MUSCL, so we
    # check wide tolerances.
    # At x=0.5 (center), rho is in the central plateau region ~[0.55, 0.8]
    # B_y flips sign somewhere near x=0.5
    idx_center = N // 2
    print(f"  rho at center (x=0.5):  {rho[idx_center]:.4f}")
    print(f"  B_y at center (x=0.5):  {By[idx_center]:.4f}")

    # Conservation check
    mass_0 = float(np.sum(U0[0]))
    mass_f = float(np.sum(U[0]))
    mom_0 = float(np.sum(U0[1]))
    mom_f = float(np.sum(U[1]))
    print(f"  mass drift:     {(mass_f - mass_0) / mass_0:.2e}")
    print(f"  momentum drift: {(mom_f - mom_0):.2e} (initial momentum = {mom_0:.3e})")
