"""Verification tests for the 1D ideal MHD HLL solver."""

from __future__ import annotations

import numpy as np
import pytest

from euler import euler_flux, primitive_to_conservative
from mhd import (
    BX_BRIO_WU,
    GAMMA_BRIO_WU,
    brio_wu_initial,
    fast_magnetosonic_speed,
    mhd_conservative_to_primitive,
    mhd_flux,
    mhd_hll_flux,
    mhd_primitive_to_conservative,
    mhd_simulate,
)


def test_mhd_primitive_conservative_roundtrip():
    rho = np.array([1.0, 0.5])
    u = np.array([0.3, -0.2])
    v = np.array([0.1, 0.0])
    w = np.array([0.0, 0.4])
    By = np.array([1.0, -1.0])
    Bz = np.array([0.0, 0.2])
    p = np.array([1.0, 0.3])
    U = mhd_primitive_to_conservative(rho, u, v, w, By, Bz, p, gamma=2.0, Bx=0.75)
    rho2, u2, v2, w2, By2, Bz2, p2 = mhd_conservative_to_primitive(U, gamma=2.0, Bx=0.75)
    np.testing.assert_allclose(rho2, rho)
    np.testing.assert_allclose(u2, u)
    np.testing.assert_allclose(v2, v)
    np.testing.assert_allclose(w2, w)
    np.testing.assert_allclose(By2, By)
    np.testing.assert_allclose(Bz2, Bz)
    np.testing.assert_allclose(p2, p)


def test_fast_speed_exceeds_sound_and_alfven_speeds():
    """c_f >= max(a, c_A_x) for any state; this is the defining property of fast MHD waves."""
    rho = np.array([1.0, 0.5, 2.0])
    p = np.array([1.0, 0.5, 3.0])
    By = np.array([0.5, 1.0, 0.0])
    Bz = np.array([0.0, 0.3, 1.0])
    Bx = 0.75
    gamma = 2.0
    cf = fast_magnetosonic_speed(rho, p, By, Bz, gamma, Bx)
    a = np.sqrt(gamma * p / rho)
    cA_x = np.sqrt(Bx * Bx / rho)
    assert np.all(cf >= a - 1e-12)
    assert np.all(cf >= cA_x - 1e-12)


def test_fast_speed_brio_wu_left_state():
    """Analytical value at Brio-Wu left IC: c_f = sqrt(3.2122...) ~ 1.7923."""
    cf = fast_magnetosonic_speed(
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([0.0]),
        gamma=2.0, Bx=0.75,
    )
    assert abs(cf[0] - 1.7923) < 1e-3


def test_hll_reduces_to_physical_flux_on_identical_states():
    U = mhd_primitive_to_conservative(
        np.array([1.0, 0.7]), np.array([0.3, -0.1]),
        np.array([0.0, 0.2]), np.array([0.0, 0.0]),
        np.array([0.5, -0.3]), np.array([0.0, 0.1]),
        np.array([1.0, 0.5]),
        gamma=2.0, Bx=0.75,
    )
    F_expected = mhd_flux(U, gamma=2.0, Bx=0.75)
    F_hll = mhd_hll_flux(U, U, gamma=2.0, Bx=0.75)
    np.testing.assert_allclose(F_hll, F_expected, atol=1e-12)


def test_mhd_flux_with_zero_B_matches_embedded_euler():
    """With B=0 the 7-var MHD flux rows [0,1,6] reduce to Euler flux rows [0,1,2];
    rows [2,3,4,5] become transport of v, w, 0, 0 respectively."""
    rho = np.array([1.0, 0.5])
    u = np.array([0.3, -0.2])
    p = np.array([1.0, 0.3])
    zero = np.zeros_like(rho)

    U_mhd = mhd_primitive_to_conservative(rho, u, zero, zero, zero, zero, p, gamma=1.4, Bx=0.0)
    F_mhd = mhd_flux(U_mhd, gamma=1.4, Bx=0.0)

    U_eul = primitive_to_conservative(rho, u, p, gamma=1.4)
    F_eul = euler_flux(U_eul, gamma=1.4)

    np.testing.assert_allclose(F_mhd[0], F_eul[0])
    np.testing.assert_allclose(F_mhd[1], F_eul[1])
    np.testing.assert_allclose(F_mhd[6], F_eul[2])
    # Transverse components are identically zero with v=w=By=Bz=Bx=0
    for k in (2, 3, 4, 5):
        np.testing.assert_allclose(F_mhd[k], 0.0, atol=1e-12)


def test_brio_wu_positivity():
    """rho and p must stay positive through the Brio-Wu simulation."""
    N = 400
    x, U0, Bx = brio_wu_initial(N)
    dx = x[1] - x[0]
    U = mhd_simulate(U0, dx, t_final=0.1, gamma=GAMMA_BRIO_WU, Bx=Bx)
    rho, _, _, _, _, _, p = mhd_conservative_to_primitive(U, GAMMA_BRIO_WU, Bx)
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)


def test_brio_wu_conserves_mass():
    """Sum of rho*dx must be preserved to machine precision.

    Rationale: mass flux at the transmissive boundaries is rho*u, and u=0 at
    the boundary cells because the fast wave has not reached x=0 or x=1 by t=0.1
    (fast speed at right state ~3.68 × 0.1 = 0.37 < 0.5).
    """
    N = 400
    x, U0, Bx = brio_wu_initial(N)
    dx = x[1] - x[0]
    U = mhd_simulate(U0, dx, t_final=0.1, gamma=GAMMA_BRIO_WU, Bx=Bx)
    mass_0 = float(np.sum(U0[0])) * dx
    mass_f = float(np.sum(U[0])) * dx
    assert abs(mass_f - mass_0) / mass_0 < 1e-12


def test_brio_wu_density_at_center_matches_reference():
    """At x=0.5, t=0.1: rho should land in [0.55, 0.80].

    Published solutions (Brio & Wu 1988 Fig 2; Toth 2000; Athena tests) give
    rho(x=0.5, t=0.1) ~ 0.67-0.70. HLL with no reconstruction is more diffusive
    than MUSCL schemes, so we accept a wider band.
    """
    N = 800
    x, U0, Bx = brio_wu_initial(N)
    dx = x[1] - x[0]
    U = mhd_simulate(U0, dx, t_final=0.1, gamma=GAMMA_BRIO_WU, Bx=Bx)
    rho, *_ = mhd_conservative_to_primitive(U, GAMMA_BRIO_WU, Bx)
    idx_center = N // 2
    assert 0.55 < rho[idx_center] < 0.80, f"rho(0.5) = {rho[idx_center]:.4f} outside [0.55, 0.80]"


def test_brio_wu_By_flips_sign():
    """B_y must take both positive and negative values in the evolved solution
    (initial state: +1 on left, -1 on right; Alfvén wave rotates it through 0)."""
    N = 400
    x, U0, Bx = brio_wu_initial(N)
    dx = x[1] - x[0]
    U = mhd_simulate(U0, dx, t_final=0.1, gamma=GAMMA_BRIO_WU, Bx=Bx)
    _, _, _, _, By, _, _ = mhd_conservative_to_primitive(U, GAMMA_BRIO_WU, Bx)
    assert By.max() > 0.1
    assert By.min() < -0.1
    # And there is at least one sign change in the evolved profile
    sign_changes = int(np.sum(np.diff(np.sign(By)) != 0))
    assert sign_changes >= 1
