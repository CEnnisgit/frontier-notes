"""Verification tests for the 1D Euler HLL solver."""

from __future__ import annotations

import numpy as np
import pytest

from euler import (
    conservative_to_primitive,
    euler_flux,
    hll_flux,
    primitive_to_conservative,
    simulate,
    sod_initial,
)
from riemann_euler import sample_grid, solve_star_state


GAMMA = 1.4


def test_primitive_conservative_roundtrip():
    rho = np.array([1.0, 0.5, 2.0])
    u = np.array([0.0, 1.3, -0.7])
    p = np.array([1.0, 0.4, 3.2])
    U = primitive_to_conservative(rho, u, p, GAMMA)
    rho2, u2, p2 = conservative_to_primitive(U, GAMMA)
    np.testing.assert_allclose(rho2, rho)
    np.testing.assert_allclose(u2, u)
    np.testing.assert_allclose(p2, p)


def test_hll_reduces_to_physical_flux_on_identical_states():
    """HLL(U, U) must equal F(U) regardless of wave speeds."""
    U = primitive_to_conservative(np.array([1.0, 2.0]), np.array([0.3, -0.5]), np.array([1.0, 0.4]), GAMMA)
    F_expected = euler_flux(U, GAMMA)
    F_hll = hll_flux(U, U, GAMMA)
    np.testing.assert_allclose(F_hll, F_expected, atol=1e-12)


def test_star_state_matches_toro_table_4_1():
    """Sod case: p* ≈ 0.30313, u* ≈ 0.92745 (Toro Table 4.1)."""
    p_star, u_star = solve_star_state(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gamma=GAMMA)
    assert abs(p_star - 0.30313) < 1e-4
    assert abs(u_star - 0.92745) < 1e-4


def test_sod_shock_tube_l1_errors_under_tolerance():
    """HLL on N=400 should match exact Sod to <2% relative L1 in density and pressure."""
    N = 400
    x, U0 = sod_initial(N)
    dx = x[1] - x[0]
    U = simulate(U0, dx, t_final=0.2, gamma=GAMMA)

    rho, u, p = conservative_to_primitive(U, GAMMA)
    rho_ex, u_ex, p_ex = sample_grid(
        x, t=0.2, x0=0.5,
        rho_l=1.0, u_l=0.0, p_l=1.0,
        rho_r=0.125, u_r=0.0, p_r=0.1,
        gamma=GAMMA,
    )

    l1_rho = float(np.mean(np.abs(rho - rho_ex))) / float(np.mean(np.abs(rho_ex)))
    l1_p = float(np.mean(np.abs(p - p_ex))) / float(np.mean(np.abs(p_ex)))

    assert l1_rho < 0.02, f"density L1 {l1_rho*100:.2f}% exceeds 2%"
    assert l1_p < 0.02, f"pressure L1 {l1_p*100:.2f}% exceeds 2%"


def test_mass_momentum_energy_conservation_for_periodic_smooth():
    """For a uniform state the solver must preserve it exactly (identity test)."""
    N = 100
    rho = np.ones(N)
    u = np.full(N, 0.3)
    p = np.ones(N)
    U0 = primitive_to_conservative(rho, u, p, GAMMA)
    dx = 1.0 / N
    U = simulate(U0, dx, t_final=0.1, gamma=GAMMA)
    # Transmissive BC on a uniform state: stays uniform (mass/momentum/energy unchanged per cell)
    np.testing.assert_allclose(U, U0, atol=1e-12)


def test_sod_positive_density_and_pressure_throughout():
    """HLL is positivity-preserving; assert this empirically on Sod."""
    N = 200
    x, U0 = sod_initial(N)
    dx = x[1] - x[0]
    U = simulate(U0, dx, t_final=0.2, gamma=GAMMA)
    rho, _, p = conservative_to_primitive(U, GAMMA)
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)
