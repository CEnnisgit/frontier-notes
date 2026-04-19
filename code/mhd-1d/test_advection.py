"""Verification tests for the 1D upwind advection solver."""

from __future__ import annotations

import numpy as np
import pytest

from advection import advect_upwind, convergence_study, run_gaussian_test


def test_gaussian_periodic_returns_near_origin():
    """After one full period, the pulse should be close to the initial state.

    Upwind is diffusive; the pulse broadens and the peak drops. We bound the
    L2 error rather than requiring exact return. Empirical value at N=400,
    sigma=0.1, CFL=0.5 is ~0.04; threshold 0.10 leaves slack without hiding bugs.
    """
    _, _, _, err = run_gaussian_test(N=400)
    assert err is not None
    assert err < 0.10, f"L2 error {err:.4f} exceeds tolerance 0.10"


def test_upwind_is_first_order():
    """Upwind + forward Euler should converge at roughly O(dx^1).

    Empirical mean order for the Gaussian test is ~0.95–1.0. Accept [0.80, 1.20]
    to leave room for discretization artifacts at small N without hiding a
    genuine scheme bug (which would push the order to 0 or >2).
    """
    _, _, orders = convergence_study(Ns=(100, 200, 400, 800))
    mean_order = float(np.mean(orders))
    assert 0.80 <= mean_order <= 1.20, f"Mean order {mean_order:.3f} outside [0.80, 1.20]"


def test_zero_velocity_is_exact_identity():
    """With c = 0 the update reduces to u^{n+1} = u^n exactly."""
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal(64)
    u = advect_upwind(u0, c=0.0, dx=0.1, dt=0.01, n_steps=50)
    np.testing.assert_array_equal(u, u0)


def test_cfl_violation_raises():
    """The solver should refuse to run past the stability limit."""
    u0 = np.ones(32)
    with pytest.raises(ValueError, match="CFL"):
        advect_upwind(u0, c=1.0, dx=0.01, dt=0.02, n_steps=1)  # CFL = 2.0


def test_negative_velocity_also_converges():
    """Upwind must flip stencil direction when c < 0; same convergence order."""
    _, _, uf_pos, _ = run_gaussian_test(N=400, c=1.0)
    _, _, uf_neg, _ = run_gaussian_test(N=400, c=-1.0)
    # Same initial Gaussian, opposite velocity, same |c| and same period ⇒ same final state.
    np.testing.assert_allclose(uf_pos, uf_neg, rtol=1e-10, atol=1e-12)
