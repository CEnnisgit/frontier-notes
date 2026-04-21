//! Verification tests for the 1D Euler HLL solver.

use mhd_1d::euler::{
    conservative_to_primitive, euler_flux, hll_flux, primitive_to_conservative, simulate,
    sod_initial,
};
use mhd_1d::riemann_euler::{sample_grid, solve_star_state};
use ndarray::{Array1, arr1};

const GAMMA: f64 = 1.4;

fn assert_allclose(a: &Array1<f64>, b: &Array1<f64>, atol: f64, rtol: f64) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let tol = atol + rtol * b[i].abs();
        assert!(diff <= tol, "mismatch at i={i}: a={}, b={}, diff={diff:e}", a[i], b[i]);
    }
}

#[test]
fn primitive_conservative_roundtrip() {
    let rho = arr1(&[1.0, 0.5, 2.0]);
    let u = arr1(&[0.0, 1.3, -0.7]);
    let p = arr1(&[1.0, 0.4, 3.2]);
    let u_state = primitive_to_conservative(&rho, &u, &p, GAMMA);
    let (rho2, u2, p2) = conservative_to_primitive(&u_state, GAMMA);
    assert_allclose(&rho2, &rho, 1e-12, 1e-12);
    assert_allclose(&u2, &u, 1e-12, 1e-12);
    assert_allclose(&p2, &p, 1e-12, 1e-12);
}

#[test]
fn hll_reduces_to_physical_flux_on_identical_states() {
    let rho = arr1(&[1.0, 2.0]);
    let u = arr1(&[0.3, -0.5]);
    let p = arr1(&[1.0, 0.4]);
    let u_state = primitive_to_conservative(&rho, &u, &p, GAMMA);
    let f_expected = euler_flux(&u_state, GAMMA);
    let f_hll = hll_flux(&u_state, &u_state, GAMMA);
    for k in 0..3 {
        for i in 0..2 {
            let diff = (f_hll[[k, i]] - f_expected[[k, i]]).abs();
            assert!(diff < 1e-12, "HLL != F at [{k},{i}]: diff={diff:e}");
        }
    }
}

#[test]
fn star_state_matches_toro_table_4_1() {
    // Sod case: Toro Table 4.1 gives p* ≈ 0.30313, u* ≈ 0.92745.
    let (p_star, u_star) = solve_star_state(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, GAMMA);
    assert!((p_star - 0.30313).abs() < 1e-4, "p* = {p_star}");
    assert!((u_star - 0.92745).abs() < 1e-4, "u* = {u_star}");
}

#[test]
fn sod_shock_tube_l1_errors_under_tolerance() {
    let n = 400;
    let (x, u0) = sod_initial(n, (0.0, 1.0), GAMMA);
    let dx = x[1] - x[0];
    let u = simulate(&u0, dx, 0.2, GAMMA, 0.4);

    let (rho, _u, p) = conservative_to_primitive(&u, GAMMA);
    let (rho_ex, _u_ex, p_ex) =
        sample_grid(&x, 0.2, 0.5, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, GAMMA);

    let l1_rho_num: f64 = rho.iter().zip(rho_ex.iter()).map(|(a, b)| (a - b).abs()).sum();
    let l1_rho_den: f64 = rho_ex.iter().map(|b| b.abs()).sum();
    let l1_rho = l1_rho_num / l1_rho_den;

    let l1_p_num: f64 = p.iter().zip(p_ex.iter()).map(|(a, b)| (a - b).abs()).sum();
    let l1_p_den: f64 = p_ex.iter().map(|b| b.abs()).sum();
    let l1_p = l1_p_num / l1_p_den;

    assert!(l1_rho < 0.02, "density L1 {:.3}% exceeds 2%", l1_rho * 100.0);
    assert!(l1_p < 0.02, "pressure L1 {:.3}% exceeds 2%", l1_p * 100.0);
}

#[test]
fn uniform_state_is_preserved() {
    let n = 100;
    let rho = Array1::<f64>::from_elem(n, 1.0);
    let u = Array1::<f64>::from_elem(n, 0.3);
    let p = Array1::<f64>::from_elem(n, 1.0);
    let u0 = primitive_to_conservative(&rho, &u, &p, GAMMA);
    let dx = 1.0 / n as f64;
    let u_final = simulate(&u0, dx, 0.1, GAMMA, 0.4);
    for k in 0..3 {
        for i in 0..n {
            let diff = (u_final[[k, i]] - u0[[k, i]]).abs();
            assert!(diff < 1e-12, "uniform state drifted at [{k},{i}]: diff={diff:e}");
        }
    }
}

#[test]
fn sod_stays_positive() {
    let n = 200;
    let (x, u0) = sod_initial(n, (0.0, 1.0), GAMMA);
    let dx = x[1] - x[0];
    let u = simulate(&u0, dx, 0.2, GAMMA, 0.4);
    let (rho, _u, p) = conservative_to_primitive(&u, GAMMA);
    assert!(rho.iter().all(|&r| r > 0.0));
    assert!(p.iter().all(|&pv| pv > 0.0));
}
