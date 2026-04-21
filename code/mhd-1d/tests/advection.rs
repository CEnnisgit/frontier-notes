//! Verification tests for the 1D upwind advection solver.

use mhd_1d::advection::{
    advect_upwind, convergence_study, run_gaussian_default, run_gaussian_test,
};
use ndarray::Array1;

#[test]
fn gaussian_periodic_returns_near_origin() {
    // Empirical L2 error at N=400, sigma=0.1, CFL=0.5 is ~0.04; threshold 0.10
    // leaves slack without hiding bugs.
    let run = run_gaussian_default(400);
    let err = run.l2_error.expect("default params are one period");
    assert!(err < 0.10, "L2 error {err:.4} exceeds tolerance 0.10");
}

#[test]
fn upwind_is_first_order() {
    // Accept [0.80, 1.20] to leave room for discretization artifacts at small N
    // without hiding a genuine scheme bug (which would push the order to 0 or >2).
    let (_ns, _errs, orders) = convergence_study(&[100, 200, 400, 800]);
    let mean: f64 = orders.iter().sum::<f64>() / orders.len() as f64;
    assert!(
        (0.80..=1.20).contains(&mean),
        "Mean order {mean:.3} outside [0.80, 1.20]"
    );
}

#[test]
fn zero_velocity_is_exact_identity() {
    // With c = 0 the update reduces to u^{n+1} = u^n exactly.
    let u0 = Array1::from_vec((0..64).map(|i| (i as f64 * 0.37).sin()).collect());
    let u = advect_upwind(&u0, 0.0, 0.1, 0.01, 50).unwrap();
    for i in 0..u0.len() {
        assert_eq!(u[i], u0[i]);
    }
}

#[test]
fn cfl_violation_is_rejected() {
    // CFL = 2.0 must be refused.
    let u0 = Array1::<f64>::ones(32);
    let err = advect_upwind(&u0, 1.0, 0.01, 0.02, 1).unwrap_err();
    assert!(err.contains("CFL"), "unexpected error message: {err}");
}

#[test]
fn negative_velocity_matches_positive() {
    // Same |c|, same period, opposite direction; final states must agree.
    let run_pos = run_gaussian_test(400, 1.0, 1.0, (0.0, 1.0), 0.1, 0.5, 0.5);
    let run_neg = run_gaussian_test(400, -1.0, 1.0, (0.0, 1.0), 0.1, 0.5, 0.5);
    for i in 0..run_pos.u_final.len() {
        let diff = (run_pos.u_final[i] - run_neg.u_final[i]).abs();
        assert!(diff < 1e-10, "mismatch at i={i}: diff={diff:.2e}");
    }
}
