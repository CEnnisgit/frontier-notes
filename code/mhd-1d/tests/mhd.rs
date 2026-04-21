//! Verification tests for the 1D ideal MHD HLL solver.

use mhd_1d::euler::{euler_flux, primitive_to_conservative as euler_p2c};
use mhd_1d::mhd::{
    brio_wu_initial, conservative_to_primitive, fast_magnetosonic_speed, mhd_flux, mhd_hll_flux,
    mhd_simulate, primitive_to_conservative, GAMMA_BRIO_WU,
};
use ndarray::{Array1, arr1};

#[test]
fn primitive_conservative_roundtrip() {
    let rho = arr1(&[1.0, 0.5]);
    let u = arr1(&[0.3, -0.2]);
    let v = arr1(&[0.1, 0.0]);
    let w = arr1(&[0.0, 0.4]);
    let by = arr1(&[1.0, -1.0]);
    let bz = arr1(&[0.0, 0.2]);
    let p = arr1(&[1.0, 0.3]);
    let u_state = primitive_to_conservative(&rho, &u, &v, &w, &by, &bz, &p, 2.0, 0.75);
    let s = conservative_to_primitive(&u_state, 2.0, 0.75);
    let atol = 1e-12;
    for i in 0..rho.len() {
        assert!((s.rho[i] - rho[i]).abs() < atol);
        assert!((s.u[i] - u[i]).abs() < atol);
        assert!((s.v[i] - v[i]).abs() < atol);
        assert!((s.w[i] - w[i]).abs() < atol);
        assert!((s.by[i] - by[i]).abs() < atol);
        assert!((s.bz[i] - bz[i]).abs() < atol);
        assert!((s.p[i] - p[i]).abs() < atol);
    }
}

#[test]
fn fast_speed_exceeds_sound_and_alfven_speeds() {
    // c_f >= max(a, c_A_x) for any state; defining property of fast MHD waves.
    let rho = arr1(&[1.0, 0.5, 2.0]);
    let p = arr1(&[1.0, 0.5, 3.0]);
    let by = arr1(&[0.5, 1.0, 0.0]);
    let bz = arr1(&[0.0, 0.3, 1.0]);
    let bx = 0.75_f64;
    let gamma = 2.0_f64;
    let cf = fast_magnetosonic_speed(&rho, &p, &by, &bz, gamma, bx);
    for i in 0..rho.len() {
        let a = (gamma * p[i] / rho[i]).sqrt();
        let c_ax = (bx * bx / rho[i]).sqrt();
        assert!(cf[i] >= a - 1e-12);
        assert!(cf[i] >= c_ax - 1e-12);
    }
}

#[test]
fn fast_speed_brio_wu_left_state() {
    // Analytical value at Brio-Wu left IC: c_f ~ 1.7923.
    let cf = fast_magnetosonic_speed(
        &arr1(&[1.0_f64]),
        &arr1(&[1.0_f64]),
        &arr1(&[1.0_f64]),
        &arr1(&[0.0_f64]),
        2.0,
        0.75,
    );
    assert!((cf[0] - 1.7923).abs() < 1e-3, "c_f = {}", cf[0]);
}

#[test]
fn hll_reduces_to_physical_flux_on_identical_states() {
    let u_state = primitive_to_conservative(
        &arr1(&[1.0, 0.7]),
        &arr1(&[0.3, -0.1]),
        &arr1(&[0.0, 0.2]),
        &arr1(&[0.0, 0.0]),
        &arr1(&[0.5, -0.3]),
        &arr1(&[0.0, 0.1]),
        &arr1(&[1.0, 0.5]),
        2.0,
        0.75,
    );
    let f_expected = mhd_flux(&u_state, 2.0, 0.75);
    let f_hll = mhd_hll_flux(&u_state, &u_state, 2.0, 0.75);
    for k in 0..7 {
        for i in 0..2 {
            let diff = (f_hll[[k, i]] - f_expected[[k, i]]).abs();
            assert!(diff < 1e-12, "HLL != F at [{k},{i}]: diff={diff:e}");
        }
    }
}

#[test]
fn mhd_flux_with_zero_b_matches_embedded_euler() {
    // With B=0, rows [0,1,6] of MHD flux equal Euler flux rows [0,1,2];
    // rows [2,3,4,5] are transport of v, w, 0, 0 respectively (all zero here).
    let rho = arr1(&[1.0, 0.5]);
    let u = arr1(&[0.3, -0.2]);
    let p = arr1(&[1.0, 0.3]);
    let zero = Array1::<f64>::zeros(rho.len());

    let u_mhd = primitive_to_conservative(
        &rho, &u, &zero, &zero, &zero, &zero, &p, 1.4, 0.0,
    );
    let f_mhd = mhd_flux(&u_mhd, 1.4, 0.0);

    let u_eul = euler_p2c(&rho, &u, &p, 1.4);
    let f_eul = euler_flux(&u_eul, 1.4);

    for i in 0..rho.len() {
        assert!((f_mhd[[0, i]] - f_eul[[0, i]]).abs() < 1e-12);
        assert!((f_mhd[[1, i]] - f_eul[[1, i]]).abs() < 1e-12);
        assert!((f_mhd[[6, i]] - f_eul[[2, i]]).abs() < 1e-12);
        for k in 2..6 {
            assert!(f_mhd[[k, i]].abs() < 1e-12, "MHD[{k},{i}] = {}", f_mhd[[k, i]]);
        }
    }
}

#[test]
fn brio_wu_stays_positive() {
    let n = 400;
    let (x, u0, bx) = brio_wu_initial(n, (0.0, 1.0));
    let dx = x[1] - x[0];
    let u = mhd_simulate(&u0, dx, 0.1, GAMMA_BRIO_WU, bx, 0.4);
    let s = conservative_to_primitive(&u, GAMMA_BRIO_WU, bx);
    for i in 0..s.rho.len() {
        assert!(s.rho[i] > 0.0, "rho[{i}] = {}", s.rho[i]);
        assert!(s.p[i] > 0.0, "p[{i}] = {}", s.p[i]);
    }
}

#[test]
fn brio_wu_conserves_mass() {
    // Fast wave from right state at ~3.68 × 0.1 = 0.37 < 0.5 doesn't reach
    // the boundary by t=0.1, so mass at the transmissive boundary is zero.
    let n = 400;
    let (x, u0, bx) = brio_wu_initial(n, (0.0, 1.0));
    let dx = x[1] - x[0];
    let u = mhd_simulate(&u0, dx, 0.1, GAMMA_BRIO_WU, bx, 0.4);
    let mass_0: f64 = (0..n).map(|i| u0[[0, i]]).sum::<f64>() * dx;
    let mass_f: f64 = (0..n).map(|i| u[[0, i]]).sum::<f64>() * dx;
    let drift = ((mass_f - mass_0) / mass_0).abs();
    assert!(drift < 1e-12, "mass drift {drift:e} exceeds 1e-12");
}

#[test]
fn brio_wu_density_at_center_in_plateau() {
    // Published refs (Brio-Wu 1988 Fig 2; Toth 2000; Athena tests) give
    // rho(x=0.5, t=0.1) ~ 0.67-0.70. HLL with no reconstruction is more
    // diffusive than MUSCL, so we accept [0.55, 0.80].
    let n = 800;
    let (x, u0, bx) = brio_wu_initial(n, (0.0, 1.0));
    let dx = x[1] - x[0];
    let u = mhd_simulate(&u0, dx, 0.1, GAMMA_BRIO_WU, bx, 0.4);
    let s = conservative_to_primitive(&u, GAMMA_BRIO_WU, bx);
    let rho_c = s.rho[n / 2];
    assert!(
        (0.55..=0.80).contains(&rho_c),
        "rho(0.5) = {rho_c:.4} outside [0.55, 0.80]"
    );
}

#[test]
fn brio_wu_by_flips_sign() {
    // Initial: +1 left, -1 right. The Alfvén wave rotates B_y through 0;
    // in the evolved profile we expect both positive and negative values
    // and at least one sign change.
    let n = 400;
    let (x, u0, bx) = brio_wu_initial(n, (0.0, 1.0));
    let dx = x[1] - x[0];
    let u = mhd_simulate(&u0, dx, 0.1, GAMMA_BRIO_WU, bx, 0.4);
    let s = conservative_to_primitive(&u, GAMMA_BRIO_WU, bx);
    let by_max = s.by.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let by_min = s.by.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(by_max > 0.1, "B_y max too small: {by_max:.4}");
    assert!(by_min < -0.1, "B_y min too big: {by_min:.4}");
    let mut sign_changes = 0;
    for i in 1..s.by.len() {
        if s.by[i - 1].signum() != s.by[i].signum() {
            sign_changes += 1;
        }
    }
    assert!(sign_changes >= 1, "B_y has no sign change");
}
