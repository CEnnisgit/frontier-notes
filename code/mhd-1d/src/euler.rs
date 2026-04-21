//! 1D compressible Euler equations with HLL flux.
//!
//! State vector (conservative): `U = [rho, rho*u, E]^T`
//! where `E = p/(gamma-1) + 0.5*rho*u^2`.
//!
//! Flux: `F(U) = [rho*u, rho*u^2 + p, (E + p)*u]^T`.
//!
//! Numerical scheme: HLL approximate Riemann flux at cell interfaces,
//! forward Euler in time, transmissive boundary conditions. HLL is
//! positivity-preserving for ideal gases. It smears contact discontinuities
//! (HLLC's job); we keep HLL here because it extends to MHD cleanly.

use ndarray::{Array1, Array2};

pub const GAMMA_DEFAULT: f64 = 1.4;

/// Build `U = (rho, rho*u, E)` from primitive arrays. Inputs must have the same length.
pub fn primitive_to_conservative(
    rho: &Array1<f64>,
    u: &Array1<f64>,
    p: &Array1<f64>,
    gamma: f64,
) -> Array2<f64> {
    let n = rho.len();
    assert_eq!(u.len(), n);
    assert_eq!(p.len(), n);
    let mut out = Array2::<f64>::zeros((3, n));
    for i in 0..n {
        let kinetic = 0.5 * rho[i] * u[i] * u[i];
        let e = p[i] / (gamma - 1.0) + kinetic;
        out[[0, i]] = rho[i];
        out[[1, i]] = rho[i] * u[i];
        out[[2, i]] = e;
    }
    out
}

/// Unpack `U` into `(rho, u, p)`.
pub fn conservative_to_primitive(
    u_state: &Array2<f64>,
    gamma: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = u_state.ncols();
    let mut rho = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n);
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        let rho_i = u_state[[0, i]];
        let u_i = u_state[[1, i]] / rho_i;
        let e = u_state[[2, i]];
        rho[i] = rho_i;
        u[i] = u_i;
        p[i] = (gamma - 1.0) * (e - 0.5 * rho_i * u_i * u_i);
    }
    (rho, u, p)
}

/// Physical flux `F(U)`; same shape as `U`.
pub fn euler_flux(u_state: &Array2<f64>, gamma: f64) -> Array2<f64> {
    let (rho, u, p) = conservative_to_primitive(u_state, gamma);
    let n = rho.len();
    let mut f = Array2::<f64>::zeros((3, n));
    for i in 0..n {
        let rho_i = rho[i];
        let u_i = u[i];
        let p_i = p[i];
        let e = u_state[[2, i]];
        f[[0, i]] = rho_i * u_i;
        f[[1, i]] = rho_i * u_i * u_i + p_i;
        f[[2, i]] = (e + p_i) * u_i;
    }
    f
}

/// HLL numerical flux at every interface. `U_L` and `U_R` are left/right states per face.
///
/// Wave speed estimates use `S_L = min(u_L - a_L, u_R - a_R)`,
/// `S_R = max(u_L + a_L, u_R + a_R)` — the two-wave Davis (1988) bound.
pub fn hll_flux(u_l: &Array2<f64>, u_r: &Array2<f64>, gamma: f64) -> Array2<f64> {
    assert_eq!(u_l.shape(), u_r.shape());
    let n = u_l.ncols();

    let (rho_l, u_l_prim, p_l) = conservative_to_primitive(u_l, gamma);
    let (rho_r, u_r_prim, p_r) = conservative_to_primitive(u_r, gamma);

    let f_l = euler_flux(u_l, gamma);
    let f_r = euler_flux(u_r, gamma);

    let mut flux = Array2::<f64>::zeros((3, n));
    for i in 0..n {
        let a_l = (gamma * p_l[i] / rho_l[i]).sqrt();
        let a_r = (gamma * p_r[i] / rho_r[i]).sqrt();
        let s_l = (u_l_prim[i] - a_l).min(u_r_prim[i] - a_r);
        let s_r = (u_l_prim[i] + a_l).max(u_r_prim[i] + a_r);

        if s_l >= 0.0 {
            for k in 0..3 {
                flux[[k, i]] = f_l[[k, i]];
            }
        } else if s_r <= 0.0 {
            for k in 0..3 {
                flux[[k, i]] = f_r[[k, i]];
            }
        } else {
            let denom = s_r - s_l;
            for k in 0..3 {
                flux[[k, i]] = (s_r * f_l[[k, i]] - s_l * f_r[[k, i]]
                    + s_l * s_r * (u_r[[k, i]] - u_l[[k, i]]))
                    / denom;
            }
        }
    }
    flux
}

/// One forward-Euler step using HLL fluxes and transmissive (zero-gradient) BCs.
pub fn step_forward_euler(u_state: &Array2<f64>, gamma: f64, dx: f64, dt: f64) -> Array2<f64> {
    let n = u_state.ncols();
    let mut u_left = Array2::<f64>::zeros((3, n + 1));
    let mut u_right = Array2::<f64>::zeros((3, n + 1));

    for k in 0..3 {
        // Interior faces 1..n-1 get neighbor cells.
        for i in 1..n {
            u_left[[k, i]] = u_state[[k, i - 1]];
            u_right[[k, i]] = u_state[[k, i]];
        }
        // Transmissive: ghost == boundary.
        u_left[[k, 0]] = u_state[[k, 0]];
        u_right[[k, 0]] = u_state[[k, 0]];
        u_left[[k, n]] = u_state[[k, n - 1]];
        u_right[[k, n]] = u_state[[k, n - 1]];
    }

    let f = hll_flux(&u_left, &u_right, gamma);
    let lambda = dt / dx;
    let mut out = u_state.clone();
    for k in 0..3 {
        for i in 0..n {
            out[[k, i]] -= lambda * (f[[k, i + 1]] - f[[k, i]]);
        }
    }
    out
}

/// Max of `|u| + a` over the state — the CFL bound.
pub fn max_wave_speed(u_state: &Array2<f64>, gamma: f64) -> f64 {
    let (rho, u, p) = conservative_to_primitive(u_state, gamma);
    let mut max_speed: f64 = 0.0;
    for i in 0..rho.len() {
        let a = (gamma * p[i] / rho[i]).sqrt();
        let speed = u[i].abs() + a;
        if speed > max_speed {
            max_speed = speed;
        }
    }
    max_speed
}

/// Integrate to `t_final` with adaptive dt (CFL-constrained).
pub fn simulate(
    u0: &Array2<f64>,
    dx: f64,
    t_final: f64,
    gamma: f64,
    cfl_target: f64,
) -> Array2<f64> {
    let mut u = u0.clone();
    let mut t = 0.0;
    while t < t_final - 1e-14 {
        let mut dt = cfl_target * dx / max_wave_speed(&u, gamma);
        if t + dt > t_final {
            dt = t_final - t;
        }
        u = step_forward_euler(&u, gamma, dx, dt);
        t += dt;
    }
    u
}

/// Sod shock tube initial condition on `domain`, gamma = 1.4.
///
/// Left  (x < 0.5): rho=1.0,  u=0, p=1.0
/// Right (x > 0.5): rho=0.125, u=0, p=0.1
/// Reference t = 0.2.
pub fn sod_initial(n: usize, domain: (f64, f64), gamma: f64) -> (Array1<f64>, Array2<f64>) {
    let (x_min, x_max) = domain;
    let dx = (x_max - x_min) / n as f64;
    let x: Array1<f64> = Array1::from_iter((0..n).map(|i| x_min + (i as f64 + 0.5) * dx));
    let mut rho = Array1::<f64>::zeros(n);
    let u = Array1::<f64>::zeros(n);
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        if x[i] < 0.5 {
            rho[i] = 1.0;
            p[i] = 1.0;
        } else {
            rho[i] = 0.125;
            p[i] = 0.1;
        }
    }
    let u_state = primitive_to_conservative(&rho, &u, &p, gamma);
    (x, u_state)
}

