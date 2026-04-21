//! 1D ideal magnetohydrodynamics (MHD) with HLL flux.
//!
//! State vector (7 variables; `B_x` is a constant parameter since `∇·B = 0` in 1D):
//!
//! ```text
//! U = [rho, rho*u, rho*v, rho*w, B_y, B_z, E]^T
//! ```
//!
//! Units are natural (`mu_0 = 1`). Total energy:
//! `E = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2`.
//!
//! Flux (x-direction, `B_x` constant):
//!
//! ```text
//! F[0] = rho*u
//! F[1] = rho*u^2 + p_total - B_x^2
//! F[2] = rho*u*v - B_x*B_y
//! F[3] = rho*u*w - B_x*B_z
//! F[4] = u*B_y - v*B_x
//! F[5] = u*B_z - w*B_x
//! F[6] = (E + p_total)*u - B_x*(u*B_x + v*B_y + w*B_z)
//! ```
//!
//! with `p_total = p + 0.5*|B|^2`.
//!
//! Canonical 1D test: Brio & Wu, JCP 75 (1988) 400. Reproduced here at `t=0.1`
//! using `gamma = 2.0` and `B_x = 0.75`.

use ndarray::{Array1, Array2};

pub const GAMMA_BRIO_WU: f64 = 2.0;
pub const BX_BRIO_WU: f64 = 0.75;

#[allow(clippy::too_many_arguments)]
pub fn primitive_to_conservative(
    rho: &Array1<f64>,
    u: &Array1<f64>,
    v: &Array1<f64>,
    w: &Array1<f64>,
    by: &Array1<f64>,
    bz: &Array1<f64>,
    p: &Array1<f64>,
    gamma: f64,
    bx: f64,
) -> Array2<f64> {
    let n = rho.len();
    let mut out = Array2::<f64>::zeros((7, n));
    for i in 0..n {
        let kinetic = 0.5 * rho[i] * (u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
        let magnetic = 0.5 * (bx * bx + by[i] * by[i] + bz[i] * bz[i]);
        let e = p[i] / (gamma - 1.0) + kinetic + magnetic;
        out[[0, i]] = rho[i];
        out[[1, i]] = rho[i] * u[i];
        out[[2, i]] = rho[i] * v[i];
        out[[3, i]] = rho[i] * w[i];
        out[[4, i]] = by[i];
        out[[5, i]] = bz[i];
        out[[6, i]] = e;
    }
    out
}

pub struct PrimitiveState {
    pub rho: Array1<f64>,
    pub u: Array1<f64>,
    pub v: Array1<f64>,
    pub w: Array1<f64>,
    pub by: Array1<f64>,
    pub bz: Array1<f64>,
    pub p: Array1<f64>,
}

pub fn conservative_to_primitive(u_state: &Array2<f64>, gamma: f64, bx: f64) -> PrimitiveState {
    let n = u_state.ncols();
    let mut rho = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n);
    let mut v = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);
    let mut by = Array1::<f64>::zeros(n);
    let mut bz = Array1::<f64>::zeros(n);
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        let rho_i = u_state[[0, i]];
        let u_i = u_state[[1, i]] / rho_i;
        let v_i = u_state[[2, i]] / rho_i;
        let w_i = u_state[[3, i]] / rho_i;
        let by_i = u_state[[4, i]];
        let bz_i = u_state[[5, i]];
        let e = u_state[[6, i]];
        let kinetic = 0.5 * rho_i * (u_i * u_i + v_i * v_i + w_i * w_i);
        let magnetic = 0.5 * (bx * bx + by_i * by_i + bz_i * bz_i);
        rho[i] = rho_i;
        u[i] = u_i;
        v[i] = v_i;
        w[i] = w_i;
        by[i] = by_i;
        bz[i] = bz_i;
        p[i] = (gamma - 1.0) * (e - kinetic - magnetic);
    }
    PrimitiveState { rho, u, v, w, by, bz, p }
}

pub fn mhd_flux(u_state: &Array2<f64>, gamma: f64, bx: f64) -> Array2<f64> {
    let s = conservative_to_primitive(u_state, gamma, bx);
    let n = s.rho.len();
    let mut f = Array2::<f64>::zeros((7, n));
    for i in 0..n {
        let (rho, u, v, w, by, bz, p) = (s.rho[i], s.u[i], s.v[i], s.w[i], s.by[i], s.bz[i], s.p[i]);
        let p_total = p + 0.5 * (bx * bx + by * by + bz * bz);
        let e = u_state[[6, i]];
        f[[0, i]] = rho * u;
        f[[1, i]] = rho * u * u + p_total - bx * bx;
        f[[2, i]] = rho * u * v - bx * by;
        f[[3, i]] = rho * u * w - bx * bz;
        f[[4, i]] = u * by - v * bx;
        f[[5, i]] = u * bz - w * bx;
        f[[6, i]] = (e + p_total) * u - bx * (u * bx + v * by + w * bz);
    }
    f
}

/// Fast magnetosonic speed:
/// `c_f^2 = 0.5*(a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4*a^2*b_x^2))`.
pub fn fast_magnetosonic_speed(
    rho: &Array1<f64>,
    p: &Array1<f64>,
    by: &Array1<f64>,
    bz: &Array1<f64>,
    gamma: f64,
    bx: f64,
) -> Array1<f64> {
    let n = rho.len();
    let mut cf = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a2 = gamma * p[i] / rho[i];
        let b2 = (bx * bx + by[i] * by[i] + bz[i] * bz[i]) / rho[i];
        let bx2 = bx * bx / rho[i];
        let disc = ((a2 + b2).powi(2) - 4.0 * a2 * bx2).max(0.0);
        cf[i] = (0.5 * (a2 + b2 + disc.sqrt())).sqrt();
    }
    cf
}

pub fn mhd_hll_flux(
    u_l: &Array2<f64>,
    u_r: &Array2<f64>,
    gamma: f64,
    bx: f64,
) -> Array2<f64> {
    let n = u_l.ncols();
    let s_l_prim = conservative_to_primitive(u_l, gamma, bx);
    let s_r_prim = conservative_to_primitive(u_r, gamma, bx);
    let cf_l = fast_magnetosonic_speed(&s_l_prim.rho, &s_l_prim.p, &s_l_prim.by, &s_l_prim.bz, gamma, bx);
    let cf_r = fast_magnetosonic_speed(&s_r_prim.rho, &s_r_prim.p, &s_r_prim.by, &s_r_prim.bz, gamma, bx);

    let f_l = mhd_flux(u_l, gamma, bx);
    let f_r = mhd_flux(u_r, gamma, bx);

    let mut flux = Array2::<f64>::zeros((7, n));
    for i in 0..n {
        let s_l = (s_l_prim.u[i] - cf_l[i]).min(s_r_prim.u[i] - cf_r[i]);
        let s_r = (s_l_prim.u[i] + cf_l[i]).max(s_r_prim.u[i] + cf_r[i]);
        if s_l >= 0.0 {
            for k in 0..7 {
                flux[[k, i]] = f_l[[k, i]];
            }
        } else if s_r <= 0.0 {
            for k in 0..7 {
                flux[[k, i]] = f_r[[k, i]];
            }
        } else {
            let denom = s_r - s_l;
            for k in 0..7 {
                flux[[k, i]] = (s_r * f_l[[k, i]] - s_l * f_r[[k, i]]
                    + s_l * s_r * (u_r[[k, i]] - u_l[[k, i]]))
                    / denom;
            }
        }
    }
    flux
}

pub fn mhd_step(u_state: &Array2<f64>, gamma: f64, bx: f64, dx: f64, dt: f64) -> Array2<f64> {
    let n = u_state.ncols();
    let mut u_left = Array2::<f64>::zeros((7, n + 1));
    let mut u_right = Array2::<f64>::zeros((7, n + 1));
    for k in 0..7 {
        for i in 1..n {
            u_left[[k, i]] = u_state[[k, i - 1]];
            u_right[[k, i]] = u_state[[k, i]];
        }
        u_left[[k, 0]] = u_state[[k, 0]];
        u_right[[k, 0]] = u_state[[k, 0]];
        u_left[[k, n]] = u_state[[k, n - 1]];
        u_right[[k, n]] = u_state[[k, n - 1]];
    }
    let f = mhd_hll_flux(&u_left, &u_right, gamma, bx);
    let lambda = dt / dx;
    let mut out = u_state.clone();
    for k in 0..7 {
        for i in 0..n {
            out[[k, i]] -= lambda * (f[[k, i + 1]] - f[[k, i]]);
        }
    }
    out
}

pub fn mhd_max_wave_speed(u_state: &Array2<f64>, gamma: f64, bx: f64) -> f64 {
    let s = conservative_to_primitive(u_state, gamma, bx);
    let cf = fast_magnetosonic_speed(&s.rho, &s.p, &s.by, &s.bz, gamma, bx);
    let mut m: f64 = 0.0;
    for i in 0..s.rho.len() {
        let speed = s.u[i].abs() + cf[i];
        if speed > m {
            m = speed;
        }
    }
    m
}

pub fn mhd_simulate(
    u0: &Array2<f64>,
    dx: f64,
    t_final: f64,
    gamma: f64,
    bx: f64,
    cfl_target: f64,
) -> Array2<f64> {
    let mut u = u0.clone();
    let mut t = 0.0;
    while t < t_final - 1e-14 {
        let mut dt = cfl_target * dx / mhd_max_wave_speed(&u, gamma, bx);
        if t + dt > t_final {
            dt = t_final - t;
        }
        u = mhd_step(&u, gamma, bx, dx, dt);
        t += dt;
    }
    u
}

/// Brio & Wu 1988 shock-tube initial conditions.
///
/// Left  (x < 0.5): rho=1.0,   p=1.0, u=v=w=0, B_y=+1, B_z=0
/// Right (x > 0.5): rho=0.125, p=0.1, u=v=w=0, B_y=-1, B_z=0
/// B_x = 0.75 everywhere; gamma = 2.0; reference t = 0.1.
pub fn brio_wu_initial(n: usize, domain: (f64, f64)) -> (Array1<f64>, Array2<f64>, f64) {
    let (x_min, x_max) = domain;
    let dx = (x_max - x_min) / n as f64;
    let x: Array1<f64> = Array1::from_iter((0..n).map(|i| x_min + (i as f64 + 0.5) * dx));
    let n = x.len();
    let mut rho = Array1::<f64>::zeros(n);
    let u = Array1::<f64>::zeros(n);
    let v = Array1::<f64>::zeros(n);
    let w = Array1::<f64>::zeros(n);
    let mut by = Array1::<f64>::zeros(n);
    let bz = Array1::<f64>::zeros(n);
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        if x[i] < 0.5 {
            rho[i] = 1.0;
            p[i] = 1.0;
            by[i] = 1.0;
        } else {
            rho[i] = 0.125;
            p[i] = 0.1;
            by[i] = -1.0;
        }
    }
    let u_state = primitive_to_conservative(
        &rho, &u, &v, &w, &by, &bz, &p, GAMMA_BRIO_WU, BX_BRIO_WU,
    );
    (x, u_state, BX_BRIO_WU)
}
