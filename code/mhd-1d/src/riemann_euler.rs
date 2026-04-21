//! Exact Riemann solver for 1D compressible Euler.
//!
//! Implements the two-shock / two-rarefaction Newton iteration for the
//! star-region pressure, then samples the self-similar solution at any
//! `xi = (x - x0)/t`.
//!
//! Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
//! 3rd ed., §4.2–4.5. Equation numbers in the comments cite that text.

use ndarray::Array1;

pub const GAMMA_DEFAULT: f64 = 1.4;

fn sound_speed(rho: f64, p: f64, gamma: f64) -> f64 {
    (gamma * p / rho).sqrt()
}

/// Pressure function `f_K` and its derivative `f'_K` (Toro eqs 4.6 / 4.7).
fn pressure_function(p: f64, rho_k: f64, p_k: f64, gamma: f64) -> (f64, f64) {
    let a_k = sound_speed(rho_k, p_k, gamma);
    if p > p_k {
        // Shock branch.
        let big_a = 2.0 / ((gamma + 1.0) * rho_k);
        let big_b = (gamma - 1.0) / (gamma + 1.0) * p_k;
        let sq = (big_a / (p + big_b)).sqrt();
        let f = (p - p_k) * sq;
        let fp = sq * (1.0 - 0.5 * (p - p_k) / (big_b + p));
        (f, fp)
    } else {
        // Rarefaction branch.
        let f = (2.0 * a_k / (gamma - 1.0))
            * ((p / p_k).powf((gamma - 1.0) / (2.0 * gamma)) - 1.0);
        let fp = (1.0 / (rho_k * a_k)) * (p / p_k).powf(-(gamma + 1.0) / (2.0 * gamma));
        (f, fp)
    }
}

fn initial_pressure_guess(
    rho_l: f64,
    u_l: f64,
    p_l: f64,
    rho_r: f64,
    u_r: f64,
    p_r: f64,
    gamma: f64,
) -> f64 {
    // Primitive-variable (PVRS) guess, Toro eq 4.46, clamped positive.
    let a_l = sound_speed(rho_l, p_l, gamma);
    let a_r = sound_speed(rho_r, p_r, gamma);
    let p_pv = 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r);
    p_pv.max(1e-8)
}

/// Iteratively solve `f_L(p) + f_R(p) + Δu = 0` for `p_star`; return `(p_star, u_star)`.
pub fn solve_star_state(
    rho_l: f64,
    u_l: f64,
    p_l: f64,
    rho_r: f64,
    u_r: f64,
    p_r: f64,
    gamma: f64,
) -> (f64, f64) {
    let tol = 1e-10;
    let max_iter = 100;
    let du = u_r - u_l;
    let mut p = initial_pressure_guess(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma);
    let mut converged = false;
    for _ in 0..max_iter {
        let (f_l, fp_l) = pressure_function(p, rho_l, p_l, gamma);
        let (f_r, fp_r) = pressure_function(p, rho_r, p_r, gamma);
        let f = f_l + f_r + du;
        let fp = fp_l + fp_r;
        let mut p_new = p - f / fp;
        if p_new < 0.0 {
            p_new = 0.5 * p;
        }
        let rel = (p_new - p).abs() / (0.5 * (p + p_new));
        p = p_new;
        if rel < tol {
            converged = true;
            break;
        }
    }
    assert!(converged, "Riemann star-state iteration did not converge");

    let (f_l, _) = pressure_function(p, rho_l, p_l, gamma);
    let (f_r, _) = pressure_function(p, rho_r, p_r, gamma);
    let u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l);
    (p, u_star)
}

/// Exact `(rho, u, p)` at self-similar coordinate `xi = (x - x0)/t`.
#[allow(clippy::too_many_arguments)]
pub fn sample(
    xi: f64,
    rho_l: f64,
    u_l: f64,
    p_l: f64,
    rho_r: f64,
    u_r: f64,
    p_r: f64,
    gamma: f64,
    p_star: f64,
    u_star: f64,
) -> (f64, f64, f64) {
    let a_l = sound_speed(rho_l, p_l, gamma);
    let a_r = sound_speed(rho_r, p_r, gamma);

    if xi <= u_star {
        // Left of contact.
        if p_star > p_l {
            // Left shock.
            let s_shock = u_l
                - a_l
                    * ((gamma + 1.0) / (2.0 * gamma) * p_star / p_l
                        + (gamma - 1.0) / (2.0 * gamma))
                        .sqrt();
            if xi < s_shock {
                return (rho_l, u_l, p_l);
            }
            let num = p_star / p_l + (gamma - 1.0) / (gamma + 1.0);
            let den = (gamma - 1.0) / (gamma + 1.0) * p_star / p_l + 1.0;
            let rho_star_l = rho_l * (num / den);
            (rho_star_l, u_star, p_star)
        } else {
            // Left rarefaction.
            let s_head = u_l - a_l;
            let a_star_l = a_l * (p_star / p_l).powf((gamma - 1.0) / (2.0 * gamma));
            let s_tail = u_star - a_star_l;
            if xi < s_head {
                return (rho_l, u_l, p_l);
            }
            if xi > s_tail {
                let rho_star_l = rho_l * (p_star / p_l).powf(1.0 / gamma);
                return (rho_star_l, u_star, p_star);
            }
            // Inside fan.
            let c = 2.0 / (gamma + 1.0)
                + (gamma - 1.0) / ((gamma + 1.0) * a_l) * (u_l - xi);
            let rho = rho_l * c.powf(2.0 / (gamma - 1.0));
            let u = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * u_l + xi);
            let p = p_l * c.powf(2.0 * gamma / (gamma - 1.0));
            (rho, u, p)
        }
    } else {
        // Right of contact — mirror.
        if p_star > p_r {
            let s_shock = u_r
                + a_r
                    * ((gamma + 1.0) / (2.0 * gamma) * p_star / p_r
                        + (gamma - 1.0) / (2.0 * gamma))
                        .sqrt();
            if xi > s_shock {
                return (rho_r, u_r, p_r);
            }
            let num = p_star / p_r + (gamma - 1.0) / (gamma + 1.0);
            let den = (gamma - 1.0) / (gamma + 1.0) * p_star / p_r + 1.0;
            let rho_star_r = rho_r * (num / den);
            (rho_star_r, u_star, p_star)
        } else {
            let s_head = u_r + a_r;
            let a_star_r = a_r * (p_star / p_r).powf((gamma - 1.0) / (2.0 * gamma));
            let s_tail = u_star + a_star_r;
            if xi > s_head {
                return (rho_r, u_r, p_r);
            }
            if xi < s_tail {
                let rho_star_r = rho_r * (p_star / p_r).powf(1.0 / gamma);
                return (rho_star_r, u_star, p_star);
            }
            let c = 2.0 / (gamma + 1.0)
                - (gamma - 1.0) / ((gamma + 1.0) * a_r) * (u_r - xi);
            let rho = rho_r * c.powf(2.0 / (gamma - 1.0));
            let u = 2.0 / (gamma + 1.0) * (-a_r + (gamma - 1.0) / 2.0 * u_r + xi);
            let p = p_r * c.powf(2.0 * gamma / (gamma - 1.0));
            (rho, u, p)
        }
    }
}

/// Evaluate the exact solution at each cell center in `x` at time `t`.
#[allow(clippy::too_many_arguments)]
pub fn sample_grid(
    x: &Array1<f64>,
    t: f64,
    x0: f64,
    rho_l: f64,
    u_l: f64,
    p_l: f64,
    rho_r: f64,
    u_r: f64,
    p_r: f64,
    gamma: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let (p_star, u_star) = solve_star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma);
    let n = x.len();
    let mut rho = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n);
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = (x[i] - x0) / t;
        let (r, uu, pp) =
            sample(xi, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, p_star, u_star);
        rho[i] = r;
        u[i] = uu;
        p[i] = pp;
    }
    (rho, u, p)
}
