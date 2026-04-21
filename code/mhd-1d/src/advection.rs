//! 1D linear advection: `du/dt + c du/dx = 0`, periodic BC.
//!
//! Upwind finite-volume with forward Euler. First-order accurate in both time and
//! space for smooth solutions. Numerical diffusion from the modified-equation
//! analysis of first-order upwind is
//!
//! ```text
//! nu_num = c * dx * (1 - CFL) / 2
//! ```
//!
//! A Gaussian pulse run for one full period on a periodic domain is the primary
//! verification test; see [`run_gaussian_test`] and [`convergence_study`].

use ndarray::Array1;

/// Advance `du/dt + c du/dx = 0` by `n_steps` with upwind FV + forward Euler,
/// periodic boundary conditions.
///
/// Returns `Err` if the CFL condition `|c| * dt / dx < 1` is violated, since the
/// scheme is not stable past that bound.
pub fn advect_upwind(
    u0: &Array1<f64>,
    c: f64,
    dx: f64,
    dt: f64,
    n_steps: usize,
) -> Result<Array1<f64>, String> {
    let cfl = c * dt / dx;
    if cfl.abs() >= 1.0 {
        return Err(format!(
            "CFL = {cfl:.4} must satisfy |CFL| < 1 for stability"
        ));
    }

    let n = u0.len();
    let mut u = u0.clone();
    let mut flux = Array1::<f64>::zeros(n);
    let lambda = dt / dx;

    for _ in 0..n_steps {
        if c >= 0.0 {
            // Upwind from the left: flux at face i+1/2 is c * u[i].
            for i in 0..n {
                flux[i] = c * u[i];
            }
        } else {
            // Upwind from the right: flux at face i+1/2 is c * u[i+1].
            for i in 0..n {
                flux[i] = c * u[(i + 1) % n];
            }
        }
        // Update u[i] -= lambda * (flux[i] - flux[i-1])  (periodic).
        let mut next = Array1::<f64>::zeros(n);
        for i in 0..n {
            let i_minus = if i == 0 { n - 1 } else { i - 1 };
            next[i] = u[i] - lambda * (flux[i] - flux[i_minus]);
        }
        u = next;
    }
    Ok(u)
}

/// Result of a single Gaussian-pulse run on a periodic domain.
#[derive(Debug, Clone)]
pub struct GaussianRun {
    pub x: Array1<f64>,
    pub u_initial: Array1<f64>,
    pub u_final: Array1<f64>,
    /// `Some(err)` iff `t_final` is an integer multiple of the period,
    /// so the analytical final state equals the initial state.
    pub l2_error: Option<f64>,
}

/// Advect a Gaussian pulse on `[x_min, x_max]` for exactly `t_final`.
///
/// With defaults (`c=1`, `domain=(0,1)`, `t_final=1.0`), exactly one full period
/// fits, so `l2_error` is always `Some`.
pub fn run_gaussian_test(
    n: usize,
    c: f64,
    t_final: f64,
    domain: (f64, f64),
    sigma: f64,
    x0: f64,
    cfl_target: f64,
) -> GaussianRun {
    let (x_min, x_max) = domain;
    let length = x_max - x_min;
    let dx = length / n as f64;
    let x: Array1<f64> =
        Array1::from_iter((0..n).map(|i| x_min + (i as f64 + 0.5) * dx));
    let u_initial: Array1<f64> = x.mapv(|xi| (-((xi - x0) / sigma).powi(2)).exp());

    let dt_cfl = cfl_target * dx / c.abs();
    let n_steps = (t_final / dt_cfl).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let u_final = advect_upwind(&u_initial, c, dx, dt, n_steps)
        .expect("run_gaussian_test chose a stable CFL");

    let period = length / c.abs();
    let n_periods = t_final / period;
    let l2_error = if (n_periods - n_periods.round()).abs() < 1e-10 {
        let mut sum = 0.0;
        for i in 0..n {
            let d = u_final[i] - u_initial[i];
            sum += d * d;
        }
        Some((sum / n as f64).sqrt())
    } else {
        None
    };

    GaussianRun {
        x,
        u_initial,
        u_final,
        l2_error,
    }
}

/// Default parameters matching the Python reference: `c=1`, `domain=(0,1)`,
/// `t_final=1.0`, `sigma=0.1`, `x0=0.5`, `cfl_target=0.5`.
pub fn run_gaussian_default(n: usize) -> GaussianRun {
    run_gaussian_test(n, 1.0, 1.0, (0.0, 1.0), 0.1, 0.5, 0.5)
}

/// Sweep `Ns`, return `(Ns, errors, orders)` where `orders[i]` is the
/// empirical order between `Ns[i]` and `Ns[i+1]`.
pub fn convergence_study(ns: &[usize]) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let errors: Vec<f64> = ns
        .iter()
        .map(|&n| {
            run_gaussian_default(n)
                .l2_error
                .expect("default params cover exactly one period")
        })
        .collect();

    let orders: Vec<f64> = ns
        .windows(2)
        .zip(errors.windows(2))
        .map(|(n_pair, e_pair)| {
            (e_pair[0] / e_pair[1]).ln() / (n_pair[1] as f64 / n_pair[0] as f64).ln()
        })
        .collect();

    (ns.to_vec(), errors, orders)
}
