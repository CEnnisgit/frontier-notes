//! Pendulum-on-cart with an electromagnetic-style actuator.
//!
//! State: `[x, x_dot, theta, theta_dot, F]`, θ measured from upright
//! (θ = 0 is the unstable equilibrium we're trying to stabilize, +θ tilts
//! right). The actuator takes a commanded force `F_cmd` and delivers `F`
//! through a first-order lag (coil time constant `tau_a`), with symmetric
//! saturation on `F_cmd`. Integration is fixed-step RK4 at `dt`.
//!
//! Dynamics derived from the Lagrangian (cart mass M, pendulum mass m,
//! rod length l, gravity g):
//!
//! ```text
//!   x_ddot     = [F + m l θ_dot^2 sin θ − m g sin θ cos θ] / (M + m sin^2 θ)
//!   θ_ddot     = (g sin θ − x_ddot cos θ) / l
//!   F_dot      = (sat(F_cmd) − F) / tau_a
//! ```

use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub m_cart: f64,
    pub m_pend: f64,
    pub l: f64,
    pub g: f64,
    pub tau_a: f64,
    pub f_max: f64,
    pub dt: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            m_cart: 1.0,
            m_pend: 0.1,
            l: 0.5,
            g: 9.81,
            tau_a: 0.05,
            f_max: 20.0,
            dt: 0.01,
        }
    }
}

/// Observation noise (Gaussian per-component, additive).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoiseSpec {
    pub x_sigma: f64,
    pub x_dot_sigma: f64,
    pub theta_sigma: f64,
    pub theta_dot_sigma: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub x: f64,
    pub x_dot: f64,
    pub theta: f64,
    pub theta_dot: f64,
    pub force: f64,
}

impl State {
    pub fn upright(theta0: f64) -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: theta0,
            theta_dot: 0.0,
            force: 0.0,
        }
    }

    pub fn as_array(&self) -> [f64; 5] {
        [self.x, self.x_dot, self.theta, self.theta_dot, self.force]
    }
}

pub struct Env {
    pub p: Params,
    pub s: State,
    pub t: f64,
}

impl Env {
    pub fn new(p: Params, s0: State) -> Self {
        Self { p, s: s0, t: 0.0 }
    }

    pub fn reset(&mut self, s0: State) {
        self.s = s0;
        self.t = 0.0;
    }

    /// Advance the state by one `dt` using RK4 on the 5D (state + actuator) system.
    /// `f_cmd` is the commanded force; saturation is applied inside the derivative.
    pub fn step(&mut self, f_cmd: f64) -> State {
        let y = self.s.as_array();
        let k1 = deriv(&y, f_cmd, &self.p);
        let y2 = add(&y, &k1, self.p.dt * 0.5);
        let k2 = deriv(&y2, f_cmd, &self.p);
        let y3 = add(&y, &k2, self.p.dt * 0.5);
        let k3 = deriv(&y3, f_cmd, &self.p);
        let y4 = add(&y, &k3, self.p.dt);
        let k4 = deriv(&y4, f_cmd, &self.p);
        let mut y_new = [0.0; 5];
        for i in 0..5 {
            y_new[i] = y[i] + self.p.dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        self.s = State {
            x: y_new[0],
            x_dot: y_new[1],
            theta: y_new[2],
            theta_dot: y_new[3],
            force: y_new[4],
        };
        self.t += self.p.dt;
        self.s
    }

    /// Observation with Gaussian noise (does not mutate true state).
    pub fn observe(&self, noise: &NoiseSpec, rng: &mut StdRng) -> [f64; 4] {
        let sample = |sigma: f64, rng: &mut StdRng| -> f64 {
            if sigma == 0.0 {
                0.0
            } else {
                Normal::new(0.0, sigma).unwrap().sample(rng)
            }
        };
        [
            self.s.x + sample(noise.x_sigma, rng),
            self.s.x_dot + sample(noise.x_dot_sigma, rng),
            self.s.theta + sample(noise.theta_sigma, rng),
            self.s.theta_dot + sample(noise.theta_dot_sigma, rng),
        ]
    }

    /// Apply an instantaneous impulse to `x_dot` or `theta_dot`. Used to
    /// inject the perturbation in the "perturbed" demo regime.
    pub fn perturb_impulse(&mut self, dx_dot: f64, dtheta_dot: f64) {
        self.s.x_dot += dx_dot;
        self.s.theta_dot += dtheta_dot;
    }

    /// Total mechanical energy (kinetic + potential), for the sanity test.
    /// With theta=0 upright, the potential is `m g l cos θ` — maximum at
    /// the unstable equilibrium.
    pub fn energy(&self) -> f64 {
        let p = &self.p;
        let m = p.m_pend;
        let l = p.l;
        let ke_cart = 0.5 * p.m_cart * self.s.x_dot * self.s.x_dot;
        let vx_tip = self.s.x_dot + l * self.s.theta_dot * self.s.theta.cos();
        let vy_tip = -l * self.s.theta_dot * self.s.theta.sin();
        let ke_pend = 0.5 * m * (vx_tip * vx_tip + vy_tip * vy_tip);
        let pe = m * p.g * l * self.s.theta.cos();
        ke_cart + ke_pend + pe
    }
}

fn add(y: &[f64; 5], k: &[f64; 5], h: f64) -> [f64; 5] {
    let mut out = [0.0; 5];
    for i in 0..5 {
        out[i] = y[i] + h * k[i];
    }
    out
}

fn deriv(y: &[f64; 5], f_cmd: f64, p: &Params) -> [f64; 5] {
    let x_dot = y[1];
    let theta = y[2];
    let theta_dot = y[3];
    let force = y[4];

    let s = theta.sin();
    let c = theta.cos();
    let denom = p.m_cart + p.m_pend * s * s;

    let x_ddot = (force + p.m_pend * p.l * theta_dot * theta_dot * s
        - p.m_pend * p.g * s * c)
        / denom;
    let theta_ddot = (p.g * s - x_ddot * c) / p.l;

    let f_cmd_sat = f_cmd.clamp(-p.f_max, p.f_max);
    let force_dot = (f_cmd_sat - force) / p.tau_a;

    [x_dot, x_ddot, theta_dot, theta_ddot, force_dot]
}

/// Continuous-time linearization around the upright fixed point, ignoring
/// the actuator lag (treating `F` as a direct input `u`). Used for LQR /
/// pole-placement design and for the sanity test.
///
/// State order: `[x, x_dot, theta, theta_dot]`, input: `F`.
pub fn linearize_upright(p: &Params) -> ([[f64; 4]; 4], [f64; 4]) {
    let m = p.m_pend;
    let mc = p.m_cart;
    let g = p.g;
    let l = p.l;
    let a = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m * g / mc, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, (mc + m) * g / (mc * l), 0.0],
    ];
    let b = [0.0, 1.0 / mc, 0.0, -1.0 / (mc * l)];
    (a, b)
}

/// Sample a random initial state near upright, for Monte-Carlo rollouts.
pub fn sample_initial(rng: &mut StdRng, theta_amp: f64, x_amp: f64) -> State {
    State {
        x: rng.gen_range(-x_amp..x_amp),
        x_dot: 0.0,
        theta: rng.gen_range(-theta_amp..theta_amp),
        theta_dot: 0.0,
        force: 0.0,
    }
}
