//! Pendulum-on-cart environment.
//!
//! State: `[x, x_dot, theta, theta_dot, i_coil]`. Angle θ is measured from
//! upright (θ=0 is the unstable equilibrium), positive CCW. The actuator is
//! a first-order coil: `di/dt = (V_sat - i) / tau_coil`, with the cart force
//! `F = k_force * i`. Commanded voltage saturates at `±v_sat`.
//!
//! Equations of motion (point-mass pole on a cart, no friction):
//! ```text
//!   x_ddot = (F + m*l*sin(θ)*θ_dot² - m*g*sin(θ)*cos(θ)) / (M + m*sin(θ)²)
//!   θ_ddot = (g*sin(θ) - cos(θ)*x_ddot) / l
//! ```
//! Derived from the Lagrangian in the Week-7 plan; see `tests/env.rs` for
//! conservation and linearization checks.

use ndarray::{array, Array1};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct EnvParams {
    pub m_cart: f64,
    pub m_pole: f64,
    pub l: f64,
    pub g: f64,
    pub tau_coil: f64,
    pub k_force: f64,
    pub v_sat: f64,
    pub obs_noise: [f64; 4],
    pub dt: f64,
}

pub fn default_params() -> EnvParams {
    EnvParams {
        m_cart: 1.0,
        m_pole: 0.1,
        l: 0.5,
        g: 9.81,
        tau_coil: 0.02,
        k_force: 1.0,
        v_sat: 10.0,
        obs_noise: [0.0, 0.0, 0.0, 0.0],
        dt: 5e-3,
    }
}

pub struct Env {
    state: Array1<f64>,
    params: EnvParams,
    rng: StdRng,
}

impl Env {
    pub fn new(params: EnvParams, seed: u64) -> Self {
        Self {
            state: Array1::zeros(5),
            params,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn params(&self) -> &EnvParams {
        &self.params
    }

    pub fn reset(&mut self, init_xxdtt: [f64; 4]) {
        let [x, xd, th, thd] = init_xxdtt;
        self.state = array![x, xd, th, thd, 0.0];
    }

    pub fn true_state(&self) -> &Array1<f64> {
        &self.state
    }

    pub fn true_state_xxdtt(&self) -> [f64; 4] {
        [self.state[0], self.state[1], self.state[2], self.state[3]]
    }

    pub fn coil_current(&self) -> f64 {
        self.state[4]
    }

    /// Apply an instantaneous impulse to the cart: Δ(x_dot) = impulse / m_total.
    /// Used in the demo to simulate a whack at t=2.5 s.
    pub fn apply_impulse(&mut self, impulse_ns: f64) {
        let m_total = self.params.m_cart + self.params.m_pole;
        self.state[1] += impulse_ns / m_total;
    }

    /// One RK4 step of the full 5-dim system with zero-order-hold voltage.
    pub fn step(&mut self, v_command: f64) -> [f64; 4] {
        let v_sat = v_command.clamp(-self.params.v_sat, self.params.v_sat);
        let dt = self.params.dt;

        let k1 = dynamics(&self.state, v_sat, &self.params);
        let s2 = &self.state + &(&k1 * (dt * 0.5));
        let k2 = dynamics(&s2, v_sat, &self.params);
        let s3 = &self.state + &(&k2 * (dt * 0.5));
        let k3 = dynamics(&s3, v_sat, &self.params);
        let s4 = &self.state + &(&k3 * dt);
        let k4 = dynamics(&s4, v_sat, &self.params);

        self.state = &self.state + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0));

        self.noisy_obs()
    }

    fn noisy_obs(&mut self) -> [f64; 4] {
        let mut obs = self.true_state_xxdtt();
        for (o, &sigma) in obs.iter_mut().zip(self.params.obs_noise.iter()) {
            if sigma > 0.0 {
                let n = Normal::new(0.0, sigma).expect("valid normal");
                *o += n.sample(&mut self.rng);
            }
        }
        obs
    }

    /// Total mechanical energy of the cart-pole pair (excludes coil energy).
    /// Conserved when `k_force * i == 0` (no work on the cart).
    pub fn energy(&self) -> f64 {
        let (m_c, m_p) = (self.params.m_cart, self.params.m_pole);
        let (l, g) = (self.params.l, self.params.g);
        let (_x, xd, th, thd) = (
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
        );
        let c = th.cos();
        let t_kin =
            0.5 * (m_c + m_p) * xd * xd + m_p * l * c * xd * thd + 0.5 * m_p * l * l * thd * thd;
        let v_pot = m_p * g * l * c;
        t_kin + v_pot
    }
}

/// ds/dt for state `[x, x_dot, theta, theta_dot, i_coil]` under voltage `V`.
fn dynamics(s: &Array1<f64>, v_sat: f64, p: &EnvParams) -> Array1<f64> {
    let (xd, th, thd, i) = (s[1], s[2], s[3], s[4]);
    let (m_c, m_p, l, g) = (p.m_cart, p.m_pole, p.l, p.g);
    let (sn, cs) = (th.sin(), th.cos());

    let f_cart = p.k_force * i;
    let denom = m_c + m_p * sn * sn;
    let xdd = (f_cart + m_p * l * sn * thd * thd - m_p * g * sn * cs) / denom;
    let thdd = (g * sn - cs * xdd) / l;
    let idd = (v_sat - i) / p.tau_coil;

    array![xd, xdd, thd, thdd, idd]
}

/// Linearized A, B about the upright equilibrium, state `[x, x_dot, θ, θ_dot]`,
/// control `u = F` (cart force, post-actuator). Actuator dynamics not included.
pub fn linearize_cartpole(p: &EnvParams) -> ([[f64; 4]; 4], [f64; 4]) {
    let (m_c, m_p, l, g) = (p.m_cart, p.m_pole, p.l, p.g);
    let m_total = m_c + m_p;
    let a = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m_p * g / m_c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * m_total / (m_c * l), 0.0],
    ];
    let b = [0.0, 1.0 / m_c, 0.0, -1.0 / (m_c * l)];
    (a, b)
}
