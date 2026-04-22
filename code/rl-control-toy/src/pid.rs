//! Classical controllers for the upright cart-pole.
//!
//! Two flavors, both consuming the full 4-element observation
//! `[x, x_dot, theta, theta_dot]` and returning a commanded force `F_cmd`:
//!
//! * `Lqr` — state-feedback `K` designed by discrete-time value iteration on
//!   the Riccati equation of the upright linearization. This is the
//!   "LQR on the linearized regime" controller.
//! * `Pid` — a hand-tuned two-loop PD: fast loop on angle, slow loop on cart
//!   position. The "PID" name is conventional; there's no integral term
//!   because the plant's DC gain is zero and an I-term is unstable without
//!   careful anti-windup at actuator saturation.

use crate::env::{linearize_upright, Params};
use ndarray::{arr1, arr2, Array1, Array2};

pub struct Lqr {
    /// State-feedback gain row vector such that `F_cmd = -K · x`.
    pub k: [f64; 4],
}

impl Lqr {
    /// Design an LQR for the upright linearization with cost diag(Q) on state
    /// and R on the scalar input. Solves the discrete-time ARE by 400 steps
    /// of value iteration after an Euler discretization at `dt_design`.
    pub fn design(p: &Params, q_diag: [f64; 4], r: f64, dt_design: f64) -> Self {
        let (a_c, b_c) = linearize_upright(p);
        let (a_d, b_d) = euler_discretize(&a_c, &b_c, dt_design);

        let q = arr2(&[
            [q_diag[0], 0.0, 0.0, 0.0],
            [0.0, q_diag[1], 0.0, 0.0],
            [0.0, 0.0, q_diag[2], 0.0],
            [0.0, 0.0, 0.0, q_diag[3]],
        ]);
        let r_mat = arr2(&[[r]]);
        let a = arr2(&a_d);
        let b_col: Array2<f64> = arr1(&b_d).insert_axis(ndarray::Axis(1));

        let mut p_mat = q.clone();
        for _ in 0..400 {
            // P_next = Q + A^T P A - A^T P B (R + B^T P B)^-1 B^T P A
            let at_p = a.t().dot(&p_mat);
            let at_p_a = at_p.dot(&a);
            let at_p_b = at_p.dot(&b_col);
            let bt_p = b_col.t().dot(&p_mat);
            let bt_p_b = bt_p.dot(&b_col);
            let r_pb = &r_mat + &bt_p_b; // scalar-valued 1x1
            let r_pb_inv = 1.0 / r_pb[[0, 0]];
            let bt_p_a = bt_p.dot(&a);
            let correction = &at_p_b * r_pb_inv; // (4, 1)
            let correction = correction.dot(&bt_p_a); // (4, 4)
            p_mat = &q + &at_p_a - &correction;
        }

        // K = (R + B^T P B)^-1 B^T P A, row vector (1, 4).
        let bt_p = b_col.t().dot(&p_mat);
        let bt_p_b = bt_p.dot(&b_col)[[0, 0]];
        let k_scale = 1.0 / (r_mat[[0, 0]] + bt_p_b);
        let bt_p_a = bt_p.dot(&a);
        let k_row: Array1<f64> = bt_p_a.row(0).to_owned() * k_scale;
        Self {
            k: [k_row[0], k_row[1], k_row[2], k_row[3]],
        }
    }

    pub fn act(&self, obs: [f64; 4]) -> f64 {
        -(self.k[0] * obs[0] + self.k[1] * obs[1] + self.k[2] * obs[2] + self.k[3] * obs[3])
    }
}

fn euler_discretize(
    a_c: &[[f64; 4]; 4],
    b_c: &[f64; 4],
    dt: f64,
) -> ([[f64; 4]; 4], [f64; 4]) {
    let mut a_d = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a_d[i][j] = if i == j { 1.0 } else { 0.0 } + a_c[i][j] * dt;
        }
    }
    let mut b_d = [0.0; 4];
    for i in 0..4 {
        b_d[i] = b_c[i] * dt;
    }
    (a_d, b_d)
}

/// Hand-tuned two-loop PD controller. Kept as a separate class so the
/// demo can show both a black-box tuned controller and the principled LQR.
pub struct Pid {
    pub k_theta: f64,
    pub k_theta_dot: f64,
    pub k_x: f64,
    pub k_x_dot: f64,
}

impl Default for Pid {
    /// Gains tuned by hand against the default `Params`. Angle loop only —
    /// we don't servo cart position. That's deliberate: the controller stays
    /// PID-on-the-primary-loop in spirit, and the cart drift becomes the
    /// visible contrast against LQR's full-state control.
    ///
    /// Sign convention follows the env: +θ tilts right, so positive θ needs
    /// positive `F_cmd` (push cart right) to rotate the rod back toward
    /// upright. Cart gains are small: enough to damp large excursions, not
    /// enough to servo `x → 0`.
    fn default() -> Self {
        Self {
            k_theta: 80.0,
            k_theta_dot: 15.0,
            k_x: 0.6,
            k_x_dot: 1.2,
        }
    }
}

impl Pid {
    pub fn act(&self, obs: [f64; 4]) -> f64 {
        let [x, x_dot, theta, theta_dot] = obs;
        self.k_theta * theta + self.k_theta_dot * theta_dot + self.k_x * x + self.k_x_dot * x_dot
    }
}
