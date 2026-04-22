//! Classical controllers for the upright-stabilization regime.
//!
//! - `Pid`: the scalar integrator with anti-windup.
//! - `UprightController`: cascade of two PIDs — outer loop on cart position
//!   asks for a small pole-angle target, inner loop on angle commands force.
//! - `Lqr`: gain computed by iterating the discrete algebraic Riccati equation
//!   on the linearized cart-pole (actuator dynamics not modeled — the coil lag
//!   is short enough that the linearized plant is a good-enough approximation
//!   for this toy).
//!
//! Both controllers output a commanded voltage; the env's saturation is the
//! ground truth on actuator limits.

use crate::env::{linearize_cartpole, EnvParams};

#[derive(Clone, Debug)]
pub struct Pid {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub i_clamp: f64,
    integral: f64,
}

impl Pid {
    pub fn new(kp: f64, ki: f64, kd: f64, i_clamp: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            i_clamp,
            integral: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
    }

    /// Standard PID on `err` with a pre-computed derivative `d_err` (so callers
    /// can feed in measured velocity rather than suffer numerical derivative
    /// noise).
    pub fn step(&mut self, err: f64, d_err: f64, dt: f64) -> f64 {
        self.integral = (self.integral + err * dt).clamp(-self.i_clamp, self.i_clamp);
        self.kp * err + self.ki * self.integral + self.kd * d_err
    }
}

/// Cascade upright stabilizer.
/// Outer loop: cart position → small pole-angle target.
/// Inner loop: pole angle → cart force → voltage (via `1/k_force`).
pub struct UprightController {
    theta_pid: Pid,
    x_to_theta_kp: f64,
    x_to_theta_kd: f64,
    k_force: f64,
    v_sat: f64,
}

impl UprightController {
    pub fn tuned(params: &EnvParams) -> Self {
        Self {
            theta_pid: Pid::new(60.0, 20.0, 10.0, 5.0),
            x_to_theta_kp: 0.08,
            x_to_theta_kd: 0.15,
            k_force: params.k_force,
            v_sat: params.v_sat,
        }
    }

    pub fn reset(&mut self) {
        self.theta_pid.reset();
    }

    /// `obs = [x, x_dot, theta, theta_dot]`. Returns a commanded voltage.
    pub fn command(&mut self, obs: [f64; 4], dt: f64) -> f64 {
        let [x, xd, th, thd] = obs;
        let theta_target = -self.x_to_theta_kp * x - self.x_to_theta_kd * xd;
        let err = th - theta_target;
        let d_err = thd;
        let force = self.theta_pid.step(err, d_err, dt);
        (force / self.k_force).clamp(-self.v_sat, self.v_sat)
    }
}

pub struct Lqr {
    k: [f64; 4],
    k_force: f64,
    v_sat: f64,
}

impl Lqr {
    /// Compute the LQR gain by iterating the discrete Riccati equation on the
    /// forward-Euler discretization of the linearized plant. Converges in a
    /// few hundred iterations for this 4-state system.
    pub fn tuned(params: &EnvParams) -> Self {
        let (a, b) = linearize_cartpole(params);
        let dt = 0.01;
        let ad = add_mat(identity4(), scale_mat(a, dt));
        let bd = scale_vec(b, dt);

        let q = diag([1.0, 0.1, 10.0, 1.0]);
        let r = 0.01_f64;

        let mut p = q;
        for _ in 0..2000 {
            let p_next = riccati_step(&ad, &bd, &p, &q, r);
            if max_abs_diff(&p, &p_next) < 1e-10 {
                p = p_next;
                break;
            }
            p = p_next;
        }

        let btp = vt_mat(&bd, &p);
        let scalar = r + dot4(&btp, &bd);
        let btpa = vt_mat_times_mat(&bd, &p, &ad);
        let k = [
            btpa[0] / scalar,
            btpa[1] / scalar,
            btpa[2] / scalar,
            btpa[3] / scalar,
        ];

        Self {
            k,
            k_force: params.k_force,
            v_sat: params.v_sat,
        }
    }

    pub fn gain(&self) -> [f64; 4] {
        self.k
    }

    pub fn command(&self, obs: [f64; 4]) -> f64 {
        let force = -(self.k[0] * obs[0]
            + self.k[1] * obs[1]
            + self.k[2] * obs[2]
            + self.k[3] * obs[3]);
        (force / self.k_force).clamp(-self.v_sat, self.v_sat)
    }
}

// ---------- tiny 4x4 / 4-vector helpers (avoids pulling a linalg crate) ----

type Mat4 = [[f64; 4]; 4];
type Vec4 = [f64; 4];

fn identity4() -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        m[i][i] = 1.0;
    }
    m
}

fn diag(d: Vec4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        m[i][i] = d[i];
    }
    m
}

fn add_mat(a: Mat4, b: Mat4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j] + b[i][j];
        }
    }
    m
}

fn scale_mat(a: Mat4, s: f64) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j] * s;
        }
    }
    m
}

fn scale_vec(v: Vec4, s: f64) -> Vec4 {
    [v[0] * s, v[1] * s, v[2] * s, v[3] * s]
}

fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            m[i][j] = s;
        }
    }
    m
}

fn transpose(a: &Mat4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[j][i];
        }
    }
    m
}

fn dot4(a: &Vec4, b: &Vec4) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// v^T * M → row vector (returned as Vec4 representing the row).
fn vt_mat(v: &Vec4, m: &Mat4) -> Vec4 {
    let mut r = [0.0; 4];
    for j in 0..4 {
        let mut s = 0.0;
        for i in 0..4 {
            s += v[i] * m[i][j];
        }
        r[j] = s;
    }
    r
}

/// v^T * M1 * M2 → 1x4 row.
fn vt_mat_times_mat(v: &Vec4, m1: &Mat4, m2: &Mat4) -> Vec4 {
    let row = vt_mat(v, m1);
    vt_mat(&row, m2)
}

fn outer(a: &Vec4, b: &Vec4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i] * b[j];
        }
    }
    m
}

fn max_abs_diff(a: &Mat4, b: &Mat4) -> f64 {
    let mut m: f64 = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

/// Discrete Riccati: P = A'PA - (A'PB)(R + B'PB)^{-1}(B'PA) + Q.
fn riccati_step(a: &Mat4, b: &Vec4, p: &Mat4, q: &Mat4, r: f64) -> Mat4 {
    let at = transpose(a);
    let atp = mat_mul(&at, p);
    let atpa = mat_mul(&atp, a);

    // A' P B → 4-vector (column).
    let mut atpb = [0.0; 4];
    for i in 0..4 {
        let mut s = 0.0;
        for k in 0..4 {
            s += atp[i][k] * b[k];
        }
        atpb[i] = s;
    }
    // B' P A → 4-vector (row, same values since P symmetric + transpose).
    let btpa = vt_mat_times_mat(b, p, a);
    // B' P B → scalar.
    let btp = vt_mat(b, p);
    let btpb = dot4(&btp, b);
    let scalar = r + btpb;

    let correction = scale_mat(outer(&atpb, &btpa), 1.0 / scalar);
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = atpa[i][j] - correction[i][j] + q[i][j];
        }
    }
    out
}
