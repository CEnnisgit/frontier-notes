//! Sanity tests for the cart-pole dynamics.

use rl_control_toy::env::{linearize_upright, Env, Params, State};

/// The hanging-down equilibrium (θ = π) is a physical stable point: release
/// the pendulum from rest with no input and mechanical energy should be
/// conserved to RK4 order over a few seconds.
#[test]
fn energy_conserved_for_free_pendulum() {
    let p = Params::default();
    let s0 = State {
        x: 0.0,
        x_dot: 0.0,
        theta: std::f64::consts::PI + 0.2, // slightly off hanging-down
        theta_dot: 0.0,
        force: 0.0,
    };
    let mut env = Env::new(p, s0);
    let e0 = env.energy();
    for _ in 0..400 {
        env.step(0.0);
    }
    let e1 = env.energy();
    let rel = ((e1 - e0) / e0).abs();
    assert!(rel < 1e-3, "RK4 energy drift {rel:.3e} exceeds 1e-3");
}

/// For small θ, the nonlinear dynamics should match the linearization.
/// We integrate the nonlinear env and the linear ODE analytically-ish
/// (same RK4 on the linear RHS) from the same IC for a short horizon and
/// require the trajectories match to a few percent.
#[test]
fn nonlinear_matches_linearization_near_upright() {
    let p = Params::default();
    let (a, b) = linearize_upright(&p);

    let theta0 = 0.01; // ≈ 0.57°
    let mut env = Env::new(
        p,
        State {
            x: 0.0,
            x_dot: 0.0,
            theta: theta0,
            theta_dot: 0.0,
            force: 0.0,
        },
    );

    // Linear state vector (no actuator lag — pass F directly).
    let mut z = [0.0, 0.0, theta0, 0.0];
    let f_const = 0.0;

    // Upright is unstable (λ ≈ 4.65 rad/s), so the trajectory doubles every
    // ~150 ms. Keep the horizon short enough that O(θ³) terms stay small.
    let n_steps = 30;
    for _ in 0..n_steps {
        env.step(f_const);
        z = rk4_linear(&a, &b, &z, f_const, p.dt);
    }

    let state = env.s;
    let err_theta = (state.theta - z[2]).abs();
    let err_x = (state.x - z[0]).abs();
    assert!(err_theta < 5e-3, "θ mismatch: {err_theta:.3e}");
    assert!(err_x < 5e-3, "x mismatch: {err_x:.3e}");
}

/// Actuator saturation: commanding far above `f_max` should leave `F`
/// approaching the cap rather than blowing past it.
#[test]
fn actuator_saturation_holds() {
    let p = Params::default();
    let mut env = Env::new(p, State::upright(0.0));
    for _ in 0..200 {
        env.step(1e6);
    }
    assert!(env.s.force <= p.f_max + 1e-6);
    assert!(env.s.force >= 0.99 * p.f_max);
}

fn rk4_linear(
    a: &[[f64; 4]; 4],
    b: &[f64; 4],
    z: &[f64; 4],
    u: f64,
    dt: f64,
) -> [f64; 4] {
    let f = |z: &[f64; 4]| -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i] += a[i][j] * z[j];
            }
            out[i] += b[i] * u;
        }
        out
    };
    let k1 = f(z);
    let z2 = [
        z[0] + 0.5 * dt * k1[0],
        z[1] + 0.5 * dt * k1[1],
        z[2] + 0.5 * dt * k1[2],
        z[3] + 0.5 * dt * k1[3],
    ];
    let k2 = f(&z2);
    let z3 = [
        z[0] + 0.5 * dt * k2[0],
        z[1] + 0.5 * dt * k2[1],
        z[2] + 0.5 * dt * k2[2],
        z[3] + 0.5 * dt * k2[3],
    ];
    let k3 = f(&z3);
    let z4 = [
        z[0] + dt * k3[0],
        z[1] + dt * k3[1],
        z[2] + dt * k3[2],
        z[3] + dt * k3[3],
    ];
    let k4 = f(&z4);
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = z[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    out
}
