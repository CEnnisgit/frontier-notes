//! Sanity tests for the pendulum-on-cart environment.

use rl_control_toy::env::{default_params, linearize_cartpole, Env, EnvParams};

fn free_params() -> EnvParams {
    // k_force = 0 disables the actuator → total mechanical energy must be
    // conserved whatever voltage is commanded.
    EnvParams {
        k_force: 0.0,
        dt: 1e-3,
        ..default_params()
    }
}

#[test]
fn energy_conserves_with_no_actuator_force() {
    let mut env = Env::new(free_params(), 0);
    let theta0 = 15f64.to_radians();
    env.reset([0.0, 0.0, theta0, 0.0]);
    let e0 = env.energy();
    for _ in 0..2000 {
        env.step(0.0);
    }
    let e = env.energy();
    let drift = (e - e0).abs() / e0.abs();
    assert!(
        drift < 1e-4,
        "energy drift over 2 s was {:.3e} (e0={:.6}, e={:.6})",
        drift,
        e0,
        e
    );
}

#[test]
fn linearization_matches_nonlinear_near_upright() {
    // Near θ=0 with zero velocities, one RK4 step of the full nonlinear
    // system should match the forward-Euler step on the linearized plant
    // to high precision. (Actuator off so F = 0, no bias term.)
    let params = free_params();
    let (a, _b) = linearize_cartpole(&params);

    let theta0 = 1e-3;
    let mut env = Env::new(params.clone(), 0);
    env.reset([0.0, 0.0, theta0, 0.0]);
    env.step(0.0);
    let nonlinear = env.true_state_xxdtt();

    let s0 = [0.0, 0.0, theta0, 0.0];
    let mut lin = [0.0; 4];
    for i in 0..4 {
        let mut row = 0.0;
        for j in 0..4 {
            row += a[i][j] * s0[j];
        }
        lin[i] = s0[i] + params.dt * row;
    }

    for i in 0..4 {
        let err = (nonlinear[i] - lin[i]).abs();
        // dt=1e-3, θ=1e-3, so the leading neglected term is O(dt²·θ + dt·θ²) ≈ 1e-9.
        assert!(
            err < 1e-6,
            "linearization mismatch on state[{i}]: nonlinear={:.6e}, linear={:.6e}",
            nonlinear[i],
            lin[i]
        );
    }
}

#[test]
fn actuator_saturates_and_coil_approaches_steady_state() {
    // Command huge voltage; voltage saturates at v_sat. Coil current should
    // approach v_sat (since k_force=1 → di/dt = (v_sat - i)/tau) after
    // several time constants.
    let params = EnvParams {
        v_sat: 10.0,
        tau_coil: 0.02,
        k_force: 1.0,
        dt: 1e-3,
        ..default_params()
    };
    let mut env = Env::new(params.clone(), 0);
    env.reset([0.0, 0.0, 0.0, 0.0]);
    let big = 1e6;
    let n_steps = (10.0 * params.tau_coil / params.dt) as usize;
    for _ in 0..n_steps {
        env.step(big);
    }
    let i = env.coil_current();
    assert!(
        (i - params.v_sat).abs() < 1e-3,
        "coil current {:.6} did not reach v_sat={}",
        i,
        params.v_sat
    );
}

#[test]
fn noise_is_zero_mean() {
    // k_force=0 → cart free. With pole at θ=0, θ_dot=0 and cart at rest, the
    // full state stays frozen at the initial configuration, so any deviation
    // in observations is the Gaussian measurement noise.
    let params = EnvParams {
        obs_noise: [1e-2, 1e-2, 1e-2, 1e-2],
        k_force: 0.0,
        ..default_params()
    };
    let mut env = Env::new(params.clone(), 42);
    let true_state = [0.1, 0.0, 0.0, 0.0];
    env.reset(true_state);
    let n = 2000usize;
    let mut acc = [0.0; 4];
    for _ in 0..n {
        let obs = env.step(0.0);
        for j in 0..4 {
            acc[j] += obs[j] - true_state[j];
        }
    }
    for j in 0..4 {
        let mean = acc[j] / n as f64;
        let sigma = params.obs_noise[j] / (n as f64).sqrt();
        assert!(
            mean.abs() < 4.0 * sigma,
            "dim {j}: empirical mean {:.3e} exceeds 4·σ_emp = {:.3e}",
            mean,
            4.0 * sigma
        );
    }
    // Sanity: true state really did stay pinned.
    let final_state = env.true_state_xxdtt();
    for j in 0..4 {
        assert!(
            (final_state[j] - true_state[j]).abs() < 1e-10,
            "true state drifted on dim {j}: {:.3e}",
            final_state[j] - true_state[j]
        );
    }
}
