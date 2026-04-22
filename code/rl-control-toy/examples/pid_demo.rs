//! Upright-stabilization demo: PID cascade vs. LQR across three regimes.
//!
//! `cargo run -p rl-control-toy --example pid_demo --release`
//!
//! Writes a 3-panel figure to `figures/pid.png`. Each panel shows θ(t) for
//! both controllers under one of: (a) clean observations, (b) noisy
//! observations, (c) clean observations with a cart-impulse disturbance at
//! t=2.5 s.

use anyhow::Result;
use plotters::prelude::*;
use rl_control_toy::env::{default_params, Env, EnvParams};
use rl_control_toy::pid::{Lqr, UprightController};
use std::path::PathBuf;

const T_SIM: f64 = 5.0;
const THETA0: f64 = 0.10; // ~5.7° initial tilt

#[derive(Clone, Copy)]
enum Scenario {
    Clean,
    Noisy,
    Perturbed,
}

impl Scenario {
    fn title(&self) -> &'static str {
        match self {
            Scenario::Clean => "(a) Clean observations",
            Scenario::Noisy => "(b) Noisy observations",
            Scenario::Perturbed => "(c) Cart impulse at t=2.5 s",
        }
    }

    fn params(&self, base: &EnvParams) -> EnvParams {
        match self {
            Scenario::Noisy => EnvParams {
                obs_noise: [1e-3, 5e-3, 2e-3, 1e-2],
                ..base.clone()
            },
            _ => base.clone(),
        }
    }

    fn impulse_at(&self, t: f64, dt: f64) -> Option<f64> {
        if matches!(self, Scenario::Perturbed) && (t - 2.5).abs() < dt * 0.5 {
            Some(1.0)
        } else {
            None
        }
    }
}

enum ControllerKind {
    Pid,
    Lqr,
}

fn run(
    scenario: Scenario,
    kind: ControllerKind,
    base: &EnvParams,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let p = scenario.params(base);
    let mut env = Env::new(p.clone(), seed);
    env.reset([0.0, 0.0, THETA0, 0.0]);

    let mut pid = UprightController::tuned(&p);
    pid.reset();
    let lqr = Lqr::tuned(&p);

    let n = (T_SIM / p.dt) as usize;
    let mut ts = Vec::with_capacity(n);
    let mut ths = Vec::with_capacity(n);

    let mut obs = [0.0, 0.0, THETA0, 0.0];
    for k in 0..n {
        let t = k as f64 * p.dt;
        if let Some(imp) = scenario.impulse_at(t, p.dt) {
            env.apply_impulse(imp);
        }
        let v = match kind {
            ControllerKind::Pid => pid.command(obs, p.dt),
            ControllerKind::Lqr => lqr.command(obs),
        };
        obs = env.step(v);
        ts.push(t);
        ths.push(env.true_state_xxdtt()[2]); // plot *true* θ so both controllers get the same yardstick
    }
    (ts, ths)
}

fn main() -> Result<()> {
    let base = default_params();
    let scenarios = [Scenario::Clean, Scenario::Noisy, Scenario::Perturbed];

    let mut results: Vec<(Scenario, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
    for sc in scenarios {
        let (ts, pid_th) = run(sc, ControllerKind::Pid, &base, 1);
        let (_, lqr_th) = run(sc, ControllerKind::Lqr, &base, 1);
        results.push((sc, ts, pid_th, lqr_th));
    }

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("figures");
    std::fs::create_dir_all(&out_dir)?;
    let out = out_dir.join("pid.png");

    let root = BitMapBackend::new(&out, (1400, 420)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 3));

    for (panel, (sc, ts, pid_th, lqr_th)) in panels.iter().zip(results.iter()) {
        let y_max = pid_th
            .iter()
            .chain(lqr_th.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.15);
        let y_min = pid_th
            .iter()
            .chain(lqr_th.iter())
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(-0.15);

        let mut chart = ChartBuilder::on(panel)
            .caption(sc.title(), ("sans-serif", 18).into_font())
            .margin(12)
            .x_label_area_size(36)
            .y_label_area_size(52)
            .build_cartesian_2d(0f64..T_SIM, y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc("t [s]")
            .y_desc("θ [rad]")
            .draw()?;

        // θ=0 upright reference.
        chart.draw_series(LineSeries::new(
            [(0.0, 0.0), (T_SIM, 0.0)],
            BLACK.mix(0.4).stroke_width(1),
        ))?;

        chart
            .draw_series(LineSeries::new(
                ts.iter().zip(pid_th.iter()).map(|(&t, &y)| (t, y)),
                BLUE.stroke_width(2),
            ))?
            .label("PID cascade")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE.stroke_width(2)));
        chart
            .draw_series(LineSeries::new(
                ts.iter().zip(lqr_th.iter()).map(|(&t, &y)| (t, y)),
                RED.stroke_width(2),
            ))?
            .label("LQR")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.85))
            .draw()?;
    }

    root.present()?;
    println!("wrote {}", out.display());
    Ok(())
}
