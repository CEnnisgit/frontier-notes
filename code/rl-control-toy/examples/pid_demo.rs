//! Week 7 — PID baseline demo.
//!
//! Three scenarios side-by-side, all stabilizing upright:
//!   (a) clean          — perfect state feedback, no disturbance
//!   (b) noisy          — Gaussian state-measurement noise
//!   (c) perturbed      — mid-episode impulse on the cart
//!
//! Produces `figures/pid.png`. Also runs the LQR controller in each
//! scenario as a second series so we can see hand-tuned PD vs principled
//! LQR on the same plant.
//!
//! `cargo run -p rl-control-toy --example pid_demo --release`

use anyhow::{Context, Result};
use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rl_control_toy::env::{Env, NoiseSpec, Params, State};
use rl_control_toy::pid::{Lqr, Pid};
use std::path::{Path, PathBuf};

const T_FINAL: f64 = 5.0;
const NOISE_SEED: u64 = 7;

fn main() -> Result<()> {
    let p = Params::default();
    let pid = Pid::default();
    let lqr = Lqr::design(&p, [1.0, 1.0, 20.0, 2.0], 0.2, p.dt);
    println!(
        "LQR gains K = [{:.3}, {:.3}, {:.3}, {:.3}]",
        lqr.k[0], lqr.k[1], lqr.k[2], lqr.k[3]
    );

    let clean_noise = NoiseSpec::default();
    let measurement_noise = NoiseSpec {
        x_sigma: 0.005,
        x_dot_sigma: 0.05,
        theta_sigma: 0.01,
        theta_dot_sigma: 0.1,
    };

    let pid_clean = simulate(&p, &pid, &clean_noise, None);
    let lqr_clean = simulate(&p, &lqr, &clean_noise, None);

    let pid_noisy = simulate(&p, &pid, &measurement_noise, None);
    let lqr_noisy = simulate(&p, &lqr, &measurement_noise, None);

    let pid_pert = simulate(&p, &pid, &clean_noise, Some((2.0, 2.0, 0.0)));
    let lqr_pert = simulate(&p, &lqr, &clean_noise, Some((2.0, 2.0, 0.0)));

    report("clean", &pid_clean, &lqr_clean);
    report("noisy", &pid_noisy, &lqr_noisy);
    report("perturbed", &pid_pert, &lqr_pert);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("figures");
    std::fs::create_dir_all(&out_dir)?;
    let out = out_dir.join("pid.png");
    plot_three_panel(
        &[
            ("clean", &pid_clean, &lqr_clean),
            ("noisy", &pid_noisy, &lqr_noisy),
            ("perturbed (impulse at t=2.0 s)", &pid_pert, &lqr_pert),
        ],
        &out,
    )?;

    Ok(())
}

trait Controller {
    fn act(&self, obs: [f64; 4]) -> f64;
}

impl Controller for Pid {
    fn act(&self, obs: [f64; 4]) -> f64 {
        Pid::act(self, obs)
    }
}

impl Controller for Lqr {
    fn act(&self, obs: [f64; 4]) -> f64 {
        Lqr::act(self, obs)
    }
}

struct Trace {
    t: Vec<f64>,
    theta: Vec<f64>,
    x: Vec<f64>,
    f_cmd: Vec<f64>,
    /// RMS angle in radians over the simulation — summary metric.
    theta_rms: f64,
    /// Did |θ| ever exceed 45° ≈ 0.785 rad (a liberal "fell over" threshold)?
    fell: bool,
}

/// Run one rollout; optionally apply a cart-velocity impulse `(t_impulse,
/// dx_dot, dtheta_dot)` at the first step after `t_impulse`.
fn simulate<C: Controller>(
    p: &Params,
    ctrl: &C,
    noise: &NoiseSpec,
    impulse: Option<(f64, f64, f64)>,
) -> Trace {
    let n = (T_FINAL / p.dt).round() as usize;
    let mut env = Env::new(*p, State::upright(0.05));
    let mut rng = StdRng::seed_from_u64(NOISE_SEED);
    let mut t = Vec::with_capacity(n + 1);
    let mut theta = Vec::with_capacity(n + 1);
    let mut x = Vec::with_capacity(n + 1);
    let mut f_cmd_vec = Vec::with_capacity(n + 1);
    let mut impulse_done = impulse.is_none();

    for _ in 0..n {
        if let Some((t_p, dx_dot, dth_dot)) = impulse {
            if !impulse_done && env.t >= t_p {
                env.perturb_impulse(dx_dot, dth_dot);
                impulse_done = true;
            }
        }
        let obs = env.observe(noise, &mut rng);
        let f_cmd = ctrl.act(obs);
        t.push(env.t);
        theta.push(env.s.theta);
        x.push(env.s.x);
        f_cmd_vec.push(f_cmd);
        env.step(f_cmd);
    }
    // final snapshot
    t.push(env.t);
    theta.push(env.s.theta);
    x.push(env.s.x);
    f_cmd_vec.push(*f_cmd_vec.last().unwrap_or(&0.0));

    let theta_rms = (theta.iter().map(|v| v * v).sum::<f64>() / theta.len() as f64).sqrt();
    let fell = theta.iter().any(|v| v.abs() > 0.785);

    Trace {
        t,
        theta,
        x,
        f_cmd: f_cmd_vec,
        theta_rms,
        fell,
    }
}

fn report(label: &str, pid: &Trace, lqr: &Trace) {
    println!(
        "  {:<26}  PID θ_rms={:.4} rad  fell={:<5}  |  LQR θ_rms={:.4} rad  fell={}",
        label, pid.theta_rms, pid.fell, lqr.theta_rms, lqr.fell
    );
}

fn plot_three_panel(panels: &[(&str, &Trace, &Trace)], out: &Path) -> Result<()> {
    let backend = BitMapBackend::new(out, (1500, 900)).into_drawing_area();
    backend.fill(&WHITE)?;
    let titled = backend.titled(
        "Week 7 — PID (red) and LQR (blue) stabilizing upright cart-pole",
        ("sans-serif", 20).into_font(),
    )?;
    let columns = titled.split_evenly((1, panels.len()));

    for (area, (label, pid, lqr)) in columns.iter().zip(panels.iter()) {
        let rows = area.split_evenly((3, 1));

        plot_series(
            &rows[0],
            &format!("{label}: theta (rad)"),
            "t [s]",
            "θ",
            &pid.t,
            &[("PID", &pid.theta, RED), ("LQR", &lqr.theta, BLUE)],
            Some((&lqr.t, &lqr.theta)),
        )?;
        plot_series(
            &rows[1],
            "cart position x (m)",
            "t [s]",
            "x",
            &pid.t,
            &[("PID", &pid.x, RED), ("LQR", &lqr.x, BLUE)],
            Some((&lqr.t, &lqr.x)),
        )?;
        plot_series(
            &rows[2],
            "commanded force F_cmd (N)",
            "t [s]",
            "F",
            &pid.t,
            &[("PID", &pid.f_cmd, RED), ("LQR", &lqr.f_cmd, BLUE)],
            Some((&lqr.t, &lqr.f_cmd)),
        )?;
    }

    backend.present().context("render pid demo figure")?;
    println!("wrote {}", out.display());
    Ok(())
}

fn plot_series(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    caption: &str,
    x_label: &str,
    y_label: &str,
    t: &[f64],
    series: &[(&str, &Vec<f64>, RGBColor)],
    _lqr_override: Option<(&Vec<f64>, &Vec<f64>)>,
) -> Result<()> {
    let all = series.iter().flat_map(|(_, v, _)| v.iter().copied());
    let mut y_min = all.clone().fold(f64::INFINITY, f64::min);
    let mut y_max = all.fold(f64::NEG_INFINITY, f64::max);
    let pad = (y_max - y_min).abs().max(1e-3) * 0.1;
    y_min -= pad;
    y_max += pad;
    let x_max = *t.last().unwrap();

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 14).into_font())
        .margin(8)
        .x_label_area_size(25)
        .y_label_area_size(45)
        .build_cartesian_2d(0f64..x_max, y_min..y_max)?;
    chart.configure_mesh().x_desc(x_label).y_desc(y_label).draw()?;

    for (name, values, color) in series {
        let pts: Vec<(f64, f64)> = t.iter().zip(values.iter()).map(|(&a, &b)| (a, b)).collect();
        chart
            .draw_series(LineSeries::new(pts, color.stroke_width(2)))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 15, y)], color.stroke_width(2)));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    Ok(())
}
