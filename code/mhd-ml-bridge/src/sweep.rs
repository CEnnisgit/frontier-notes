//! Parameter sweep harness for Week 5 — Q10.
//!
//! Takes a trained MLP + the same Stats used for normalization, and evaluates
//! per-field rMSE/range as one physical knob widens past the Week-4 training
//! envelope. The other knobs are sampled from the Week-4 distributions so
//! failure modes attribute cleanly to the widened axis.

use crate::data::{
    run_one, run_one_with_gamma, sample_params, GAMMA, N_FEATURES, N_OUT, N_OUTPUTS,
};
use crate::surrogate::{per_field_rmse_over_range, predict, Mlp, Stats};
use anyhow::Result;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepAxis {
    /// Right-state density. Training was `[0.10, 0.15]`; sweep widens to
    /// `{0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40}`.
    RhoRatio,
    /// Flip the sign of both `B_y_L` and `B_y_R` (training was
    /// `By_L > 0`, `By_R < 0`).
    BySignFlip,
    /// Heat-capacity ratio γ. Training was a fixed `γ = 2.0`; sweep probes
    /// `{1.4, 1.67, 2.0, 2.5, 3.0}`.
    Gamma,
    /// Guide-field `B_x`. Training was `[0.60, 0.90]`; sweep probes
    /// `{0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5}`.
    Bx,
}

impl SweepAxis {
    pub fn name(&self) -> &'static str {
        match self {
            SweepAxis::RhoRatio => "rho_ratio",
            SweepAxis::BySignFlip => "by_sign_flip",
            SweepAxis::Gamma => "gamma",
            SweepAxis::Bx => "bx",
        }
    }

    pub fn all() -> &'static [SweepAxis] {
        &[
            SweepAxis::RhoRatio,
            SweepAxis::BySignFlip,
            SweepAxis::Gamma,
            SweepAxis::Bx,
        ]
    }

    pub fn values(&self) -> Vec<f64> {
        match self {
            SweepAxis::RhoRatio => vec![0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40],
            SweepAxis::BySignFlip => vec![0.0, 1.0],
            SweepAxis::Gamma => vec![1.4, 1.67, 2.0, 2.5, 3.0],
            SweepAxis::Bx => vec![0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5],
        }
    }
}

#[derive(Debug, Clone)]
pub struct SweepPoint {
    pub value: f64,
    pub per_field_rmse_over_range: [f32; 4],
}

/// Pin the swept knob to `value`; sample the rest from the Week-4
/// distributions via `sample_params`. Returns the 7-parameter IC vector and
/// the effective γ (only non-default for `SweepAxis::Gamma`).
fn build_params_for_axis(
    axis: SweepAxis,
    value: f64,
    rng: &mut StdRng,
) -> ([f32; N_FEATURES], f64) {
    let mut p = sample_params(rng);
    let mut gamma = GAMMA;
    match axis {
        SweepAxis::RhoRatio => {
            p[3] = value as f32; // rho_R
        }
        SweepAxis::BySignFlip => {
            if value > 0.5 {
                p[2] = -p[2]; // By_L
                p[5] = -p[5]; // By_R
            }
        }
        SweepAxis::Gamma => {
            gamma = value;
        }
        SweepAxis::Bx => {
            p[6] = value as f32;
        }
    }
    (p, gamma)
}

fn truth_row(params: &[f32; N_FEATURES], gamma: f64) -> [f32; N_OUTPUTS] {
    let out = if (gamma - GAMMA).abs() < 1e-12 {
        run_one(params)
    } else {
        run_one_with_gamma(params, gamma)
    };
    let mut flat = [0.0f32; N_OUTPUTS];
    for f in 0..4 {
        for j in 0..N_OUT {
            flat[f * N_OUT + j] = out[[f, j]] as f32;
        }
    }
    flat
}

/// Evaluate the MLP across all sweep values of `axis`. Draws `n_per_point`
/// samples at each value; seeds are deterministic from `seed`.
pub fn run_sweep(
    mlp: &Mlp,
    stats: &Stats,
    axis: SweepAxis,
    n_per_point: usize,
    seed: u64,
) -> Result<Vec<SweepPoint>> {
    let values = axis.values();
    let mut points = Vec::with_capacity(values.len());
    for (i, v) in values.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
        let mut input_rows: Vec<[f32; N_FEATURES]> = Vec::with_capacity(n_per_point);
        let mut truth_rows: Vec<[f32; N_OUTPUTS]> = Vec::with_capacity(n_per_point);
        for _ in 0..n_per_point {
            let (p, gamma) = build_params_for_axis(axis, *v, &mut rng);
            truth_rows.push(truth_row(&p, gamma));
            input_rows.push(p);
        }
        let pred_rows = predict(mlp, stats, &input_rows)?;
        let (rmse, _) = per_field_rmse_over_range(&truth_rows, &pred_rows);
        points.push(SweepPoint {
            value: *v,
            per_field_rmse_over_range: rmse,
        });
    }
    Ok(points)
}

/// Worst-case-ish example for plotting: picks the sweep value with the
/// largest mean per-field rMSE on `axis`, re-samples one realization, and
/// returns `(truth_row, pred_row, params, sweep_value)`.
pub fn worst_case(
    mlp: &Mlp,
    stats: &Stats,
    axis: SweepAxis,
    n_per_point: usize,
    seed: u64,
) -> Result<([f32; N_OUTPUTS], [f32; N_OUTPUTS], [f32; N_FEATURES], f64)> {
    let sweep = run_sweep(mlp, stats, axis, n_per_point, seed)?;
    let worst = sweep
        .iter()
        .max_by(|a, b| {
            let ma: f32 = a.per_field_rmse_over_range.iter().sum();
            let mb: f32 = b.per_field_rmse_over_range.iter().sum();
            ma.partial_cmp(&mb).unwrap()
        })
        .expect("non-empty sweep");
    let mut rng = StdRng::seed_from_u64(seed ^ 0xDEAD_BEEF);
    let (p, gamma) = build_params_for_axis(axis, worst.value, &mut rng);
    let truth = truth_row(&p, gamma);
    let pred_rows = predict(mlp, stats, &[p])?;
    Ok((truth, pred_rows[0], p, worst.value))
}
