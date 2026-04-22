//! Week 6 — Q11. DeepONet vs wide-MLP, both trained on the widened distribution.
//!
//! `cargo run -p mhd-ml-bridge --example train_operator --release`
//!
//! Produces:
//!   * `figures/operator/training_curve.png`  — operator + MLP loss histories
//!   * `figures/operator/val_example.png`     — 4-field prediction panel
//!   * `figures/operator/sweep_compare.png`   — per-axis rMSE/range, both models

use anyhow::{Context, Result};
use mhd_ml_bridge::data::{N_FEATURES, N_OUT, N_OUTPUTS};
use mhd_ml_bridge::operator::{predict_operator, train_operator_wide};
use mhd_ml_bridge::surrogate::{per_field_rmse_over_range, train_model_wide, TrainResult};
use mhd_ml_bridge::sweep::{run_sweep, SweepAxis, SweepPoint};
use plotters::prelude::*;
use std::path::{Path, PathBuf};

const N_PER_POINT: usize = 16;
const SWEEP_SEED: u64 = 42;

fn main() -> Result<()> {
    println!("\n=== Week 6 Q11: DeepONet vs wide-MLP ===\n");

    let (op, _op_vm, op_stats, op_tr) = train_operator_wide()?;
    println!(
        "DeepONet val per-field rMSE/range: rho={:.3} u={:.3} By={:.3} p={:.3}",
        op_tr.field_rmse_over_range[0],
        op_tr.field_rmse_over_range[1],
        op_tr.field_rmse_over_range[2],
        op_tr.field_rmse_over_range[3],
    );

    let (mlp, _mlp_vm, mlp_stats, mlp_tr) = train_model_wide()?;
    println!(
        "wide MLP val per-field rMSE/range:  rho={:.3} u={:.3} By={:.3} p={:.3}",
        mlp_tr.field_rmse_over_range[0],
        mlp_tr.field_rmse_over_range[1],
        mlp_tr.field_rmse_over_range[2],
        mlp_tr.field_rmse_over_range[3],
    );

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("figures")
        .join("operator");
    std::fs::create_dir_all(&out_dir)?;

    plot_training_curves(&op_tr, &mlp_tr, &out_dir.join("training_curve.png"))?;
    plot_val_example_compare(&op_tr, &mlp_tr, &out_dir.join("val_example.png"))?;

    // Compare both models across the Q10 sweep axes (same harness, same seeds).
    // `run_sweep` is generic over any (model, Stats, predict_fn), but the current
    // signature takes `&Mlp` — so we do the operator side via direct `predict`
    // calls below, mirroring what `run_sweep` does internally.
    println!("\nrunning Q10 sweeps on both models (n_per_point={N_PER_POINT})");
    let mut mlp_results: Vec<(SweepAxis, Vec<SweepPoint>)> = Vec::new();
    let mut op_results: Vec<(SweepAxis, Vec<SweepPoint>)> = Vec::new();
    for &axis in SweepAxis::all() {
        let mlp_pts = run_sweep(&mlp, &mlp_stats, axis, N_PER_POINT, SWEEP_SEED)?;
        let op_pts = run_operator_sweep(&op, &op_stats, axis, N_PER_POINT, SWEEP_SEED)?;
        println!(
            "  axis={:<14}  mlp_worst={:.3}  op_worst={:.3}",
            axis.name(),
            worst_sum(&mlp_pts),
            worst_sum(&op_pts),
        );
        mlp_results.push((axis, mlp_pts));
        op_results.push((axis, op_pts));
    }

    plot_sweep_compare(&mlp_results, &op_results, &out_dir.join("sweep_compare.png"))?;

    // Summary table: worst sum(per-field rMSE/range) per axis, side-by-side.
    println!("\nworst sum(per-field rMSE/range) on widened distribution:");
    println!("  {:<14}  {:>10}  {:>10}", "axis", "wide MLP", "DeepONet");
    for ((axis, mlp_pts), (_, op_pts)) in mlp_results.iter().zip(op_results.iter()) {
        println!(
            "  {:<14}  {:>10.3}  {:>10.3}",
            axis.name(),
            worst_sum(mlp_pts),
            worst_sum(op_pts),
        );
    }
    Ok(())
}

fn worst_sum(pts: &[SweepPoint]) -> f32 {
    pts.iter()
        .map(|p| p.per_field_rmse_over_range.iter().sum::<f32>())
        .fold(0.0f32, f32::max)
}

/// Parallel to `sweep::run_sweep` but using the DeepONet `predict_operator`.
fn run_operator_sweep(
    op: &mhd_ml_bridge::operator::DeepONet,
    stats: &mhd_ml_bridge::surrogate::Stats,
    axis: SweepAxis,
    n_per_point: usize,
    seed: u64,
) -> Result<Vec<SweepPoint>> {
    use mhd_ml_bridge::data::{run_one, run_one_with_gamma, sample_params, GAMMA};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let values = axis.values();
    let mut points = Vec::with_capacity(values.len());
    for (i, v) in values.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
        let mut input_rows: Vec<[f32; N_FEATURES]> = Vec::with_capacity(n_per_point);
        let mut truth_rows: Vec<[f32; N_OUTPUTS]> = Vec::with_capacity(n_per_point);
        for _ in 0..n_per_point {
            let mut p = sample_params(&mut rng);
            let mut gamma = GAMMA;
            match axis {
                SweepAxis::RhoRatio => p[3] = *v as f32,
                SweepAxis::BySignFlip => {
                    if *v > 0.5 {
                        p[2] = -p[2];
                        p[5] = -p[5];
                    }
                }
                SweepAxis::Gamma => gamma = *v,
                SweepAxis::Bx => p[6] = *v as f32,
            }
            let out = if (gamma - GAMMA).abs() < 1e-12 {
                run_one(&p)
            } else {
                run_one_with_gamma(&p, gamma)
            };
            let mut flat = [0.0f32; N_OUTPUTS];
            for f in 0..4 {
                for j in 0..N_OUT {
                    flat[f * N_OUT + j] = out[[f, j]] as f32;
                }
            }
            truth_rows.push(flat);
            input_rows.push(p);
        }
        let pred_rows = predict_operator(op, stats, &input_rows)?;
        let (rmse, _) = per_field_rmse_over_range(&truth_rows, &pred_rows);
        points.push(SweepPoint {
            value: *v,
            per_field_rmse_over_range: rmse,
        });
    }
    Ok(points)
}

fn plot_training_curves(op_tr: &TrainResult, mlp_tr: &TrainResult, out: &Path) -> Result<()> {
    let op_val: Vec<(f32, f32)> = op_tr.val_hist.iter().enumerate().map(|(i, &y)| (i as f32, y)).collect();
    let op_train: Vec<(f32, f32)> = op_tr.train_hist.iter().enumerate().map(|(i, &y)| (i as f32, y)).collect();
    let mlp_val: Vec<(f32, f32)> = mlp_tr.val_hist.iter().enumerate().map(|(i, &y)| (i as f32, y)).collect();
    let mlp_train: Vec<(f32, f32)> = mlp_tr.train_hist.iter().enumerate().map(|(i, &y)| (i as f32, y)).collect();

    let all = op_tr.train_hist.iter()
        .chain(op_tr.val_hist.iter())
        .chain(mlp_tr.train_hist.iter())
        .chain(mlp_tr.val_hist.iter())
        .copied();
    let y_min = all.clone().fold(f32::INFINITY, f32::min).max(1e-6);
    let y_max = all.fold(f32::NEG_INFINITY, f32::max);

    let backend = BitMapBackend::new(out, (900, 600)).into_drawing_area();
    backend.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&backend)
        .caption(
            "Q11 — DeepONet vs wide-MLP training (widened distribution)",
            ("sans-serif", 20).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..op_tr.train_hist.len() as f32, (y_min..y_max).log_scale())?;
    chart.configure_mesh().x_desc("epoch").y_desc("normalized MSE").draw()?;

    chart.draw_series(LineSeries::new(op_train, BLUE.mix(0.6).stroke_width(1)))?
        .label("DeepONet train")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE.mix(0.6).stroke_width(1)));
    chart.draw_series(LineSeries::new(op_val, BLUE.stroke_width(2)))?
        .label("DeepONet val")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE.stroke_width(2)));
    chart.draw_series(LineSeries::new(mlp_train, RED.mix(0.6).stroke_width(1)))?
        .label("wide MLP train")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.mix(0.6).stroke_width(1)));
    chart.draw_series(LineSeries::new(mlp_val, RED.stroke_width(2)))?
        .label("wide MLP val")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));

    chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    backend.present().context("render training curve")?;
    println!("wrote {}", out.display());
    Ok(())
}

fn plot_val_example_compare(op_tr: &TrainResult, mlp_tr: &TrainResult, out: &Path) -> Result<()> {
    // Pick the same val example index; operator and MLP used the same seed, so
    // their val sets match element-wise.
    let idx = 0;
    let dx = 1.0f32 / N_OUT as f32;
    let x_grid: Vec<f32> = (0..N_OUT).map(|i| (i as f32 + 0.5) * dx).collect();
    let field_names = ["density rho", "velocity u", "B_y", "pressure p"];

    let backend = BitMapBackend::new(out, (1200, 800)).into_drawing_area();
    backend.fill(&WHITE)?;
    let title_area = backend.titled(
        "val[0]: HLL truth vs DeepONet vs wide-MLP",
        ("sans-serif", 20).into_font(),
    )?;
    let areas = title_area.split_evenly((2, 2));

    for (area, (f, name)) in areas.iter().zip(field_names.iter().enumerate()) {
        let truth: Vec<f32> = (0..N_OUT).map(|j| op_tr.val_truth[idx][f * N_OUT + j]).collect();
        let op_pred: Vec<f32> = (0..N_OUT).map(|j| op_tr.val_pred[idx][f * N_OUT + j]).collect();
        let mlp_pred: Vec<f32> = (0..N_OUT).map(|j| mlp_tr.val_pred[idx][f * N_OUT + j]).collect();

        let combined = truth.iter().chain(op_pred.iter()).chain(mlp_pred.iter()).copied();
        let mut y_min = combined.clone().fold(f32::INFINITY, f32::min);
        let mut y_max = combined.fold(f32::NEG_INFINITY, f32::max);
        let pad = (y_max - y_min).max(1e-3) * 0.1;
        y_min -= pad;
        y_max += pad;

        let mut chart = ChartBuilder::on(area)
            .caption(*name, ("sans-serif", 18).into_font())
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(0f32..1f32, y_min..y_max)?;
        chart.configure_mesh().x_desc("x").y_desc(*name).draw()?;

        chart.draw_series(LineSeries::new(
            x_grid.iter().zip(truth.iter()).map(|(&x, &y)| (x, y)),
            BLACK.stroke_width(2),
        ))?.label("HLL truth").legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK.stroke_width(2)));
        chart.draw_series(LineSeries::new(
            x_grid.iter().zip(op_pred.iter()).map(|(&x, &y)| (x, y)),
            BLUE.stroke_width(2),
        ))?.label("DeepONet").legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE.stroke_width(2)));
        chart.draw_series(LineSeries::new(
            x_grid.iter().zip(mlp_pred.iter()).map(|(&x, &y)| (x, y)),
            RED.stroke_width(2),
        ))?.label("wide MLP").legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));

        chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    }

    backend.present().context("render val example compare")?;
    println!("wrote {}", out.display());
    Ok(())
}

fn plot_sweep_compare(
    mlp_results: &[(SweepAxis, Vec<SweepPoint>)],
    op_results: &[(SweepAxis, Vec<SweepPoint>)],
    out: &Path,
) -> Result<()> {
    let backend = BitMapBackend::new(out, (1400, 900)).into_drawing_area();
    backend.fill(&WHITE)?;
    let title_area = backend.titled(
        "Q11 — per-axis sum(per-field rMSE/range): DeepONet (blue) vs wide MLP (red)",
        ("sans-serif", 18).into_font(),
    )?;
    let areas = title_area.split_evenly((2, 2));

    for (area, ((axis, mlp_pts), (_, op_pts))) in areas.iter().zip(mlp_results.iter().zip(op_results.iter())) {
        let xs: Vec<f64> = mlp_pts.iter().map(|p| p.value).collect();
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = ((hi - lo) * 0.05).max(1e-3);
        let x_min = lo - pad;
        let x_max = hi + pad;

        let mlp_y: Vec<(f64, f32)> = mlp_pts.iter().map(|p| (p.value, p.per_field_rmse_over_range.iter().sum::<f32>())).collect();
        let op_y:  Vec<(f64, f32)> = op_pts.iter().map(|p| (p.value, p.per_field_rmse_over_range.iter().sum::<f32>())).collect();
        let y_max = mlp_y.iter().chain(op_y.iter()).map(|(_, y)| *y).fold(0.0f32, f32::max).max(0.05) * 1.1;

        let mut chart = ChartBuilder::on(area)
            .caption(axis.name(), ("sans-serif", 18).into_font())
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, 0f32..y_max)?;
        chart.configure_mesh().x_desc("sweep value").y_desc("sum rMSE/range").draw()?;

        chart.draw_series(LineSeries::new(mlp_y.clone(), RED.stroke_width(2)))?
            .label("wide MLP")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));
        chart.draw_series(mlp_y.iter().map(|&(x, y)| Circle::new((x, y), 4, RED.filled())))?;

        chart.draw_series(LineSeries::new(op_y.clone(), BLUE.stroke_width(2)))?
            .label("DeepONet")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE.stroke_width(2)));
        chart.draw_series(op_y.iter().map(|&(x, y)| Circle::new((x, y), 4, BLUE.filled())))?;

        chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw()?;
    }

    backend.present().context("render sweep compare")?;
    println!("wrote {}", out.display());
    Ok(())
}
