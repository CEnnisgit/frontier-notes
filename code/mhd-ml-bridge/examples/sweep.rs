//! Week 5 — Q10 stress-test.
//!
//! Trains the Week-4 MLP once, then sweeps each of four physical axes
//! past the training envelope and reports per-field rMSE/range on the
//! HLL truth.
//!
//! `cargo run -p mhd-ml-bridge --example sweep --release`

use anyhow::{Context, Result};
use mhd_ml_bridge::data::{N_FEATURES, N_OUT, N_OUTPUTS};
use mhd_ml_bridge::surrogate::train_model;
use mhd_ml_bridge::sweep::{run_sweep, worst_case, SweepAxis, SweepPoint};
use plotters::prelude::*;
use std::path::{Path, PathBuf};

const N_PER_POINT: usize = 16;
const SWEEP_SEED: u64 = 42;

fn main() -> Result<()> {
    // Trained once, shared across all axes.
    let (mlp, _varmap, stats, _tr) = train_model()?;

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("figures")
        .join("sweep");
    std::fs::create_dir_all(&out_dir)?;

    println!("\nrunning sweeps with n_per_point={N_PER_POINT}");

    // Collect (axis, sweep points, worst per-field-rmse sum) for ranking.
    let mut results: Vec<(SweepAxis, Vec<SweepPoint>, f32)> = Vec::new();
    for &axis in SweepAxis::all() {
        let pts = run_sweep(&mlp, &stats, axis, N_PER_POINT, SWEEP_SEED)?;
        let worst_sum = pts
            .iter()
            .map(|p| p.per_field_rmse_over_range.iter().sum::<f32>())
            .fold(0.0f32, f32::max);
        println!("  axis={:<14}  worst sum(rMSE/range)={:.3}", axis.name(), worst_sum);
        let fig_path = out_dir.join(format!("{}.png", axis.name()));
        plot_axis_sweep(axis, &pts, &fig_path)?;
        results.push((axis, pts, worst_sum));
    }

    // Summary table.
    println!("\nper-axis worst-case per-field rMSE/range:");
    println!("  {:<14}  {:>8}  {:>6}  {:>6}  {:>6}  {:>6}", "axis", "at value", "rho", "u", "By", "p");
    for (axis, pts, _) in &results {
        let (worst_pt, _) = pts
            .iter()
            .map(|p| (p, p.per_field_rmse_over_range.iter().sum::<f32>()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        println!(
            "  {:<14}  {:>8.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}",
            axis.name(),
            worst_pt.value,
            worst_pt.per_field_rmse_over_range[0],
            worst_pt.per_field_rmse_over_range[1],
            worst_pt.per_field_rmse_over_range[2],
            worst_pt.per_field_rmse_over_range[3],
        );
    }

    // Worst-case visual: two highest-rmse axes.
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (axis, _, _) in results.iter().take(2) {
        let (truth, pred, params, value) =
            worst_case(&mlp, &stats, *axis, N_PER_POINT, SWEEP_SEED)?;
        let fig_path = out_dir.join(format!("{}_worst_example.png", axis.name()));
        plot_worst_example(*axis, value, &params, &truth, &pred, &fig_path)?;
    }

    Ok(())
}

fn plot_axis_sweep(axis: SweepAxis, pts: &[SweepPoint], out: &Path) -> Result<()> {
    let field_names = ["rho", "u", "By", "p"];
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA];

    let (x_min, x_max) = {
        let xs: Vec<f64> = pts.iter().map(|p| p.value).collect();
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = ((hi - lo) * 0.05).max(1e-3);
        (lo - pad, hi + pad)
    };
    let y_max = pts
        .iter()
        .flat_map(|p| p.per_field_rmse_over_range.iter().copied())
        .fold(0.0f32, f32::max)
        .max(0.01)
        * 1.1;

    let backend = BitMapBackend::new(out, (900, 600)).into_drawing_area();
    backend.fill(&WHITE)?;
    let caption = format!("Q10 sweep — {} (n_per_point={N_PER_POINT})", axis.name());
    let mut chart = ChartBuilder::on(&backend)
        .caption(caption, ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc(axis_x_label(axis))
        .y_desc("per-field rMSE / range")
        .draw()?;

    for (f, name) in field_names.iter().enumerate() {
        let color = colors[f];
        let series: Vec<(f64, f32)> = pts
            .iter()
            .map(|p| (p.value, p.per_field_rmse_over_range[f]))
            .collect();
        chart
            .draw_series(LineSeries::new(series.clone(), color.stroke_width(2)))?
            .label(*name)
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], color.stroke_width(2)));
        chart.draw_series(
            series
                .into_iter()
                .map(|(x, y)| Circle::new((x, y), 4, color.filled())),
        )?;
    }

    // Mark training envelope.
    if let Some((lo, hi)) = training_envelope(axis) {
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(lo, 0.0), (hi, y_max)],
                RGBColor(220, 220, 220).mix(0.35).filled(),
            )))?
            .label("training envelope")
            .legend(|(x, y)| {
                Rectangle::new(
                    [(x, y - 5), (x + 20, y + 5)],
                    RGBColor(220, 220, 220).mix(0.35).filled(),
                )
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    backend.present().context("render sweep figure")?;
    println!("  wrote {}", out.display());
    Ok(())
}

fn axis_x_label(axis: SweepAxis) -> &'static str {
    match axis {
        SweepAxis::RhoRatio => "rho_R (right-state density)",
        SweepAxis::BySignFlip => "By sign flip (0 = no flip, 1 = flipped)",
        SweepAxis::Gamma => "gamma",
        SweepAxis::Bx => "B_x (guide field)",
    }
}

fn training_envelope(axis: SweepAxis) -> Option<(f64, f64)> {
    match axis {
        SweepAxis::RhoRatio => Some((0.10, 0.15)),
        SweepAxis::BySignFlip => Some((-0.1, 0.1)), // "no-flip" only
        SweepAxis::Gamma => Some((2.0 - 1e-3, 2.0 + 1e-3)),
        SweepAxis::Bx => Some((0.60, 0.90)),
    }
}

fn plot_worst_example(
    axis: SweepAxis,
    value: f64,
    _params: &[f32; N_FEATURES],
    truth: &[f32; N_OUTPUTS],
    pred: &[f32; N_OUTPUTS],
    out: &Path,
) -> Result<()> {
    let field_names = ["density rho", "velocity u", "B_y", "pressure p"];
    let dx = 1.0f32 / N_OUT as f32;
    let x_grid: Vec<f32> = (0..N_OUT).map(|i| (i as f32 + 0.5) * dx).collect();

    let backend = BitMapBackend::new(out, (1200, 800)).into_drawing_area();
    backend.fill(&WHITE)?;
    let title_area = backend.titled(
        &format!(
            "worst-case under axis={} at value={:.3}",
            axis.name(),
            value
        ),
        ("sans-serif", 22).into_font(),
    )?;
    let areas = title_area.split_evenly((2, 2));

    for (area, (f, name)) in areas.iter().zip(field_names.iter().enumerate()) {
        let truth_f: Vec<f32> = (0..N_OUT).map(|j| truth[f * N_OUT + j]).collect();
        let pred_f: Vec<f32> = (0..N_OUT).map(|j| pred[f * N_OUT + j]).collect();

        let combined = truth_f.iter().chain(pred_f.iter()).copied();
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

        chart
            .draw_series(LineSeries::new(
                x_grid.iter().zip(truth_f.iter()).map(|(&x, &y)| (x, y)),
                BLACK.stroke_width(2),
            ))?
            .label("HLL (truth)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK.stroke_width(2)));
        chart
            .draw_series(LineSeries::new(
                x_grid.iter().zip(pred_f.iter()).map(|(&x, &y)| (x, y)),
                RED.stroke_width(2),
            ))?
            .label("MLP")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    backend.present().context("render worst-case panel")?;
    println!("  wrote {}", out.display());
    Ok(())
}
