//! Brio-Wu MLP surrogate — train and plot.
//!
//! `cargo run -p mhd-ml-bridge --example train --release`

use anyhow::{Context, Result};
use mhd_ml_bridge::data::{N_OUT, N_OUTPUTS};
use mhd_ml_bridge::surrogate::{train, TrainResult};
use plotters::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let r: TrainResult = train()?;

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("figures");
    std::fs::create_dir_all(&out_dir)?;

    plot_training_curve(&r, &out_dir.join("training_curve.png"))?;
    plot_val_example(&r, &out_dir.join("val_example.png"))?;

    Ok(())
}

fn plot_training_curve(r: &TrainResult, out: &std::path::Path) -> Result<()> {
    let train: Vec<(f32, f32)> = r
        .train_hist
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f32, y))
        .collect();
    let val: Vec<(f32, f32)> = r
        .val_hist
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f32, y))
        .collect();

    let y_min = r
        .train_hist
        .iter()
        .chain(r.val_hist.iter())
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(1e-6);
    let y_max = r
        .train_hist
        .iter()
        .chain(r.val_hist.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let backend = BitMapBackend::new(out, (900, 600)).into_drawing_area();
    backend.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&backend)
        .caption(
            "Brio-Wu MLP surrogate training (candle)",
            ("sans-serif", 22).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..r.train_hist.len() as f32,
            (y_min..y_max).log_scale(),
        )?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("normalized MSE")
        .draw()?;

    chart
        .draw_series(LineSeries::new(train, BLUE.stroke_width(2)))?
        .label("train")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE.stroke_width(2)));
    chart
        .draw_series(LineSeries::new(val, RED.stroke_width(2)))?
        .label("val")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    backend.present().context("render training curve")?;
    println!("wrote {}", out.display());
    Ok(())
}

fn plot_val_example(r: &TrainResult, out: &std::path::Path) -> Result<()> {
    let idx = 0;
    let dx = 1.0f32 / N_OUT as f32;
    let x_grid: Vec<f32> = (0..N_OUT).map(|i| (i as f32 + 0.5) * dx).collect();
    let field_names = ["density rho", "velocity u", "B_y", "pressure p"];

    let backend = BitMapBackend::new(out, (1200, 800)).into_drawing_area();
    backend.fill(&WHITE)?;
    let areas = backend.split_evenly((2, 2));

    for (area, (f, name)) in areas.iter().zip(field_names.iter().enumerate()) {
        let truth: Vec<f32> = (0..N_OUT)
            .map(|j| r.val_truth[idx][f * N_OUT + j])
            .collect();
        let pred: Vec<f32> = (0..N_OUT).map(|j| r.val_pred[idx][f * N_OUT + j]).collect();

        let combined = truth.iter().chain(pred.iter()).copied();
        let mut y_min = combined.clone().fold(f32::INFINITY, f32::min);
        let mut y_max = combined.fold(f32::NEG_INFINITY, f32::max);
        let pad = (y_max - y_min).max(1e-3) * 0.1;
        y_min -= pad;
        y_max += pad;

        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("val[0] {name}, t=0.1"),
                ("sans-serif", 18).into_font(),
            )
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0f32..1f32, y_min..y_max)?;
        chart.configure_mesh().x_desc("x").y_desc(*name).draw()?;

        chart
            .draw_series(LineSeries::new(
                x_grid.iter().zip(truth.iter()).map(|(&x, &y)| (x, y)),
                BLACK.stroke_width(2),
            ))?
            .label("HLL (truth)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK.stroke_width(2)));
        chart
            .draw_series(LineSeries::new(
                x_grid.iter().zip(pred.iter()).map(|(&x, &y)| (x, y)),
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

    backend.present().context("render val example plot")?;
    println!("wrote {}", out.display());

    // Silence unused import in case N_OUTPUTS ever drops.
    let _ = N_OUTPUTS;
    Ok(())
}
