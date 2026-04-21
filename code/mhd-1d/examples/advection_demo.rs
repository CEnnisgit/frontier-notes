//! Prints a convergence table and writes figures/advection.png.
//!
//! Run from the workspace root: `cargo run -p mhd-1d --example advection_demo`

use anyhow::{Context, Result};
use mhd_1d::advection::{convergence_study, run_gaussian_default};
use plotters::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let single = run_gaussian_default(400);
    let err = single.l2_error.expect("default: one period fits exactly");
    println!("Single run (N=400): L2 error after one period = {err:.6}\n");

    let (ns, errors, orders) = convergence_study(&[50, 100, 200, 400, 800]);
    println!("Convergence study:");
    println!("  {:>5}  {:>10}  {:>12}  {:>8}", "N", "dx", "L2 error", "order");
    for i in 0..ns.len() {
        let order_str = if i == 0 {
            "-".to_string()
        } else {
            format!("{:.3}", orders[i - 1])
        };
        println!(
            "  {:>5}  {:>10.5}  {:>12.6}  {:>8}",
            ns[i],
            1.0 / ns[i] as f64,
            errors[i],
            order_str
        );
    }
    let mean_order: f64 = orders.iter().sum::<f64>() / orders.len() as f64;
    println!("\nMean observed order: {mean_order:.3}  (expected ~1.0)");

    // Plot initial vs final profile at N=400.
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("figures");
    std::fs::create_dir_all(&out_dir)?;
    let out = out_dir.join("advection.png");
    let backend = BitMapBackend::new(&out, (900, 500)).into_drawing_area();
    backend.fill(&WHITE)?;
    let y_max = 1.05_f64;
    let y_min = -0.05_f64;
    let mut chart = ChartBuilder::on(&backend)
        .caption(
            format!("Gaussian pulse, one period, N=400, L2 error = {err:.4}"),
            ("sans-serif", 22).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0_f64..1.0_f64, y_min..y_max)?;
    chart.configure_mesh().x_desc("x").y_desc("u").draw()?;
    chart
        .draw_series(LineSeries::new(
            single
                .x
                .iter()
                .zip(single.u_initial.iter())
                .map(|(&x, &u)| (x, u)),
            BLACK.stroke_width(2),
        ))?
        .label("initial")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK.stroke_width(2)));
    chart
        .draw_series(LineSeries::new(
            single
                .x
                .iter()
                .zip(single.u_final.iter())
                .map(|(&x, &u)| (x, u)),
            RED.stroke_width(2),
        ))?
        .label("after one period (upwind)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(2)));
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    backend.present().context("render advection plot")?;
    println!("wrote {}", out.display());

    Ok(())
}
