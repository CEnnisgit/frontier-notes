//! Sod shock tube: HLL vs exact Riemann at t=0.2. Writes figures/sod.png.
//!
//! `cargo run -p mhd-1d --example plot_sod --release`

use anyhow::{Context, Result};
use mhd_1d::euler::{conservative_to_primitive, simulate, sod_initial};
use mhd_1d::riemann_euler::sample_grid;
use ndarray::Array1;
use plotters::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let gamma = 1.4;
    let n = 400;
    let t_final = 0.2;

    let (x, u0) = sod_initial(n, (0.0, 1.0), gamma);
    let dx = x[1] - x[0];
    let u = simulate(&u0, dx, t_final, gamma, 0.4);
    let (rho, u_num, p) = conservative_to_primitive(&u, gamma);

    let n_fine = 2000;
    let dx_fine = 1.0 / n_fine as f64;
    let x_fine = Array1::from_iter((0..n_fine).map(|i| (i as f64 + 0.5) * dx_fine));
    let (rho_ex, u_ex, p_ex) =
        sample_grid(&x_fine, t_final, 0.5, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gamma);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("figures");
    std::fs::create_dir_all(&out_dir)?;
    let out = out_dir.join("sod.png");
    let backend = BitMapBackend::new(&out, (1300, 400)).into_drawing_area();
    backend.fill(&WHITE)?;
    let areas = backend.split_evenly((1, 3));

    let panels: [(&str, &Array1<f64>, &Array1<f64>, f64, f64); 3] = [
        ("density", &rho, &rho_ex, 0.0, 1.1),
        ("velocity", &u_num, &u_ex, -0.05, 1.1),
        ("pressure", &p, &p_ex, 0.0, 1.1),
    ];

    for (area, (name, num, ex, y_min, y_max)) in areas.iter().zip(panels.iter()) {
        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("Sod: {name}, t={t_final}"),
                ("sans-serif", 18).into_font(),
            )
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0.0_f64..1.0_f64, *y_min..*y_max)?;
        chart.configure_mesh().x_desc("x").y_desc(*name).draw()?;

        chart
            .draw_series(LineSeries::new(
                x_fine.iter().zip(ex.iter()).map(|(&x, &y)| (x, y)),
                BLACK.stroke_width(2),
            ))?
            .label("exact")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK.stroke_width(2)));

        chart
            .draw_series(
                x.iter()
                    .zip(num.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 2, RED.filled())),
            )?
            .label(format!("HLL, N={n}"))
            .legend(|(x, y)| Circle::new((x + 10, y), 3, RED.filled()));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    backend.present().context("render sod plot")?;
    println!("wrote {}", out.display());
    Ok(())
}
