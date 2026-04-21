//! Brio-Wu shock tube: 4-panel comparison of rho, u, B_y, p at t=0.1.
//!
//! No analytical reference solution exists (MHD has no closed-form exact
//! Riemann solver); the plot is for visual comparison to published figures.
//!
//! `cargo run -p mhd-1d --example plot_brio_wu --release`

use anyhow::{Context, Result};
use mhd_1d::mhd::{
    brio_wu_initial, conservative_to_primitive, mhd_simulate, BX_BRIO_WU, GAMMA_BRIO_WU,
};
use ndarray::Array1;
use plotters::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let n_coarse = 400;
    let n_fine = 1600;
    let t_final = 0.1;

    let mut coarse_results = None;
    let mut fine_results = None;
    for (n, slot) in [
        (n_coarse, &mut coarse_results),
        (n_fine, &mut fine_results),
    ] {
        let (x, u0, bx) = brio_wu_initial(n, (0.0, 1.0));
        let dx = x[1] - x[0];
        let u = mhd_simulate(&u0, dx, t_final, GAMMA_BRIO_WU, bx, 0.4);
        let s = conservative_to_primitive(&u, GAMMA_BRIO_WU, bx);
        *slot = Some((x, s.rho, s.u, s.by, s.p));
    }
    let (x_c, rho_c, u_c, by_c, p_c) = coarse_results.unwrap();
    let (x_f, rho_f, u_f, by_f, p_f) = fine_results.unwrap();

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("figures");
    std::fs::create_dir_all(&out_dir)?;
    let out = out_dir.join("brio-wu.png");
    let backend = BitMapBackend::new(&out, (1200, 800)).into_drawing_area();
    backend.fill(&WHITE)?;
    let areas = backend.split_evenly((2, 2));

    let panels: [(&str, &Array1<f64>, &Array1<f64>, (f64, f64)); 4] = [
        ("density", &rho_f, &rho_c, (0.0, 1.1)),
        ("velocity u", &u_f, &u_c, (-0.1, 0.8)),
        ("B_y", &by_f, &by_c, (-1.1, 1.1)),
        ("pressure", &p_f, &p_c, (0.0, 1.1)),
    ];

    for (area, (name, fine, coarse, (y_min, y_max))) in areas.iter().zip(panels.iter()) {
        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("Brio-Wu {name}, t={t_final}, B_x={BX_BRIO_WU}"),
                ("sans-serif", 18).into_font(),
            )
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0.0_f64..1.0_f64, *y_min..*y_max)?;
        chart.configure_mesh().x_desc("x").y_desc(*name).draw()?;

        chart
            .draw_series(LineSeries::new(
                x_f.iter().zip(fine.iter()).map(|(&x, &y)| (x, y)),
                BLACK.stroke_width(2),
            ))?
            .label(format!("N={n_fine}"))
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLACK.stroke_width(2)));

        chart
            .draw_series(
                x_c.iter()
                    .zip(coarse.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 2, RED.filled())),
            )?
            .label(format!("N={n_coarse}"))
            .legend(|(x, y)| Circle::new((x + 10, y), 3, RED.filled()));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    backend.present().context("render brio-wu plot")?;
    println!("wrote {}", out.display());
    Ok(())
}
