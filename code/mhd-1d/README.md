# mhd-1d

A from-scratch 1D finite-volume solver, built up in three stages:

| Stage | Equation | Module | Tests |
|-------|----------|--------|-------|
| Week 1 | Linear advection: `∂u/∂t + c ∂u/∂x = 0` | `src/advection.rs` | `tests/advection.rs` |
| Week 2 | 1D Euler (compressible gas) | `src/euler.rs`, `src/riemann_euler.rs` | `tests/euler.rs` |
| Week 3 | 1D ideal MHD | `src/mhd.rs` | `tests/mhd.rs` |

The final target is the Brio-Wu shock tube — the canonical 1D ideal-MHD
benchmark (Brio & Wu, *JCP* 1988). Each stage is verified against an
analytical solution or published reference before moving on.

Week 4 bridges this classical solver to the ML method layer (see
`code/mhd-ml-bridge/`).

## Design choices

- **Rust + ndarray.** The project started in Python/NumPy+JAX; Month 1
  concluded with a full refactor to Rust for performance and maintainability.
  `ndarray` is the NumPy analog; `candle` is the ML layer (in
  `mhd-ml-bridge`).
- **Upwind FV + forward Euler for Week 1.** Simplest stable scheme.
  First-order accurate, diffusive, good teaching vehicle.
- **HLL flux for Weeks 2–3.** Positivity-preserving, no knowledge of the
  full eigenstructure needed, works fine for Sod and Brio-Wu.
- **Exact Riemann solver for Euler reference.** Toro §4.2–4.5 iteration
  on star pressure, used only as the truth solution for Sod tests.
- **Periodic BC for advection, transmissive for Euler/MHD shock tubes.**
  Matches the analytical/reference solutions.

## Running

From the workspace root `code/`:

```bash
cargo test -p mhd-1d                               # tests
cargo run -p mhd-1d --example advection_demo       # convergence study
cargo run -p mhd-1d --example plot_sod --release   # Sod shock tube 4-panel figure
cargo run -p mhd-1d --example plot_brio_wu --release
```

Figures land under `code/mhd-1d/figures/`.

## What "verified" means here

- **Advection:** L2 error bounded after one periodic revolution; empirical
  convergence order approaches 1.0 for upwind+Euler.
- **Euler (Week 2):** L1 error ≤ 2% vs Sod analytical at N=400, t=0.2.
- **MHD (Week 3):** Brio-Wu conservation (mass, momentum, energy) holds to
  machine precision; center density in the expected `[0.55, 0.80]` band;
  `B_y` flips sign across the domain.

If a test fails, debug — do not weaken the threshold without writing down
why.
