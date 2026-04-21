//! 1D hyperbolic solvers for the frontier-notes research repo.
//!
//! - [`advection`] — linear advection with upwind finite volume.
//! - [`euler`] — 1D Euler equations with HLL flux.
//! - [`riemann_euler`] — exact Euler Riemann solver (Toro, 1999, §4).
//! - [`mhd`] — 1D ideal MHD with HLL flux, verified against Brio-Wu.

pub mod advection;
pub mod euler;
pub mod mhd;
pub mod riemann_euler;
