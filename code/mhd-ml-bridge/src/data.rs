//! Generate training data by sweeping parameters around the canonical Brio-Wu IC
//! and running the sibling `mhd-1d` crate's HLL solver.
//!
//! Inputs (per example, 7 scalars):
//!   `rho_L, p_L, By_L, rho_R, p_R, By_R, Bx`
//!
//! Output (per example, flattened 4 × N_OUT):
//!   `[rho(x), u(x), By(x), p(x)]` each on a uniform grid of `N_OUT` cells
//!   at `t = T_FINAL`.

use mhd_1d::mhd::{conservative_to_primitive, mhd_simulate, primitive_to_conservative};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const GAMMA: f64 = 2.0;
pub const T_FINAL: f64 = 0.1;
pub const N_SIM: usize = 128;
pub const N_OUT: usize = 64;
pub const N_FEATURES: usize = 7;
pub const N_OUTPUTS: usize = 4 * N_OUT; // rho, u, By, p

/// Draw a single 7-vector of Brio-Wu-ish parameters, narrow Week-4 box.
pub fn sample_params(rng: &mut StdRng) -> [f32; 7] {
    [
        rng.gen_range(0.8..1.2),    // rho_L
        rng.gen_range(0.8..1.2),    // p_L
        rng.gen_range(0.8..1.2),    // By_L
        rng.gen_range(0.10..0.15),  // rho_R
        rng.gen_range(0.08..0.12),  // p_R
        rng.gen_range(-1.2..-0.8),  // By_R
        rng.gen_range(0.60..0.90),  // Bx
    ]
}

/// Draw a single 7-vector from the widened Week-6 distribution.
///
/// Widening, driven by Q10 findings at `notes/plasma/002-mlp-failure-modes.md`:
/// - By_L sign randomized (By_R always opposite), so models can't memorize
///   `By_L > 0`.
/// - rho_R widened from `[0.10, 0.15]` to `[0.05, 0.40]`.
/// - Bx widened from `[0.60, 0.90]` to `[0.0, 1.5]`.
pub fn sample_params_wide(rng: &mut StdRng) -> [f32; 7] {
    let by_l_sign: f32 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
    let by_l_mag: f32 = rng.gen_range(0.8..1.2);
    let by_r_mag: f32 = rng.gen_range(0.8..1.2);
    [
        rng.gen_range(0.8..1.2),       // rho_L
        rng.gen_range(0.8..1.2),       // p_L
        by_l_sign * by_l_mag,          // By_L (sign randomized)
        rng.gen_range(0.05..0.40),     // rho_R
        rng.gen_range(0.08..0.12),     // p_R
        -by_l_sign * by_r_mag,         // By_R (opposite sign of By_L)
        rng.gen_range(0.0..1.5),       // Bx
    ]
}

/// Run HLL MHD on a single parameter vector at γ=GAMMA.
pub fn run_one(params: &[f32; 7]) -> Array2<f64> {
    run_one_with_gamma(params, GAMMA)
}

/// Run HLL MHD on a single parameter vector at arbitrary γ; return (4, N_OUT)
/// primitives at `t = T_FINAL`. Used by the sweep harness to probe γ-OOD.
pub fn run_one_with_gamma(params: &[f32; 7], gamma: f64) -> Array2<f64> {
    let [rho_l, p_l, by_l, rho_r, p_r, by_r, bx] = params.map(|x| x as f64);
    let dx = 1.0 / N_SIM as f64;

    let mut rho = Array1::<f64>::zeros(N_SIM);
    let u = Array1::<f64>::zeros(N_SIM);
    let v = Array1::<f64>::zeros(N_SIM);
    let w = Array1::<f64>::zeros(N_SIM);
    let mut by = Array1::<f64>::zeros(N_SIM);
    let bz = Array1::<f64>::zeros(N_SIM);
    let mut p = Array1::<f64>::zeros(N_SIM);
    for i in 0..N_SIM {
        let xi = (i as f64 + 0.5) * dx;
        if xi < 0.5 {
            rho[i] = rho_l;
            p[i] = p_l;
            by[i] = by_l;
        } else {
            rho[i] = rho_r;
            p[i] = p_r;
            by[i] = by_r;
        }
    }
    let u0 = primitive_to_conservative(&rho, &u, &v, &w, &by, &bz, &p, gamma, bx);
    let u_final = mhd_simulate(&u0, dx, T_FINAL, gamma, bx, 0.4);
    let s = conservative_to_primitive(&u_final, gamma, bx);

    // Downsample N_SIM -> N_OUT by block-mean.
    let ratio = N_SIM / N_OUT;
    assert_eq!(N_SIM, N_OUT * ratio, "N_SIM must be divisible by N_OUT");
    let mut out = Array2::<f64>::zeros((4, N_OUT));
    for j in 0..N_OUT {
        let mut acc = [0.0_f64; 4];
        for k in 0..ratio {
            let idx = j * ratio + k;
            acc[0] += s.rho[idx];
            acc[1] += s.u[idx];
            acc[2] += s.by[idx];
            acc[3] += s.p[idx];
        }
        for f in 0..4 {
            out[[f, j]] = acc[f] / ratio as f64;
        }
    }
    out
}

pub struct Dataset {
    /// Shape `(n_examples, N_FEATURES)`.
    pub inputs: Vec<[f32; N_FEATURES]>,
    /// Shape `(n_examples, N_OUTPUTS)`, row-major [rho, u, By, p] stacked.
    pub outputs: Vec<[f32; N_OUTPUTS]>,
}

pub fn build_dataset(n_examples: usize, seed: u64) -> Dataset {
    build_dataset_from(n_examples, seed, sample_params)
}

/// Build a dataset from the widened Week-6 distribution.
pub fn build_dataset_wide(n_examples: usize, seed: u64) -> Dataset {
    build_dataset_from(n_examples, seed, sample_params_wide)
}

fn build_dataset_from(
    n_examples: usize,
    seed: u64,
    sampler: fn(&mut StdRng) -> [f32; 7],
) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(n_examples);
    let mut outputs = Vec::with_capacity(n_examples);
    for _ in 0..n_examples {
        let params = sampler(&mut rng);
        let out = run_one(&params); // (4, N_OUT)
        let mut flat = [0.0_f32; N_OUTPUTS];
        for f in 0..4 {
            for j in 0..N_OUT {
                flat[f * N_OUT + j] = out[[f, j]] as f32;
            }
        }
        inputs.push(params);
        outputs.push(flat);
    }
    Dataset { inputs, outputs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_small_dataset_has_expected_shape() {
        let ds = build_dataset(4, 0);
        assert_eq!(ds.inputs.len(), 4);
        assert_eq!(ds.outputs.len(), 4);
        // Sanity: rho field of first example should all be positive.
        for j in 0..N_OUT {
            assert!(ds.outputs[0][j] > 0.0, "rho[{j}] non-positive");
        }
    }
}
