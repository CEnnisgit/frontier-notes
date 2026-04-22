//! DeepONet surrogate for the Brio-Wu problem (Week 6, Q11).
//!
//! Architecture: a branch net over the 7-parameter IC vector and a trunk net
//! over the 1-D grid coordinate `x`. The two are combined by inner product
//! in a shared latent space, one channel per output field:
//!
//! ```text
//!   branch(u): R^7   -> R^{N_FIELDS * LATENT_D}
//!   trunk(x):  R^1   -> R^{LATENT_D}
//!   y_f(x) = <branch_f(u), trunk(x)>         for f in {rho, u, By, p}
//! ```
//!
//! The grid is the same `N_OUT = 64` cell-centered uniform grid that
//! `data.rs` downsamples HLL truth onto, so this is a fair per-cell compare
//! against the Week-4 MLP. Branch/trunk hidden widths are sized so the total
//! parameter count is in the same order of magnitude as the 7→128→128→256 MLP.

use crate::data::{build_dataset_wide, Dataset, N_FEATURES, N_OUT, N_OUTPUTS};
use crate::surrogate::{
    column_stats, normalize_rows, per_field_rmse_over_range, rows_to_tensor, Stats, TrainResult,
    BATCH, EPOCHS, LR, N_TRAIN, N_VAL, SEED,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub const BRANCH_HIDDEN: usize = 128;
pub const TRUNK_HIDDEN: usize = 64;
pub const LATENT_D: usize = 64;
pub const N_FIELDS: usize = 4;

pub struct DeepONet {
    b1: Linear,
    b2: Linear,
    b3: Linear,
    t1: Linear,
    t2: Linear,
    t3: Linear,
    /// Cached (N_OUT, 1) grid tensor so we don't rebuild it per forward pass.
    x_grid: Tensor,
}

impl DeepONet {
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let b1 = candle_nn::linear(N_FEATURES, BRANCH_HIDDEN, vb.pp("b1"))?;
        let b2 = candle_nn::linear(BRANCH_HIDDEN, BRANCH_HIDDEN, vb.pp("b2"))?;
        let b3 = candle_nn::linear(BRANCH_HIDDEN, N_FIELDS * LATENT_D, vb.pp("b3"))?;
        let t1 = candle_nn::linear(1, TRUNK_HIDDEN, vb.pp("t1"))?;
        let t2 = candle_nn::linear(TRUNK_HIDDEN, TRUNK_HIDDEN, vb.pp("t2"))?;
        let t3 = candle_nn::linear(TRUNK_HIDDEN, LATENT_D, vb.pp("t3"))?;

        let dx = 1.0f32 / N_OUT as f32;
        let grid: Vec<f32> = (0..N_OUT).map(|i| (i as f32 + 0.5) * dx).collect();
        let x_grid = Tensor::from_vec(grid, (N_OUT, 1), device)?;

        Ok(Self { b1, b2, b3, t1, t2, t3, x_grid })
    }

    fn trunk(&self) -> candle_core::Result<Tensor> {
        // (N_OUT, 1) -> (N_OUT, LATENT_D)
        let h = self.t1.forward(&self.x_grid)?.gelu()?;
        let h = self.t2.forward(&h)?.gelu()?;
        self.t3.forward(&h)
    }
}

impl Module for DeepONet {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let b = x.dims()[0];
        let h = self.b1.forward(x)?.gelu()?;
        let h = self.b2.forward(&h)?.gelu()?;
        let branch = self.b3.forward(&h)?; // (B, N_FIELDS * LATENT_D)

        // (B * N_FIELDS, LATENT_D)
        let branch_flat = branch.reshape((b * N_FIELDS, LATENT_D))?;

        // (N_OUT, LATENT_D) -> (LATENT_D, N_OUT)
        let trunk = self.trunk()?.t()?.contiguous()?;

        // (B * N_FIELDS, N_OUT)
        let mixed = branch_flat.matmul(&trunk)?;

        // (B, N_FIELDS * N_OUT), with N_FIELDS as the slow axis (rho|u|By|p)
        mixed.reshape((b, N_FIELDS * N_OUT))
    }
}

/// Apply the trained DeepONet to a batch of raw (unnormalized) parameter
/// vectors; return unnormalized predictions in the same layout as `predict`.
pub fn predict_operator(
    model: &DeepONet,
    stats: &Stats,
    inputs: &[[f32; N_FEATURES]],
) -> Result<Vec<[f32; N_OUTPUTS]>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let x_mean_arr: [f32; N_FEATURES] = stats.x_mean.as_slice().try_into().unwrap();
    let x_std_arr: [f32; N_FEATURES] = stats.x_std.as_slice().try_into().unwrap();
    let x_n = normalize_rows(inputs, &x_mean_arr, &x_std_arr);
    let x_t = rows_to_tensor(&x_n, &stats.device)?;
    let y_n_t = model.forward(&x_t)?;
    let y_n_vec: Vec<Vec<f32>> = y_n_t.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(inputs.len());
    for row in y_n_vec.iter() {
        let mut a = [0.0f32; N_OUTPUTS];
        for j in 0..N_OUTPUTS {
            a[j] = row[j] * (stats.y_std[j] + 1e-8) + stats.y_mean[j];
        }
        out.push(a);
    }
    Ok(out)
}

/// Train DeepONet on the widened Week-6 distribution. Same optimizer + epoch
/// budget as the Week-4 MLP so the compare is apples-to-apples.
pub fn train_operator_wide() -> Result<(DeepONet, VarMap, Stats, TrainResult)> {
    println!("generating data (widened Week-6 distribution)...");
    let train_ds = build_dataset_wide(N_TRAIN, SEED);
    let val_ds = build_dataset_wide(N_VAL, SEED + 1);
    train_operator_on_datasets(train_ds, val_ds)
}

/// Shared training core for DeepONet, mirroring `surrogate::train_on_datasets`
/// so the two models see identical data and identical optimizer hyperparameters.
pub fn train_operator_on_datasets(
    train_ds: Dataset,
    val_ds: Dataset,
) -> Result<(DeepONet, VarMap, Stats, TrainResult)> {
    let device = Device::Cpu;

    let (x_mean, x_std) = column_stats::<N_FEATURES>(&train_ds.inputs);
    let (y_mean, y_std) = column_stats::<N_OUTPUTS>(&train_ds.outputs);

    let x_train_n = normalize_rows(&train_ds.inputs, &x_mean, &x_std);
    let y_train_n = normalize_rows(&train_ds.outputs, &y_mean, &y_std);
    let x_val_n = normalize_rows(&val_ds.inputs, &x_mean, &x_std);
    let y_val_n = normalize_rows(&val_ds.outputs, &y_mean, &y_std);

    let x_train_t = rows_to_tensor(&x_train_n, &device)?;
    let y_train_t = rows_to_tensor(&y_train_n, &device)?;
    let x_val_t = rows_to_tensor(&x_val_n, &device)?;
    let y_val_t = rows_to_tensor(&y_val_n, &device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = DeepONet::new(vb, &device)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LR,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut train_hist = Vec::with_capacity(EPOCHS);
    let mut val_hist = Vec::with_capacity(EPOCHS);

    println!(
        "training DeepONet for {EPOCHS} epochs, batch={BATCH}, N_train={N_TRAIN} \
         (branch={BRANCH_HIDDEN}, trunk={TRUNK_HIDDEN}, latent={LATENT_D})"
    );
    for epoch in 0..EPOCHS {
        let mut indices: Vec<usize> = (0..N_TRAIN).collect();
        indices.shuffle(&mut rng);

        let mut epoch_losses = Vec::new();
        for start in (0..N_TRAIN).step_by(BATCH) {
            let end = (start + BATCH).min(N_TRAIN);
            let batch_idx: Vec<u32> = indices[start..end].iter().map(|&i| i as u32).collect();
            let idx_t = Tensor::from_vec(batch_idx, (end - start,), &device)?;

            let xb = x_train_t.index_select(&idx_t, 0)?;
            let yb = y_train_t.index_select(&idx_t, 0)?;

            let pred = model.forward(&xb)?;
            let loss = (&pred - &yb)?.sqr()?.mean_all()?;
            opt.backward_step(&loss)?;
            epoch_losses.push(loss.to_scalar::<f32>()?);
        }
        let train_loss = epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32;

        let val_pred_n = model.forward(&x_val_t)?;
        let val_loss = ((&val_pred_n - &y_val_t)?.sqr()?.mean_all()?).to_scalar::<f32>()?;

        train_hist.push(train_loss);
        val_hist.push(val_loss);
        if epoch % 50 == 0 || epoch == EPOCHS - 1 {
            println!("  epoch {epoch:4}  train={train_loss:.4e}  val={val_loss:.4e}");
        }
    }

    let val_pred_n = model.forward(&x_val_t)?;
    let val_pred_vec: Vec<Vec<f32>> = val_pred_n.to_vec2::<f32>()?;

    let mut val_pred_arr: Vec<[f32; N_OUTPUTS]> = Vec::with_capacity(N_VAL);
    for row in val_pred_vec.iter() {
        let mut a = [0.0f32; N_OUTPUTS];
        for j in 0..N_OUTPUTS {
            a[j] = row[j] * (y_std[j] + 1e-8) + y_mean[j];
        }
        val_pred_arr.push(a);
    }

    let (field_rmse, _) = per_field_rmse_over_range(&val_ds.outputs, &val_pred_arr);

    let stats = Stats {
        x_mean: x_mean.to_vec(),
        x_std: x_std.to_vec(),
        y_mean: y_mean.to_vec(),
        y_std: y_std.to_vec(),
        device,
    };

    let tr = TrainResult {
        train_hist,
        val_hist,
        val_inputs: val_ds.inputs.clone(),
        val_truth: val_ds.outputs.clone(),
        val_pred: val_pred_arr,
        field_rmse_over_range: field_rmse,
    };
    Ok((model, varmap, stats, tr))
}
