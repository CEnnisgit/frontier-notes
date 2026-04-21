//! Brio-Wu MLP surrogate in candle.
//!
//! Scaffolding, not science. The point is to prove the stack runs end-to-end:
//!   classical solver -> training data -> candle MLP -> predictions on held-out.

use crate::data::{build_dataset, N_FEATURES, N_OUT, N_OUTPUTS};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub const HIDDEN1: usize = 128;
pub const HIDDEN2: usize = 128;
pub const N_TRAIN: usize = 96;
pub const N_VAL: usize = 32;
pub const EPOCHS: usize = 800;
pub const BATCH: usize = 16;
pub const LR: f64 = 1e-3;
pub const SEED: u64 = 0;

pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl Mlp {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(N_FEATURES, HIDDEN1, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(HIDDEN1, HIDDEN2, vb.pp("fc2"))?;
        let fc3 = candle_nn::linear(HIDDEN2, N_OUTPUTS, vb.pp("fc3"))?;
        Ok(Self { fc1, fc2, fc3 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.fc1.forward(x)?.gelu()?;
        let h = self.fc2.forward(&h)?.gelu()?;
        self.fc3.forward(&h)
    }
}

pub struct TrainResult {
    pub train_hist: Vec<f32>,
    pub val_hist: Vec<f32>,
    pub val_inputs: Vec<[f32; N_FEATURES]>,
    pub val_truth: Vec<[f32; N_OUTPUTS]>,
    pub val_pred: Vec<[f32; N_OUTPUTS]>,
    pub field_rmse_over_range: [f32; 4],
}

fn rows_to_tensor<const N: usize>(rows: &[[f32; N]], device: &Device) -> Result<Tensor> {
    let n = rows.len();
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Ok(Tensor::from_vec(flat, (n, N), device)?)
}

fn column_stats<const N: usize>(rows: &[[f32; N]]) -> ([f32; N], [f32; N]) {
    let n = rows.len() as f32;
    let mut mean = [0.0f32; N];
    for row in rows {
        for i in 0..N {
            mean[i] += row[i];
        }
    }
    for i in 0..N {
        mean[i] /= n;
    }
    let mut var = [0.0f32; N];
    for row in rows {
        for i in 0..N {
            let d = row[i] - mean[i];
            var[i] += d * d;
        }
    }
    let mut std = [0.0f32; N];
    for i in 0..N {
        std[i] = (var[i] / n).sqrt();
    }
    (mean, std)
}

fn normalize_rows<const N: usize>(
    rows: &[[f32; N]],
    mean: &[f32; N],
    std: &[f32; N],
) -> Vec<[f32; N]> {
    rows.iter()
        .map(|r| {
            let mut out = [0.0f32; N];
            for i in 0..N {
                out[i] = (r[i] - mean[i]) / (std[i] + 1e-8);
            }
            out
        })
        .collect()
}

pub fn train() -> Result<TrainResult> {
    let device = Device::Cpu;
    println!("generating data...");
    let train_ds = build_dataset(N_TRAIN, SEED);
    let val_ds = build_dataset(N_VAL, SEED + 1);

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
    let mlp = Mlp::new(vb)?;

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

    println!("training for {EPOCHS} epochs, batch={BATCH}, N_train={N_TRAIN}");
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

            let pred = mlp.forward(&xb)?;
            let loss = (&pred - &yb)?.sqr()?.mean_all()?;
            opt.backward_step(&loss)?;
            epoch_losses.push(loss.to_scalar::<f32>()?);
        }
        let train_loss = epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32;

        let val_pred_n = mlp.forward(&x_val_t)?;
        let val_loss = ((&val_pred_n - &y_val_t)?.sqr()?.mean_all()?).to_scalar::<f32>()?;

        train_hist.push(train_loss);
        val_hist.push(val_loss);
        if epoch % 50 == 0 || epoch == EPOCHS - 1 {
            println!("  epoch {epoch:4}  train={train_loss:.4e}  val={val_loss:.4e}");
        }
    }

    // Final predictions, denormalized.
    let val_pred_n = mlp.forward(&x_val_t)?;
    let val_pred_vec: Vec<Vec<f32>> = val_pred_n.to_vec2::<f32>()?;

    let mut val_pred_arr: Vec<[f32; N_OUTPUTS]> = Vec::with_capacity(N_VAL);
    for row in val_pred_vec.iter() {
        let mut a = [0.0f32; N_OUTPUTS];
        for j in 0..N_OUTPUTS {
            a[j] = row[j] * (y_std[j] + 1e-8) + y_mean[j];
        }
        val_pred_arr.push(a);
    }

    // Per-field rMSE report on the unnormalized predictions vs truth.
    let field_names = ["rho", "u", "By", "p"];
    println!("\nval MSE per field (unnormalized):");
    let mut field_rmse = [0.0f32; 4];
    for (f, name) in field_names.iter().enumerate() {
        let mut sq = 0.0f64;
        let mut n = 0usize;
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for i in 0..N_VAL {
            for j in 0..N_OUT {
                let idx = f * N_OUT + j;
                let truth = val_ds.outputs[i][idx];
                let d = (val_pred_arr[i][idx] - truth) as f64;
                sq += d * d;
                n += 1;
                if truth < mn {
                    mn = truth;
                }
                if truth > mx {
                    mx = truth;
                }
            }
        }
        let mse = (sq / n as f64) as f32;
        let rng_ = mx - mn;
        let rmse_over_range = mse.sqrt() / rng_;
        field_rmse[f] = rmse_over_range;
        println!(
            "  {:>3}: MSE={:.4e}   range={:.3}   rMSE/range={:.3}",
            name, mse, rng_, rmse_over_range
        );
    }

    Ok(TrainResult {
        train_hist,
        val_hist,
        val_inputs: val_ds.inputs.clone(),
        val_truth: val_ds.outputs.clone(),
        val_pred: val_pred_arr,
        field_rmse_over_range: field_rmse,
    })
}
