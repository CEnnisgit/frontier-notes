# mhd-ml-bridge — scaffolding for the ML method layer

**Status:** scaffolding. This is not science. It is the smallest end-to-end
pipeline — classical solver → training data → candle MLP → predictions —
required to know the stack works before anything interesting is built on
top of it.

## What this does

1. Takes the 1D ideal MHD HLL solver from the sibling `mhd-1d` crate.
2. Sweeps 7 parameters around the canonical Brio-Wu initial condition
   (left/right `rho`, `p`, `By`, plus `Bx`).
3. Generates `N_TRAIN + N_VAL = 128` simulator runs at `t = 0.1`, downsampled
   to a 64-cell grid with 4 primitive fields (`rho`, `u`, `By`, `p`).
4. Trains a small MLP (`7 → 128 → 128 → 256`, GELU, AdamW with `wd=0`,
   MSE on z-scored outputs) mapping parameters → final primitives.
5. Reports per-field validation MSE and writes two figures.

## What this is *not*

- Not physics-informed. No conservation laws in the loss. No PDE residual.
  A trained MLP here does not "understand" MHD; it interpolates between
  simulator outputs.
- Not a speed win over HLL. The solver is cheap; the MLP inference is
  comparable-ish; the point of a surrogate is training cost amortized
  over many queries, and we haven't measured that.
- Not a generalization claim. The parameter sweep is a tight box around
  canonical Brio-Wu. OOD (flipped sign structure, different `gamma`) is
  not tested and would almost certainly fail.
- Not tuned. Hidden size, LR, epoch count picked to train in well under
  a minute on CPU, not to minimize validation error.

## Why build it anyway

The purpose is to answer three boring but load-bearing questions before
Week 5+ builds physics-aware surrogates:

1. Does candle install, forward/backward, and train on this machine? (Yes.)
2. Does the data-from-classical-solver pattern compose cleanly with a
   candle training loop? (Yes — `data.rs` calls `mhd_1d::mhd::mhd_simulate`
   via a normal crate dependency.)
3. At what scale does a plain MLP saturate on interpolation in this narrow
   parameter window? (See numbers below; a few percent per-field rMSE on
   the held-out set, which is the expected order for a low-dim manifold.)

Knowing the stack works means the next iteration can be opinionated about
physics (conservation constraints, PINN residuals, operator-learning
architectures) rather than fighting tooling.

## File layout

- `src/data.rs` — parameter sampling + `run_one` wrapper around the HLL
  MHD solver. `build_dataset(n_examples, seed)` returns a `Dataset` with
  `inputs: Vec<[f32; 7]>` and `outputs: Vec<[f32; 256]>`.
- `src/surrogate.rs` — candle MLP, AdamW, per-field validation report.
- `examples/train.rs` — end-to-end entrypoint that trains and plots.
- `figures/training_curve.png` — loss vs epoch.
- `figures/val_example.png` — MLP prediction vs HLL truth on one
  validation shot, 4 panels.

## Reproducing

```
cargo run -p mhd-ml-bridge --example train --release
```

On a mid-2020s laptop CPU this runs in well under a minute.

## Numbers from the reference run (seed=0)

```
val MSE per field (unnormalized):
  rho: MSE=2.56e-04   range=1.099   rMSE/range=0.015
    u: MSE=2.88e-04   range=1.125   rMSE/range=0.015
   By: MSE=4.88e-04   range=2.365   rMSE/range=0.009
    p: MSE=2.05e-04   range=1.125   rMSE/range=0.013
```

A few-percent relative RMS error per field is *not* a strong result — it
reflects how narrow the parameter box is and how little training data is
used. Widen the box and/or add degenerate/shock-structure-flipping cases
and the numbers will get worse in an informative way. That is Week 5.

## Roadmap beyond this

- Widen the parameter envelope until the MLP visibly fails (Week 5).
- Replace the fixed-grid output with a neural-operator-style architecture
  (e.g. DeepONet) so resolution is not baked in (Week 6).
- Add a conservation penalty (mass, momentum) to the loss and measure
  whether that helps OOD.
- Swap Brio-Wu for a 1D equilibrium problem with an actual plasma-physics
  interpretation.

None of that is in this scaffold and none of it is promised for Month 1.
