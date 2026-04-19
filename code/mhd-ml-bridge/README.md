# mhd-ml-bridge — scaffolding for the ML method layer

**Status:** scaffolding. This is not science. It is the smallest end-to-end
pipeline — classical solver → training data → JAX MLP → predictions — required
to know the stack works before anything interesting is built on top of it.

## What this does

1. Takes the 1D ideal MHD HLL solver from `../mhd-1d/mhd.py`.
2. Sweeps 7 parameters around the canonical Brio-Wu initial condition
   (left/right `rho`, `p`, `By`, plus `Bx`).
3. Generates `N_TRAIN + N_VAL = 128` simulator runs at `t = 0.1`, downsampled
   to a 64-cell grid with 4 primitive fields (`rho`, `u`, `By`, `p`).
4. Trains a small MLP (`7 → 128 → 128 → 256`, GELU, Adam, MSE on z-scored
   outputs) to map parameters → final primitives.
5. Reports per-field validation MSE and emits two figures.

## What this is *not*

- Not physics-informed. No conservation laws in the loss. No PDE residual.
  A trained MLP here does not "understand" MHD; it interpolates between
  simulator outputs.
- Not a speed win over HLL. The solver is cheap; the MLP inference is
  comparable-ish; but the point of a surrogate is training cost amortized
  over many queries, and we haven't measured that.
- Not a generalization claim. The parameter sweep is a tight box around
  canonical Brio-Wu. OOD (e.g. flipped sign structure, different `gamma`)
  is not tested and would almost certainly fail.
- Not tuned. Hidden size, LR, epoch count picked to train in ~30 s on CPU,
  not to minimize validation error.

## Why build it anyway

The Week-4 plan calls this a *method-layer toy*. The purpose is to answer
three boring but load-bearing questions before Week 5+ builds physics-aware
surrogates:

1. Does JAX install, JIT, and train on this machine? (Yes.)
2. Does the data-from-classical-solver pattern compose cleanly with a JAX
   training loop? (Yes — see `data.py` importing from the sibling week-3
   project via `sys.path`.)
3. At what scale does a plain MLP saturate on interpolation in this narrow
   parameter window? (See numbers below; comfortably good, as expected
   when the manifold is low-dimensional.)

Knowing the stack works means the next iteration can be opinionated about
physics (conservation constraints, PINN residuals, operator-learning
architectures) rather than fighting tooling.

## File layout

- `data.py` — parameter sampling + `run_one` wrapper around the HLL MHD
  solver. `build_dataset(n_examples)` returns `(X, Y)` with shapes
  `(n, 7)` and `(n, 256)`.
- `surrogate.py` — raw-JAX MLP (Glorot init, GELU), optax Adam, per-field
  validation report, two figures under `figures/`.
- `requirements.txt` — numpy, matplotlib, jax, optax.
- `figures/training_curve.png` — loss vs epoch.
- `figures/val_example.png` — MLP prediction vs HLL truth on one validation
  shot, 4 panels.

## Reproducing

```
pip install -r requirements.txt
python surrogate.py
```

On a mid-2020s laptop CPU this runs in well under a minute.

## Numbers from the reference run (seed=0)

```
val MSE per field (unnormalized):
  rho: MSE=8.94e-06   range=1.089   rMSE/range=0.3%
    u: MSE=1.99e-05   range=1.220   rMSE/range=0.4%
   By: MSE=2.76e-05   range=2.388   rMSE/range=0.2%
    p: MSE=1.79e-05   range=1.125   rMSE/range=0.4%
```

Sub-percent relative RMS error per field is *not* a strong result — it
reflects how narrow the parameter box is. Widen the box and/or add
degenerate/shock-structure-flipping cases and the numbers will get worse
in an informative way. That is work for later.

## Roadmap beyond this

- Widen the parameter envelope until the MLP visibly fails.
- Replace the fixed-grid output with a neural-operator-style architecture
  (e.g. FNO or DeepONet) so resolution is not baked in.
- Add a conservation penalty (mass, momentum) to the loss and measure
  whether that helps OOD.
- Swap Brio-Wu for a 1D equilibrium problem with an actual plasma-physics
  interpretation (e.g. a screw-pinch cross-section).

None of that is in this scaffold and none of it is promised for Month 1.
