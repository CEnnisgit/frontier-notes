# mhd-1d

A from-scratch 1D finite-volume solver, built up in three stages:

| Stage | Equation | File | Test |
|-------|----------|------|------|
| Week 1 | Linear advection: `∂u/∂t + c ∂u/∂x = 0` | `advection.py` | `test_advection.py` |
| Week 2 | 1D Euler (compressible gas) | `euler.py` | `test_euler.py` |
| Week 3 | 1D ideal MHD | `mhd.py` | `test_mhd.py` |

The final target is the Brio-Wu shock tube — the canonical 1D ideal-MHD benchmark (Brio & Wu, *JCP* 1988). Each stage is verified against an analytical solution or published reference before moving on.

Week 4 bridges this classical solver to the ML method layer (see `code/mhd-ml-bridge/`).

## Design choices

- **NumPy, not JAX.** Week 1–3 prioritize correctness and readability; vectorized NumPy is plenty for 1D. Week 4 introduces JAX when differentiability is actually needed.
- **Upwind FV + forward Euler for Week 1.** Simplest stable scheme. First-order accurate, diffusive, good teaching vehicle.
- **HLL flux for Weeks 2–3.** Positivity-preserving, no knowledge of the full eigenstructure needed, works fine for Sod and Brio-Wu.
- **Periodic BC for advection, transmissive for Euler/MHD shock tubes.** Matches the analytical/reference solutions.

## Running

Install deps once:

```bash
cd code/mhd-1d
pip install -r requirements.txt
```

Run the solver with its built-in convergence study:

```bash
python advection.py
```

Run the tests:

```bash
python -m pytest -v
```

## What "verified" means here

- **Advection:** L2 error bounded after one periodic revolution, empirical convergence order ~1.0 for upwind+Euler.
- **Euler (Week 2):** L1 error ≤ 2% vs Sod analytical at N=400, t=0.2.
- **MHD (Week 3):** Brio-Wu wave positions match published reference within one cell width.

If a test fails, debug — do not weaken the threshold without writing down why.
