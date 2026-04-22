# 003 — DeepONet vs wide-MLP on the widened Brio-Wu distribution

*Week 6, Q11. Written 2026-04-21.*

## Claim

Architecture choice is the wrong lever for the Brio-Wu surrogate at this
data scale. A DeepONet (branch over the 7-parameter IC × trunk over the
grid coordinate, same optimizer and epoch budget as the Week-4 MLP,
trained on the widened Week-6 distribution) *underperforms* a plain MLP
trained on the same data. It loses in-distribution *and* fails the Q10
sweep axes by at least as much as the wide MLP. Confirms the Q10 note's
prediction (`notes/plasma/002-mlp-failure-modes.md` §Implication for
Week 6): failure modes are about *training-distribution coverage*, not
about output representation, so swapping architectures doesn't help.

## Evidence

Both models: `N_train = 96`, `N_val = 32`, AdamW `wd=0`, 800 epochs,
batch = 16, lr = 1e-3, seed = 0. Widened distribution randomizes the
sign of `By_L` (with `By_R` always opposite), widens `rho_R ∈ [0.05, 0.40]`,
and widens `B_x ∈ [0.0, 1.5]`. DeepONet: branch `7 → 128 → 128 → 256`
(split into 4 fields × 64-dim latent), trunk `1 → 64 → 64 → 64`,
combined by per-field inner product.

### In-distribution validation (widened dist, val set seed=1)

| model        | rho   | u     | By    | p     |
|--------------|-------|-------|-------|-------|
| wide MLP     | 0.021 | 0.060 | 0.065 | 0.037 |
| DeepONet     | 0.033 | 0.084 | 0.121 | 0.060 |

DeepONet is ~1.5–2× worse per field. Training curve at
`code/mhd-ml-bridge/figures/operator/training_curve.png`. Validation
shot at `code/mhd-ml-bridge/figures/operator/val_example.png`.

### Q10 sweep (same four axes, `n_per_point = 16`, seed = 42)

Worst-case `sum(per-field rMSE/range)` over all sweep values:

| axis          | wide MLP | DeepONet | figure                                            |
|---------------|----------|----------|---------------------------------------------------|
| rho_ratio     | 0.435    | 0.505    | `code/mhd-ml-bridge/figures/operator/sweep_compare.png` |
| by_sign_flip  | 0.231    | 0.359    | same                                              |
| gamma         | 0.308    | 0.321    | same                                              |
| bx            | 0.324    | 0.441    | same                                              |

Wide MLP beats DeepONet on all four axes. Widening the training
distribution did what Q10 predicted it would — `by_sign_flip` drops from
1.00 (narrow-MLP) to 0.23 (wide MLP), because the wide distribution now
contains both sign configurations. DeepONet on the same widened data
only gets `by_sign_flip` to 0.36, worse than the MLP on every axis.

## Why DeepONet loses here

Three plausible reasons, not mutually exclusive:

1. **Rank bottleneck.** Fixed `LATENT_D = 64` per field means each
   output shot is reconstructed from the inner product of two
   64-dimensional vectors. The wide MLP has no such bottleneck — its
   final layer maps `128 → 256` directly per sample. For a fixed-grid
   `N_OUT = 64` output, that bottleneck buys nothing and costs
   expressiveness.
2. **Data scale.** DeepONet's advertised advantage — generalizing to
   new grid coordinates — is orthogonal to our actual failure mode
   (IC distribution coverage). With only 96 training examples, the
   branch net has less signal per parameter than a comparable MLP.
3. **Optimizer budget.** Both models trained identically; DeepONet's
   val loss is still noticeably higher at epoch 800 than its train
   loss. A longer run might close the gap in-distribution but wouldn't
   change the sweep result, which is data-limited.

Only (1) and (2) are architectural-vs-architectural arguments. The
conservation-penalty / equivariance path from Q10's third option
remains the next structurally-interesting thing to try.

## Implication for Week 7+

- **Architecture swaps are not the bottleneck.** Skip further operator
  variants (FNO, etc.) until the data-scale and structural-prior
  questions are answered.
- **Conservation penalty (Q12)** is now the top Week-7 candidate. The
  `by_sign_flip` equivariance failure is a specific structural hole a
  loss term could fix, independent of how big the training set is.
- **Data-scale study** is worth a half-day: does 96 → 384 training
  examples on the wide distribution close the wide-MLP vs DeepONet gap
  or widen it? That disentangles (2) from (1).

## Reproduce

```bash
cargo run -p mhd-ml-bridge --example train_operator --release
```

Trains both models back-to-back (~2 min CPU total), writes three figures
under `code/mhd-ml-bridge/figures/operator/`, and prints the comparison
tables in this note.

## See also

- `notes/plasma/002-mlp-failure-modes.md` — the Q10 writeup that
  predicted this result and motivated training on the widened
  distribution.
- `code/mhd-ml-bridge/src/operator.rs` — DeepONet implementation.
- `code/mhd-ml-bridge/README.md` §Comparison — numbers in a more
  consumable form.
- `log/2026-05-second-month.md` Week 7 — conservation-penalty plan the
  data-scale finding above argues for.
