# 002 — Where the Brio-Wu MLP breaks

*Week 5, Q10. Written 2026-04-21.*

## Claim

The Week-4 MLP scaffold (`7 → 128 → 128 → 256`, GELU, AdamW `wd=0`,
trained on `N_train = 96` IC vectors inside a tight Brio-Wu box) is not
robust to *any* of the four physical knobs we widened. One knob fails
catastrophically; the others degrade monotonically as the IC moves away
from the training midpoint. The fact that all four fail tells us the
model memorized the distribution, not the physics.

## Evidence

Ran `cargo run -p mhd-ml-bridge --example sweep --release`, which takes a
single MLP trained on the Week-4 distribution and evaluates per-field
rMSE/range against HLL truth as each axis widens. `n_per_point = 16`,
seed deterministic.

Worst-case per-field rMSE/range per axis (higher = worse; 0.01 ≈ Week-4
in-distribution baseline):

| axis          | at value | rho   | u     | By    | p     | figure                                           |
|---------------|----------|-------|-------|-------|-------|--------------------------------------------------|
| rho_ratio     | 0.400    | 0.182 | 0.223 | 0.066 | 0.133 | `code/mhd-ml-bridge/figures/sweep/rho_ratio.png`      |
| **by_sign_flip** | 1.000 | **0.322** | **0.530** | **0.149** | **0.487** | `code/mhd-ml-bridge/figures/sweep/by_sign_flip.png`   |
| gamma         | 3.000    | 0.023 | 0.140 | 0.019 | 0.059 | `code/mhd-ml-bridge/figures/sweep/gamma.png`          |
| bx            | 0.000    | 0.068 | 0.251 | 0.073 | 0.085 | `code/mhd-ml-bridge/figures/sweep/bx.png`             |

Two worst-case prediction-vs-truth panels live at
`figures/sweep/by_sign_flip_worst_example.png` and
`figures/sweep/rho_ratio_worst_example.png`.

### Axis-by-axis read

- **by_sign_flip (catastrophic).** The training data had `By_L > 0` and
  `By_R < 0` for *every* example; flipping those signs is the one
  perturbation the model has literally no gradient information about.
  Velocity rMSE/range goes to 53%, pressure to 49%. The `By` field
  itself is "only" 15% off — probably because the MLP can at least get
  the magnitude roughly right by treating `|B_y|` as the relevant input,
  even though the structure is wrong.
- **rho_ratio (graceful degradation).** At the trained midpoint
  (ρ_R ≈ 0.125) errors stay near 1–2%. Pushing ρ_R to 0.40 reduces the
  density ratio from ~8 to ~2.5; the fast-shock position shifts
  substantially and the MLP can't track it without retraining.
  Pushing ρ_R to 0.01 (stronger shock) also hurts but less dramatically
  — the shock structure is still qualitatively what the model saw.
- **bx (Euler limit is out-of-distribution).** At `B_x = 0` the MHD
  system degenerates: there's no coupling between `u` and `B_y` through
  the Lorentz term, so the solution is essentially Euler-plus-passive-`B_y`.
  The MLP never saw this regime and misses the velocity profile by 25%.
  At `B_x = 1.5` (strong guide field) the errors are milder (~5–8%).
- **gamma (surprising resilience).** γ enters the flux nonlinearly but
  the Brio-Wu wave *positions* don't move as much with γ as with the
  density ratio or field structure. Max 14% on velocity at γ=3.0.

The training-envelope bands on each figure make this visible: errors
are flat inside the band and climb monotonically outside.

## Implication for Week 6

Three candidate fixes, ranked by how much physics they'd encode:

1. **Wider training distribution.** Cheapest, most obvious. Would
   substantially knock down `rho_ratio`, `bx`, and `gamma` failures, and
   would fully fix `by_sign_flip` if the training distribution explicitly
   covers both sign configurations. This is *not* science — it's
   scaffolding hygiene. Do it anyway; any Week-6 architecture should be
   benchmarked against an MLP trained on the wider distribution, not the
   current narrow-box one.

2. **Neural operator (DeepONet).** Only buys us something if the failure
   mode is about the *output* representation (fixed-grid MLP can't
   interpolate a resolution change). Our failures are about the *input
   distribution*, not the output grid — so DeepONet on this data would
   probably match MLP performance inside the box and fail the same way
   outside it. Worth implementing *anyway* for Week 6 because it gives
   us a reference point for the architecture-vs-data tradeoff; don't
   expect it to beat wider-MLP on these axes.

3. **Conservation-penalty / symmetry-aware architecture.** The
   `by_sign_flip` failure is specifically a broken equivariance: the
   ideal-MHD equations are invariant under `B_y → -B_y` (with a matching
   flip of `B_z`), but an MLP whose inputs are the raw IC has no way to
   enforce that. A sign-equivariant architecture (e.g. an MLP on
   `|B_y|` + a separate sign head, or a group-equivariant design) would
   fix `by_sign_flip` *structurally*, not distributionally. This is the
   scientifically interesting option — but it's an aspirational Week 6+
   target, not a same-week deliverable.

### Decision

Week 6 DeepONet implementation goes ahead as planned, but with one
scope change: train it on the *widened* distribution (both `By` sign
configurations + widened `ρ_R` + `B_x` gradient). That gives us a clean
MLP-vs-DeepONet compare on the same realistic dataset, rather than
shadowboxing the narrow Week-4 box. The symmetry-aware idea is logged
for Month 3 along with the conservation-penalty loss (Q12).

## Reproduce

```bash
cargo run -p mhd-ml-bridge --example sweep --release
```

Trains the Week-4 MLP (≈ 25 s on CPU), then runs all four sweeps and
produces the figures cited above. Random seed is pinned.

## See also

- `notes/plasma/research-questions.md` §Q10 — the question this was
  supposed to answer.
- `code/mhd-ml-bridge/src/sweep.rs` — harness source.
- `log/2026-05-second-month.md` Week 6 — the DeepONet plan this feeds into.
