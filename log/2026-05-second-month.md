# Second month — 2026-04-26 → 2026-05-23

**Specialization:** A + E (fusion plasma + ML-for-physics)
**Budget:** 10–15 hr/week, four weeks
**Language:** Rust (candle for ML, ndarray for numerics, plotters for figures)
**Theme:** method-layer deep dive — stress-test the Week-4 surrogate until
it breaks, try a neural operator on the same data, then open the RL-vs-
classical-control comparison toy from the Degrave read

The source of deliverable shape is `notes/plasma/research-questions.md`:
**Q10 → Q11 → Q1** per its own "Where to start in Month 2" ranking, user-
confirmed via AskUserQuestion.

---

## Week 5 — Apr 26 – May 2: Q10 — stress-test the Brio-Wu MLP

**Goal.** Find where the Week-4 surrogate breaks and characterize *how* it
breaks. Same model, harder data.

**Reading (~2 hr):** Skim LeVeque Ch. 16 §16.3 again, specifically the wave
structure regimes of ideal MHD Riemann problems. Read the relevant §1–2 of
Li 2005 "An HLLC Riemann solver for MHD" (arxiv:math/0411389) for the
standard OOD failure catalog.

**Code (~8 hr):**
- `code/mhd-ml-bridge/src/sweep.rs` — parameter-sweep harness reusing
  `build_dataset` from `data.rs`, with a `wide_sample_params` variant that
  broadens each range one axis at a time: wider ρ ratio, sign-flipped `B_y`,
  γ sweep, `B_x` outside `[0.60, 0.90]`.
- `examples/sweep.rs` — runs the harness, generates per-field error heatmaps
  as a function of each widened axis.
- No retraining yet: the Week-5 experiment is the *same* trained model
  evaluated on harder inputs. Week 6 may retrain on the union.

**Writeup (~2 hr):** `notes/plasma/002-mlp-failure-modes.md`. Claim → evidence
→ what this tells us about where to add physics. One page. Include the
per-axis heatmaps and one or two val-example plots showing specific failures
(e.g. "at γ=1.4 the MLP predicts the wrong plateau density").

**Deliverables:**
- `code/mhd-ml-bridge/src/sweep.rs`
- `code/mhd-ml-bridge/examples/sweep.rs` — produces figures under
  `code/mhd-ml-bridge/figures/sweep/`
- `notes/plasma/002-mlp-failure-modes.md`

**End-of-week check:**
- [ ] Sweep harness runs to completion
- [ ] At least three distinct failure modes documented with figures
- [ ] `cargo test -p mhd-ml-bridge` still green

---

## Week 6 — May 3–9: Q11 — small neural operator (DeepONet)

**Goal.** Implement a minimal DeepONet in candle on the same Brio-Wu
parameter space; compare to the MLP on in-distribution accuracy and
Week-5 OOD failure modes.

**Reading (~3 hr):** Lu et al. 2021 "Learning nonlinear operators via
DeepONet" (Nature Machine Intelligence). Focus on §2 (architecture) and
the shock-tube example in the supplement.

**Code (~7 hr):**
- `code/mhd-ml-bridge/src/operator.rs` — DeepONet = branch net over the
  7-parameter IC × trunk net over grid coordinates `x`. Use
  `candle_nn::linear` stacks for both (GELU, same hidden widths as the
  MLP for a fair compare).
- `examples/train_operator.rs` — training loop, same optimizer and epoch
  budget as `train.rs`.
- Evaluate on the Week-5 OOD sweep.

**Writeup (~2 hr):** Append §Comparison to
`code/mhd-ml-bridge/README.md` — does the operator generalize better
across grid refinement? Across `B_x`? Numbers honestly reported, including
if DeepONet *loses* to the MLP on this dataset (plausible — operator-
learning only wins when the function space matches the data).

**Deliverables:**
- `code/mhd-ml-bridge/src/operator.rs`
- `code/mhd-ml-bridge/examples/train_operator.rs`
- README §Comparison with both models' numbers on the same held-out set

**End-of-week check:**
- [ ] DeepONet trains end-to-end
- [ ] README block has both models' per-field rMSE on in-distribution
- [ ] OOD numbers reported for both

---

## Week 7 — May 10–16: Q1 part 1 — pendulum-on-cart + PID baseline

**Goal.** Build the toy environment and the classical-control baseline that
Week 8's RL policy gets compared against. This is the nearest in-a-week
Degrave-style comparison.

**Reading (~3 hr):** Åström & Murray *Feedback Systems* Ch. 6 (state
feedback, LQR) and Ch. 11 (PID). Skim the Degrave 2022 control
architecture section one more time.

**Code (~8 hr):** new subproject `code/rl-control-toy/`.
- `src/env.rs` — pendulum-on-cart dynamics with an electromagnetic-style
  actuator model (coil voltage → force via a first-order lag), transmissive
  noise on state estimates, actuator saturation.
- `src/pid.rs` — hand-tuned PID for upright stabilization + LQR for the
  linearized regime. Swing-up + catch documented as a separate regime
  (the nonlinear boundary).
- `tests/env.rs` — basic conservation/linearization sanity tests.
- `examples/pid_demo.rs` — 3-panel plot: PID holds upright under
  (a) clean, (b) noisy, (c) perturbed cases.

**Deliverables:**
- `code/rl-control-toy/Cargo.toml`, `src/env.rs`, `src/pid.rs`,
  `tests/env.rs`, `examples/pid_demo.rs`
- Figure under `code/rl-control-toy/figures/pid.png`

**End-of-week check:**
- [ ] `cargo test -p rl-control-toy` green
- [ ] PID demo runs and produces the 3-panel plot
- [ ] Environment added to the workspace `members` list

---

## Week 8 — May 17–23: Q1 part 2 — RL policy + head-to-head + retrospective

**Goal.** Train a simple actor-critic on the Week-7 environment; produce
the Degrave-style comparison on the same robustness scenarios.

**Reading (~2 hr):** SAC paper (Haarnoja 2018) or the 200-line from-scratch
PPO blog post of choice. Goal: enough to write ~200 LoC of from-scratch
candle RL, *not* to pull an off-the-shelf RL library (transparency > speed
of development here).

**Code (~8 hr):**
- `code/rl-control-toy/src/rl.rs` — from-scratch actor-critic in candle.
  Replay buffer, target networks if SAC, GAE if PPO.
- `examples/train_rl.rs` — training loop.
- `examples/compare.rs` — evaluates PID + LQR + RL on the same clean/noisy/
  perturbed scenarios; produces a 3×3 comparison figure.

**Writeup (~2 hr):** `notes/plasma/003-rl-vs-classical-toy.md`. One page.
Where does RL win (generalizing across operating regimes)? Where does
classical win (robustness in a single regime, stability guarantees)?
What does this toy actually teach us about the Degrave regime? Be honest
if the numbers don't support a clean narrative.

**End-of-month retrospective:** append §Retrospective to this file with
hours actually spent per week, deliverables landed vs not, one harder /
one easier, top-of-mind question entering Month 3 (tentatively Q7 on L-H
transition scaling or Q12 on conservation-penalty loss — the real answer
comes out of what Week 5's failure modes turn up).

**Deliverables:**
- `code/rl-control-toy/src/rl.rs`
- `code/rl-control-toy/examples/{train_rl,compare}.rs`
- `notes/plasma/003-rl-vs-classical-toy.md`
- §Retrospective in this file

**End-of-week check:**
- [ ] RL training runs end-to-end
- [ ] Comparison figure has all three controllers in all three scenarios
- [ ] Retrospective written with specific content per section

---

## Commit cadence

One commit per week, titled `"Week N: <deliverable>"`, each self-contained
and green on `cargo test -p <crate>` where tests exist. Push at end of each
week to `origin/main`.

---

## Anti-stall rules (copied from Month 1, tightened)

- If Week 5 slips past May 2: **cut scope from Week 6.** Do a smaller
  DeepONet (fixed small trunk) or skip DeepONet and spend Week 6 on a
  conservation-penalty MLP variant (Q12) instead. Do *not* also slip
  Weeks 7–8.
- If Week 6 slips: drop Weeks 7–8's RL track, keep the pendulum env + PID
  as a one-week deliverable, end Month 2 at Week 7 with a retrospective.
- Do not push harder. The cost of pushing is not this month's deliverables —
  it's the 18 months after.

---

## Retrospective

*(Fill in at end of Week 8.)*
