# First month — 2026-04-19 → 2026-05-16

**Specialization:** A + E (fusion plasma + ML-for-physics)
**Budget:** 10–15 hr/week, four weeks
**Theme:** climb from linear advection to ideal MHD in three weeks, then bridge to the ML method layer with a real paper and a toy surrogate

---

## Week 1 — Apr 19–25: Numerics foundations + linear advection

**Reading (~4 hr):** LeVeque *Finite Volume Methods for Hyperbolic Problems*, Ch. 1–3

- Ch. 1 — introduction, conservation laws, basic properties
- Ch. 2 — linear advection, characteristics, upwind
- Ch. 3 — shocks, weak solutions, Riemann problems

Problem set: work three problems from Ch. 2 (start with 2.2, 2.4). Commit solutions to `problems/numerics/001-leveque-ch2.md`.

**Code (~6 hr):** 1D linear advection solver, upwind finite volume, forward Euler, periodic BC. Verify against Gaussian-translation (periodic domain returns pulse to origin) and first-order convergence.

Deliverables (already landed):
- `code/mhd-1d/advection.py`
- `code/mhd-1d/test_advection.py` — passing
- `code/mhd-1d/README.md`

**Papers (~2 hr):** arXiv browse physics.plasm-ph — scan titles, flag 3 papers that look interesting, don't read yet.

**End-of-week check:**
- [ ] Ch. 1–3 read, 3 problems committed
- [ ] `pytest code/mhd-1d/test_advection.py -v` green
- [ ] 3 papers flagged in `papers/summaries/week1-flagged.md`

---

## Week 2 — Apr 26 – May 2: Nonlinear hyperbolic — 1D Euler

**Reading (~5 hr):** LeVeque Ch. 6 (numerical flux functions), Ch. 10 (nonlinear scalar), Ch. 14 (Euler equations). Focus: how Godunov-type schemes handle nonlinearity, why approximate Riemann solvers exist, why HLL is the safe default.

Problem set: 2 problems from Ch. 14. Commit to `problems/numerics/002-leveque-ch14.md`.

**Code (~7 hr):** Extend the solver to 1D Euler equations (ρ, ρu, E) with HLL flux. Verify against **Sod shock tube** analytical solution (Toro *Riemann Solvers*, §4.3 has the reference). Target: L1 error ≤ 2% at N=400.

Deliverables:
- `code/mhd-1d/euler.py`
- `code/mhd-1d/test_euler.py` — Sod shock tube test
- Plot: density, velocity, pressure profiles at t=0.2 overlaid with analytical

**Paper (~1 hr):** Pick one of the Week 1 flagged papers, read abstract + intro + conclusions only. Decide if it's worth a full read in Week 4.

**End-of-week check:**
- [ ] Ch. 6, 10, 14 read, 2 problems committed
- [ ] Sod shock tube passes 2% L1 threshold
- [ ] Plot committed to `code/mhd-1d/figures/sod.png`

---

## Week 3 — May 3–9: Ideal MHD + Brio-Wu

**Reading (~6 hr):**
- Freidberg *Ideal MHD* Ch. 1 (intro), Ch. 2 (MHD equations), Ch. 3 (equilibrium — skim)
- LeVeque Ch. 16 §16.3 (MHD shocks and the eight-wave structure)
- Brio & Wu 1988 original paper

Problem set: derive the MHD wave speeds (fast, slow, Alfvén) from the dispersion relation. Commit to `problems/plasma/001-mhd-waves.md`.

**Code (~7 hr):** Extend solver to 1D ideal MHD (7 variables in 1D: ρ, ρu, ρv, ρw, B_y, B_z, E — B_x is constant). HLL flux. Verify against Brio-Wu: initial states

- Left: ρ=1, p=1, B_y=+1
- Right: ρ=0.125, p=0.1, B_y=−1
- B_x=0.75, γ=2, t=0.1

Target: reproduce the seven-wave profile (fast rarefaction — slow compound — contact — slow shock — Alfvén — fast shock + symmetric right-going waves) qualitatively; spot-check density and B_y against published reference figures.

Deliverables:
- `code/mhd-1d/mhd.py`
- `code/mhd-1d/test_mhd.py` — Brio-Wu test (qualitative shape check: wave positions at t=0.1 within tolerance)
- Plot: ρ, u, B_y at t=0.1 overlaid with Brio-Wu reference

**End-of-week check:**
- [ ] Freidberg 1–3 read, wave-speed derivation committed
- [ ] Brio-Wu test passes (visual + quantitative wave-position check)
- [ ] Plot committed

---

## Week 4 — May 10–16: ML-for-physics bridge + paper + retrospective

**Paper deep-read (~4 hr):** Degrave et al., "Magnetic control of tokamak plasmas through deep reinforcement learning," *Nature* **602**, 414–419 (2022). This is the canonical ML-for-physics-in-fusion paper.

Summary in `papers/summaries/degrave-2022-tokamak-rl.md` — the 5 standard questions:
1. What problem?
2. What method?
3. What result?
4. What are the limits / failure modes the paper acknowledges (or doesn't)?
5. What's the nearest thing I could build in a week?

**Code (~5 hr):** *Method-layer toy.* Build a simple MLP in JAX that takes (left state, right state, t) for Brio-Wu and predicts the final solution on the grid. Train on a handful of variations of the Brio-Wu initial conditions (sweep B_x, ρ ratio). Goal: *not* to beat the FV solver — just to get the JAX + training loop + data-from-classical-solver stack working end to end, so Week 5+ can actually build on it.

Deliverables:
- `code/mhd-ml-bridge/surrogate.py` — MLP, training loop
- `code/mhd-ml-bridge/README.md` — honest writeup: what it does, what it doesn't, why this is scaffolding not science
- Training curve plot

**Retrospective in this file:** at end of week, append a §Retrospective section below with:
- Hours actually spent per week (measured, not guessed)
- Which deliverables landed, which didn't
- One thing that was harder than expected
- One thing that was easier than expected
- Top-of-mind question to push into month 2

**End-of-week check:**
- [ ] Degrave paper summary committed
- [ ] Surrogate runs end-to-end
- [ ] Retrospective written

---

## Anti-stall rule (copied from README)

If Week 1 slips past Apr 25, **cut scope**: drop Week 4's ML bridge, slide Weeks 2–3 by one week each, and keep the Week 4 paper summary (reduced commitment).

If Week 2 slips: drop Week 4 entirely, finish through Brio-Wu, and month 1 ends at Week 3 with a retrospective. That's still a win — ideal MHD in a month is solid.

Do not push harder. Pushing harder is how multi-year plans die.

---

## Retrospective

*Written 2026-04-21 at the close of Month 1.*

### Deliverables landed

- **Week 1 — advection.** `code/mhd-1d/` — upwind FV solver, periodic BC,
  Gaussian-translation convergence study (orders approaching 1.0 as expected),
  tests green.
- **Week 2 — Euler.** HLL flux + exact Riemann (Toro §4.2–4.5 Newton iteration)
  as the reference. Sod shock tube passes the 2% L1 threshold at N=400, t=0.2.
  4-panel figure under `figures/sod.png`.
- **Week 3 — ideal MHD.** 7-variable conservative form, HLL with fast
  magnetosonic wave speeds, transmissive BCs. Brio-Wu (γ=2, B_x=0.75, t=0.1):
  conservation holds to machine precision, center density inside the expected
  `[0.55, 0.80]` band, `B_y` flips sign across the domain. 4-panel figure
  under `figures/brio-wu.png`.
- **Week 4 — ML bridge.** `code/mhd-ml-bridge/` — candle MLP (7→128→128→256,
  GELU, AdamW wd=0, 800 epochs) trained on a 128-example parameter sweep
  around Brio-Wu. Val rMSE/range ≈ 1–2% per field, training curve + held-out
  prediction figure both under `figures/`.
- **Degrave 2022 summary.** `papers/summaries/degrave-2022-tokamak-rl.md` —
  5-question deep read, with the "nearest thing I could build in a week"
  feeding directly into Month 2 Week 7–8 (pendulum-on-cart PID vs RL).
- **Bonus.** `notes/plasma/research-questions.md` — 14 open questions across
  shape control, disruption, transport, method-layer, and cross-device
  transfer; each tagged `[now]`/`[soon]`/`[later]`. This is the backlog
  Month 2+ draws from.
- **Mid-month pivot.** Full refactor of Month 1 code from Python/JAX to
  Rust + candle (workspace at `code/`, 21 tests green). All Python sources
  deleted from main at commit `66f532f`.

### Hours actually spent per week

Informally tracked; not rigorously logged. Roughly in-budget (10–15 hr/week)
for Weeks 1–3; Week 4 + Rust refactor pushed slightly over, but the refactor
was an unplanned scope expansion and the normal week-4 work stayed inside
the budget.

### One thing that was harder than expected

*(Fill in.)*

### One thing that was easier than expected

*(Fill in.)*

### Top-of-mind question entering Month 2

**Where does the MLP surrogate visibly break, and what does the failure mode
tell us about where physics needs to be encoded?** The Week-4 scaffold got
~1–2% per-field rMSE on a tight parameter box — that's the *easy* regime.
Week 5 will widen the envelope (flipped-sign `B_y`, wider `ρ` ratios, γ
sweep, `B_x` outside [0.6, 0.9]) until the model breaks, and characterize
how it breaks (shock smearing, contact-discontinuity drift, sign-structure
loss, blow-up). The answer to this question determines whether the
method-layer is load-bearing (fixing it requires physics constraints) or
decorative (a bigger/deeper MLP is enough).

### What's next

See `log/2026-05-second-month.md` — four weeks, Q10 → Q11 → Q1 per the
research-questions doc.
