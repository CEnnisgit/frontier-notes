# Research questions — fusion plasma + ML-for-physics

Living doc. Questions I want to push on, grouped by tractability and topic.
Specialization is A+E per `decisions/001-specialization.md`; nothing here
should drift outside that envelope without first updating the decision doc.

**Convention:** each question tagged `[now]`, `[soon]`, or `[later]`.

- `[now]` — I could start this inside Month 2 with the tools I have.
- `[soon]` — needs one more foundational block (2D MHD, a bigger simulator,
  a specific dataset) before it's attackable.
- `[later]` — needs infrastructure, collaborators, or scale I don't have.
  Tracked so I don't forget, not to work on.

---

## A. Shape & position control (the Degrave regime)

The regime where ML-for-plasma has a real result. The interesting
questions now are about *what's left* after Degrave 2022.

1. **[now]** Can a classical LQR/MPC controller, tuned with modern automatic
   differentiation against the same reduced-physics simulator (FGE-like),
   close the gap with the RL policy on the "easy" shapes? I.e. is the RL
   win really about the *algorithm* or about the *optimization target and
   simulator quality*? A clean head-to-head study would tell you which
   piece is load-bearing. I can do a toy version of this on
   pendulum-on-cart in a week; the plasma version needs a shared simulator.

2. **[soon]** Does the sim-to-real "classical trim" used in Degrave have a
   principled replacement? The trim corrects a distribution shift; adding
   it as part of the training loop (domain randomization, adaptive policy)
   is the obvious move but wasn't the paper's approach. How much of the
   trim can be absorbed into training, and at what cost?

3. **[later]** Transfer: train a single policy on TCV, fine-tune to
   DIII-D and ASDEX-U with shot-count budgets measured. This is the
   right long-horizon question for the approach, and exactly what
   Seo et al. 2024 begins to attack.

## B. Disruption prediction & avoidance

The other ML-for-tokamak success thread (Kates-Harbeck 2019, Abbate 2021).
Prediction is solved-ish on individual devices; avoidance (close the loop
back to actuators) is the live frontier.

4. **[soon]** In published disruption predictors, how much of the
   performance comes from features the physicist would have chosen vs.
   features the neural net discovered? A systematic ablation study would
   be useful — both for interpretability and for knowing what matters
   when you port to a device with different diagnostics.

5. **[soon]** "Prediction" usually means "n-second warning"; the value
   of that warning depends on what actuator response is available in
   n seconds. What is the joint Pareto front of prediction horizon vs.
   mitigation effectiveness? The prediction-side papers don't compute
   this; the hardware-side papers (shattered pellet injection timing)
   don't compute it either; it lives in the gap.

6. **[later]** Unified detection+mitigation: the paper pattern is "RL
   avoids *one* instability" (Seo 2021 — tearing modes). A single policy
   that routes between avoidance strategies depending on detected
   precursor type would be the natural next step but is a 3-5 person-year
   program, not a solo month.

## C. Transport, confinement, MHD — the hard core

The physics the Degrave simulator explicitly doesn't capture. Also the
set of problems where ML has *not* had a clean win yet. This is the
research-question I'd most like to push on, but it's also the least
tractable solo.

7. **[soon]** Is the L-H transition predictable from upstream scalars
   (ne, Te, B, P_heating) in an interpretable way, and does a neural
   surrogate add anything over the existing scaling laws (Martin 2008
   et al.)? This is a concrete "neural network vs. power-law fit"
   benchmark on a famous physics problem. Data is public on some
   devices.

8. **[soon]** On the reduced-MHD side: can a neural operator (FNO,
   DeepONet) learn the 1D screw-pinch stability boundary in
   (pressure gradient, safety factor, resistivity) from a small dataset
   of eigenvalue computations? This is the first tractable step past
   "interpolate final-state snapshots" toward "learn an operator".
   Requires a 1D eigenvalue solver I don't have yet — that's the Month 2
   blocker.

9. **[later]** Turbulent transport coefficients from first-principles
   gyrokinetic simulations, surrogate-modeled. This is a "stand on
   the shoulders of GENE / CGYRO / GKW" project and needs access to
   their output archives.

## D. Method-layer — surrogate models, PINNs, neural operators

Where the tooling questions live. Less about fusion specifically,
more about "what does an ML model even need to be, to be useful in
physics?"

10. **[now]** On the Brio-Wu surrogate I just built: widen the parameter
    envelope until the MLP *visibly* fails. Identify the failure mode —
    is it the fixed-grid output, the smoothness assumption, or the
    low-dimensional parameterization? A good failure plot is more
    informative than another accuracy number.

11. **[now]** Replace the fixed-grid output with a neural-operator-style
    architecture (DeepONet or a tiny FNO in 1D). Does it train on the
    same data? Does it generalize better across `Bx`? This is the
    natural Week-5-or-6 task.

12. **[soon]** Conservation-penalty loss: add a term that penalizes
    mass/momentum/energy drift between predicted profiles. Does the
    surrogate become more robust OOD? Does it lose interpolation
    quality? This is the cleanest "does physics-informed actually help"
    mini-experiment.

13. **[later]** PINNs as PDE solvers: the literature is mixed at best
    (Karniadakis 2021, versus critiques from the numerical-analysis
    community about conditioning, gradient pathologies, etc.). Worth
    understanding deeply before committing, and worth *not* building
    on if the honest answer is "FV wins". Reading list first, code
    later.

## E. Cross-device transfer & scale-up to ITER/SPARC

14. **[later]** All of the above questions have an "and does it
    transfer to ITER?" appendix. ITER has longer timescales, different
    diagnostics, and (crucially) exists in simulation right now, not
    in hardware. Any useful ML-for-ITER work is a 2027+ question
    unless the simulator side matures.

---

## Where to start in Month 2

Candidates ranked by (interest × tractability):

1. **Question 10** — stress-test the Brio-Wu surrogate until it
   breaks. Week 5. Low risk, high information yield; directly builds
   on what just shipped.
2. **Question 11** — FNO/DeepONet on the same data. Week 6. Natural
   next step and teaches a new architecture.
3. **Question 1** — LQR/MPC vs RL head-to-head on pendulum-on-cart.
   Week 7-8. A cleaner methodological study than any of the plasma
   questions and a useful reference point for everything else.

Questions 4, 7, and 12 go into the Month-3 queue.

---

## Pruning rules

- If a question has been sitting at `[later]` for three months without
  moving, re-examine whether it should stay on the list.
- If I start a `[now]` question and discover it needs something from
  `[soon]`/`[later]`, downgrade it and note the blocker.
- No question should be here without a sentence on *why it matters*
  and *how I'd know I was done*. If I can't write those two sentences,
  the question isn't clear enough yet.
