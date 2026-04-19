# Degrave et al. 2022 — Magnetic control of tokamak plasmas through deep RL

- **Citation:** Degrave, J., Felici, F., Buchli, J., Neunert, M., Tracey, B., Carpanese, F., Ewalds, T., Hafner, R., Abdolmaleki, A., de las Casas, D., Donner, C., Fritz, L., Galperti, C., Huber, A., Keeling, J., Tsimpoukelli, M., Kay, J., Merle, A., Moret, J.-M., Noury, S., Pesamosca, F., Pfau, D., Sauter, O., Sommariva, C., Coda, S., Duval, B., Fasoli, A., Kohli, P., Kavukcuoglu, K., Hassabis, D., & Riedmiller, M. "Magnetic control of tokamak plasmas through deep reinforcement learning." *Nature* **602**, 414–419 (2022). [doi:10.1038/s41586-021-04301-9](https://doi.org/10.1038/s41586-021-04301-9)
- **Authors' affiliations:** DeepMind + Swiss Plasma Center (EPFL, Lausanne)
- **Hardware:** TCV (Tokamak à Configuration Variable) — 1.54 m major radius, highly shape-flexible
- **Read-by:** Cenni, 2026-04-19

## One-line claim

Model-free deep RL, trained entirely in a simulator, can drive a real tokamak's 19 poloidal field coils at 10 kHz to reach a range of plasma shapes — including a two-plasma "droplet" configuration — without per-shape handwritten controllers.

## Why this paper matters (context)

Tokamak shape and position control has historically been the dominion of **classical feedback control**: LQR, PID, model-predictive, gain-scheduled controllers per plasma scenario. Each new configuration (elongation, triangularity, divertor geometry) required substantial engineering work to design a controller — mostly because the plasma-coil response is nonlinear, time-varying, and state-dependent, and because stability margins are scenario-specific.

This paper is the first demonstration that a **single learned policy** can cover a *family* of shapes — and reach some (the "droplet") that were never demonstrated on TCV with classical methods. It's also a cleanly-scoped sim-to-real result in a regime where sim-to-real is usually hard (real plasmas are noisy, the simulator isn't full-physics, the control loop is hard-real-time at 10 kHz).

For ML-for-physics more broadly, this is the "breakthrough demonstration" paper — the one you cite to show that neural controllers are a real tool in fusion, not speculation.

## What they did (method)

**Simulator.** They built a reduced-physics free-boundary equilibrium evolution simulator — essentially a 2D axisymmetric Grad-Shafranov solver coupled to circuit equations for the coil system. Fast enough to roll out millions of trajectories; simple enough that they don't capture transport, turbulence, or kinetic effects (which would be far too expensive for RL training at scale).

**Policy architecture.** A deep actor-critic network (following MPO — Maximum a posteriori Policy Optimization, an off-policy actor-critic algorithm DeepMind developed). Inputs are ~92 magnetic diagnostic measurements; outputs are 19 poloidal-field coil voltages. Reward shaped around a target plasma shape trajectory (shape error, limits on coil currents, position penalties).

**Training.** Distributed RL training across many simulator instances on TPUs. Curriculum: start with simple objectives, gradually demand more complex shapes. They used **episode-length randomization** and perturbations to help robustness.

**Deployment.** The trained policy ran at 10 kHz on TCV's real-time control system. Critically, the RL policy was **augmented with a classical feedback trim** — a small correction loop on top that handled the inevitable sim-to-real gap. The paper is explicit about this: the RL output alone was not accurate enough for some of the more demanding configurations; the trim made it work.

## What they found (results)

On real hardware they demonstrated:

1. **Baseline single-null elongated plasma** (standard scenario) — classical controllers also do this, but the RL one did it without per-shape tuning
2. **"Negative triangularity"** — a shape of growing interest because it suppresses edge-localized modes (ELMs)
3. **"Snowflake"** divertor — a configuration with multiple X-points, hard to stabilize classically
4. **"Droplet"** — two separate plasma columns in the same vacuum vessel, simultaneously. This is the configuration classical controllers had never achieved on TCV, and it's the crown-jewel demonstration of the paper

The shape-error metrics (how closely the plasma boundary matches the target at a set of control points) were generally within ~1 cm across the shapes they attempted, which is competitive with handcrafted controllers for the easier scenarios and *better* for the harder ones where classical tuning was the limiting factor.

## What the paper carefully doesn't claim

Reading a Nature paper for what it *doesn't* claim is half the work. Here's what's missing:

- **No claim about the hard fusion problems.** This is shape and position control only. The actual scientific limits on fusion — confinement quality, disruption prediction/avoidance, transport, heat exhaust, plasma-wall interaction — are untouched. Shape control was a solved problem classically (modulo per-shape engineering cost); this makes it easier, not fundamentally better.
- **No robustness guarantees.** Classical controllers come with stability-margin theorems; the RL policy has none. "It worked on the shots we tried" is not the same as "it won't catastrophically fail on an out-of-distribution scenario."
- **No claim about scaling to larger tokamaks.** TCV is small. ITER and SPARC have longer thermal time constants, different coil topologies, different disruption risks. Nothing in this paper proves the approach transfers.
- **No standalone deployment.** As noted above, the RL output was trimmed by a classical controller. Pure-RL deployment on a real tokamak has not been demonstrated in this paper.
- **No claim about the simulator being adequate.** They explicitly acknowledge the FGE simulator doesn't capture transport. For shape control this is arguably fine (shape is dominated by equilibrium, not transport), but for anything else it won't be.

None of these are flaws in the paper — the authors are careful. But the headline narrative around the paper ("RL is solving fusion") substantially oversold what's actually shown.

## My critical read

**Load-bearing innovation:** The combination of (a) a reduced-physics simulator that's good enough for shape control *and* fast enough for RL, and (b) demonstrating that the sim-to-real gap can be closed with a modest classical trim. This is a transferable recipe, not just a one-off demonstration.

**Load-bearing caveat:** The demonstration is that **shape control is amenable to RL**, not that **fusion control is amenable to RL**. Shape control is a linear-ish problem in a 10-kHz regime where the underlying physics (magnetics, circuit equations) is well-understood and in-sim. The open problems (disruptions, transport, MHD instabilities, ELMs) involve physics that is either not in the simulator or not well-understood at all — very different regime.

**What's underrated here:** The systems engineering — running a deep net at 10 kHz on a plasma experiment is non-trivial; the TCV team has a lot of infrastructure built up. The DeepMind contribution is the RL machinery; the SPC contribution is the control-system plumbing without which nothing works. This is the right pattern for ML-for-physics collaborations and is worth studying in its own right.

**What I'd look for in a follow-up paper:**
- Disruption prediction *and* avoidance (currently separate papers; uniting them would be meaningful)
- Control during heating transients (neutral beam ramp-up, ion cyclotron ramp)
- Transfer learning between tokamaks (train on TCV, evaluate on DIII-D or ASDEX-U)
- Physics-informed reward shaping (shape-error is easy; ITER-relevant metrics like heat flux footprint are hard)
- Integration with disruption-mitigation hardware (shattered pellet injection etc.) — closing the loop between detection and response

## Nearest thing I could build in a week

**Toy proxy: pendulum-on-cart with electromagnetic-style actuation, controlled by a PID and by RL.**

Rationale: the point isn't to replicate Degrave at home (impossible without a tokamak) but to **feel the failure modes** of RL vs classical control on a nonlinear system. Pendulum-on-cart is:
- Nonlinear (like a plasma)
- Has multiple operating regimes (upright vs. swing-up vs. parked at bottom — like different plasma shapes)
- Has instabilities (like a plasma)
- Has both "easy to hand-design a controller" and "tricky enough that handcrafting a controller for the full envelope is painful"

Deliverables for a week:
- `code/rl-control-toy/env.py` — pendulum-on-cart dynamics with a simple "coil voltage → force" actuator model
- `code/rl-control-toy/pid.py` — hand-tuned PID for upright stabilization + swing-up
- `code/rl-control-toy/rl.py` — MPO-style actor-critic (or simpler: SAC or PPO) learning the same task
- Comparison experiment: robustness to (a) sensor noise, (b) actuator saturation, (c) a sudden mass perturbation (like a plasma disruption)

The point: get a gut feel for *where* RL wins and *where* classical control wins. The Degrave paper's answer — "RL wins on generalizing across configurations, classical wins on robustness in a single config" — should be reproducible in miniature.

Later, once the 1D MHD solver is mature, a more direct analogue: a toy 1D plasma column with an equilibrium position controlled by coil currents, with position-setpoint tracking as the task. The jump from pendulum-on-cart to 1D plasma is mostly bookkeeping at that point.

## Tracked for further reading

- **Seo et al. 2024** (*Nature Machine Intelligence*, DIII-D) — RL for plasma shape control on a second tokamak; tests whether the Degrave approach transfers.
- **Abbate et al. 2021** (*Nuclear Fusion*) — ML disruption prediction (the other ML-for-tokamak success story, pre-dating Degrave).
- **Seo et al. 2021** (*Nuclear Fusion*) — avoidance of tearing modes via RL.
- **Kates-Harbeck et al. 2019** (*Nature*) — the pre-cursor deep-learning disruption prediction paper.
- **Boyer et al. 2021+** — Princeton work on RL for control of NSTX-U.

These would form the core "ML for tokamak control" syllabus if the next paper-deep-read points in this direction.
