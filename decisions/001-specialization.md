# Decision 001 — Pick a specialization track

**Status:** open
**Date:** 2026-04-19
**Owner:** Cenni

## Why this decision matters

Going PhD-deep in "energy and transportation" as a whole is a decade+ project per sub-field. Picking one primary specialization (with literacy in adjacent ones) is the difference between shallow tourism and real contribution. Five candidates are compared below with identical structure, then scored on a matrix at the end.

The rubric:
- **Leverage** — if you succeed, how much does the world change?
- **Tractability** — how much of this is limited by physics vs. engineering vs. budget?
- **Fit** — alignment with your stated interests (energy + transportation + honest curiosity about anti-gravity-adjacent frontier).
- **Crossover** — does this track touch both energy and transportation, or only one?
- **Time-to-first-contribution** — how fast a PhD-capable person can do original work after committing.

---

## Track A — Fusion plasma physics

### Core problem
Confine a plasma at fusion-relevant conditions (nτT ≳ 3×10²¹ keV·s/m³ for D-T) long enough to produce net energy. Two live paradigms: magnetic confinement (tokamaks, stellarators) and inertial confinement (laser or pulsed-power driven). The engineering bottleneck has shifted from "can we do it" (NIF ignition Dec 2022; JET Q≈0.67; SPARC targeting Q>2 mid-decade) to "can we do it economically, continuously, and at grid scale."

### Physics required
MHD (Freidberg), kinetic theory (Nicholson, Krall & Trivelpiece), plasma transport (Helander & Sigmar for neoclassical), atomic physics for radiation loss, neutronics for blanket design. Heavy stat mech and E&M foundations. QM shows up in DFT for plasma-facing materials. GR and QFT not load-bearing.

### Computational stack
Fluid/MHD: BOUT++, NIMROD, M3D-C1. Kinetic/PIC: Gkeyll, XGC, WarpX. Gyrokinetic: GENE, GYRO, CGYRO. Equilibrium: EFIT, FreeGS, VMEC. Neutronics: OpenMC, Serpent, MCNP. Most legacy is Fortran + MPI; newer codes use C++ + Kokkos/CUDA. Python for analysis (OMFIT, IMAS data schema). Julia gaining ground (Plasma.jl, GGDUtils.jl).

Scale: everything from laptop-scale 1D solvers to exascale (ITER integrated modeling runs on Frontier-class machines).

### Time-to-first-contribution
For a PhD-capable entrant: 6–12 months to understand one code well enough to contribute a merged feature. 18–24 months to produce an original result worth a preprint. Access to experimental data (C-Mod archive, DIII-D collaborator program, SPARC via CFS if you're industry-side) accelerates this a lot.

### 5-year frontier (plausible to contribute to)
- Surrogate models for plasma turbulence (FNO/DeepONet-style) replacing gyrokinetic runs in integrated simulations
- Stellarator optimization (the SIMSOPT / stellopt revolution is live)
- HTS magnet design, quench modeling, fault analysis
- Disruption prediction (ML on experimental archives)
- Breeding blanket neutronics + tritium self-sufficiency
- Aneutronic (p-B11) — TAE, HB11 — still a harder physics problem; open but risky bet

### Out of reach as a solo researcher
Building a tokamak. Owning the plasma. Anything requiring a billion-dollar device's operating time unless you're inside a collaboration.

### Industrial vs academic
Both are live. Academic: PPPL, MIT PSFC, UKAEA, Max Planck IPP, Princeton/UW/UCSD groups. Industry: CFS, Helion, TAE, Tokamak Energy, Type One Energy, Shine, Realta Fusion, Pacific Fusion. Industry pays more and moves faster; academia has better deep-thinking tolerance. Both hire from the same pipeline.

### Failure mode
The "pick a fusion concept and defend it" trap. Concepts go in and out of favor; the skills (MHD, kinetic theory, numerical methods, experimental diagnostics) transfer across. Specialize in the *methods*, not the *machine*.

### First-month on-ramp project
Implement a 1D MHD solver (ideal MHD, advection + Alfvén waves + Brio-Wu shock tube test). ~200 lines of Python/JAX. Verify against Brio-Wu analytical solution. This is the "hello world" of magnetized plasma sim.

---

## Track B — Advanced in-space propulsion

### Core problem
Getting between places in the solar system in reasonable times with reasonable mass. Chemical rockets give Isp ≈ 300–450 s; this is a thermodynamic near-ceiling. Real gains require *either* much higher exhaust velocity (electric propulsion: Isp 1000–10000 s) or higher power-to-mass (nuclear thermal, nuclear electric, fusion drives). Transportation to Mars in weeks rather than months, or to outer planets in years rather than decades, lives here.

### Physics required
Rocketry + orbital mechanics (Sutton & Biblarz, Vallado), plasma physics (electric propulsion is a plasma problem), thermodynamics (nuclear thermal is a thermo problem), neutronics (for NTP/NEP reactors), heat transfer + materials (everything is temperature-limited). Control theory for trajectory optimization. For fusion-drive and beamed propulsion: all of Track A or comparable.

### Computational stack
Trajectory optimization: GMAT, Copernicus, POLIASTRO, cislunar-toolkit, Astrogator. Differentiable: JAX-based trajectory optimizers (brandon-rhodes/Skyfield for ephemerides, custom JAX for optimization). Electric propulsion plasma: HPHall, DRACO, WarpX adapted for Hall thrusters, COLISEUM. Neutronics for NTP: OpenMC/Serpent. CFD for nozzles: OpenFOAM, SU2. Materials: Quantum ESPRESSO for plasma-wall interaction.

### Time-to-first-contribution
6 months to a useful trajectory optimizer contribution. 12–18 months to a research-grade EP simulation result. Hardware contributions require lab access — LPP, Georgia Tech HPEPL, PPPL's electric thruster lab.

### 5-year frontier
- Closed-loop trajectory optimization with differentiable sim + RL (SpaceX's landing is the canonical example; solar-system scale is open)
- EP lifetime modeling (erosion physics, 1000s-hour test replacement)
- Nuclear thermal propulsion revival (DRACO program, Ultra Safe Nuclear)
- Direct fusion drive (Princeton Satellite Systems, Helion adjacencies)
- Pulsed plasma / MPDT / VASIMR engineering
- Beamed energy / laser sail physics (Breakthrough Starshot regime)

### Out of reach
Actually flying the thing without an agency or a company. Anything requiring a radioisotope source without NRC entanglement. Serious hardware lifetime testing is equipment-gated.

### Industrial vs academic
Industry dominant now: SpaceX, Blue Origin, Relativity, Stoke, Ad Astra, USNC, Ultra Safe, Helion (propulsion adjacency), Longshot. NASA NIAC funds the wilder end. Academic: Georgia Tech HPEPL, MIT SPL, Stanford SPRG, Michigan PEPL.

### Failure mode
Falling into a single pet concept (VASIMR has burned multiple careers; EM-drive burned several as well). Also: confusing "cool simulation" with "works in vacuum."

### First-month on-ramp project
JAX-based differentiable low-thrust trajectory optimizer: Earth → Mars transfer with constant-Isp electric propulsion, minimizing propellant mass subject to arrival date. Benchmark against a published reference (Kéchichian or textbook). ~500 lines. Exposes orbital mechanics, optimization, and the Isp/T/m trade-space at once.

---

## Track C — Superconductors + power electronics

### Core problem
Move electrical energy without losses; generate fields larger than iron can hold. HTS (REBCO, BSCCO) tapes at commercial scale unlock: compact fusion magnets (CFS's SPARC bet is 90% a magnet bet), grid-scale DC transmission, all-electric aviation, high-power maglev. Room-temperature ambient-pressure superconductors remain the open grail — LK-99 was noise, but the search space is vast and materials ML is real.

### Physics required
Solid-state (Ashcroft & Mermin, Coleman *Introduction to Many-Body Physics*), superconductivity (Tinkham, Annett), electromagnetism (Jackson), power electronics (Erickson), materials science. DFT + many-body methods for novel superconductors. Statistical mechanics for phase transitions. QFT (condensed matter flavor) for pairing mechanisms.

### Computational stack
DFT: Quantum ESPRESSO, VASP, ABINIT. Many-body: TRIQS, iQIST, DFT+DMFT. Materials ML: Matbench, JARVIS, M3GNet, MACE, CHGNet. Power systems: OpenDSS, PSCAD, PowerFactory. Magnet modeling: COMSOL (commercial), getDP/gmsh (open), custom FEM.

### Time-to-first-contribution
DFT materials search: 3–6 months to a first candidate prediction, 1–3 years until anyone synthesizes it (and most predictions don't survive). Magnet design / quench modeling: 6–12 months to a real contribution in a collaboration. Power electronics: faster, more incremental.

### 5-year frontier
- ML-accelerated materials discovery for superconductors (both electron-phonon and unconventional)
- HTS fusion magnet reliability, quench protection, AC loss modeling
- Solid-state transformers for grid + charging infrastructure
- All-electric aviation powertrains (cryogenic motors + HTS)
- Superconducting digital electronics (SFQ for quantum control + HPC)

### Out of reach
Running a synthesis lab alone. Building a >20 T magnet solo. Any experimental program requiring kg of REBCO tape without industrial partnership.

### Industrial vs academic
Industrial: CFS (magnets), Tokamak Energy, AMSC, Faraday Factory (tape), VEIR (cable), SuperGrid, Hyperloop-adjacent firms (mostly dead), Wright Electric, MagniX. Academic: MIT FBML, NHMFL (Florida), Brookhaven, Argonne, Karlsruhe KIT.

### Failure mode
Getting seduced by room-temperature-superconductor hype cycles and burning months on the latest LK-99-style claim. Materials ML looks like a shortcut but most predictions fail at synthesis — keep one foot on experimental ground.

### First-month on-ramp project
Implement the BCS gap equation solver (self-consistent, finite-T) for a single band with a given electron-phonon coupling. Verify Tc against analytical weak-coupling limit. ~150 lines. Then extend to two-band (MgB₂-style) and compare to experimental Tc. Exposes the mean-field machinery that underlies most of the field.

---

## Track D — Gravitation & precision tests of GR

### Core problem
GR is the best-tested theory we have but it's probably not final — it doesn't quantize cleanly, and dark energy / dark matter / hierarchy-problem hints sit uncomfortably within it. The frontier: gravitational wave astronomy (LIGO/Virgo/KAGRA now, LISA in the 2030s, Einstein Telescope / Cosmic Explorer beyond), precision equivalence-principle tests (MICROSCOPE successor missions), fifth-force / modified-gravity phenomenology at short range, tests of Lorentz invariance. This is where the honest version of your anti-gravity question lives: *is GR complete, and if not, what deviations are experimentally accessible?*

### Physics required
GR (Carroll → Wald → MTW), QFT (Peskin → Srednicki → Weinberg), effective field theory of gravity (Donoghue lectures), cosmology (Baumann, Dodelson), differential geometry (Lee). Statistical mechanics for GW data analysis. Numerical relativity for black hole binaries (Baumgarte & Shapiro).

### Computational stack
Numerical relativity: SpEC (Caltech), Einstein Toolkit, GRChombo, SXS. GW data analysis: LALSuite, PyCBC, Bilby, GWpy. Parameter estimation: dynesty, emcee, bilby. Theoretical: xAct/xPert (Mathematica), OGRe (symbolic GR in Julia), GRTensor.

### Time-to-first-contribution
GW data analysis: 6–9 months (lots of open data, Bilby makes contribution accessible). Numerical relativity: 1–2 years; the codes are deep. Modified gravity phenomenology: variable — theorist's path, gated by physics taste more than tooling.

### 5-year frontier
- LISA preparation — theoretical templates, waveform modeling for EMRIs
- Beyond-GR templates in LIGO O5/O6 data
- Primordial black hole constraints from GW + pulsar timing
- Quantum gravity phenomenology: tabletop tests (Bose-Marletto-Vedral, Carney, Penrose collapse models)
- Casimir / vacuum-energy precision experiments
- Equivalence principle tests at 10⁻¹⁷ and below

### Out of reach
Building a detector. Most of the interesting experimental work is collaboration-gated (LIGO is >1500 authors). But data analysis and theory are more open than any other track here.

### Honest anti-gravity assessment (belongs here)
- **Alcubierre warp metric**: mathematically valid GR solution, requires negative energy density in bulk (not just virtual Casimir between plates). No known matter has this property. Modifications (Van Den Broeck, Lentz, Bobrick-Martire) reduce the required energy but not to zero. Not a near-term engineering target; may be a productive thought-experiment for constraining quantum gravity.
- **EM drive**: falsified. Tajmar / Dresden 2021 showed the claimed thrust was thermal artifact.
- **Woodward / Mach-effect thrusters**: repeated independent replication attempts fail. The theoretical basis (Sciama-Mach principle of inertia) is at best speculative.
- **Zero-point energy extraction**: you can extract work *once* from a Casimir configuration. It's not a renewable source.
- **"Anti-gravity" as pop-culture means it**: no experimental evidence. GR would need modification of a kind that would show up in tests that have been run with extraordinary precision.

The productive version of the anti-gravity interest is: work inside GR and its extensions, *and* stay in touch with experiment. If at some point the frontier starts demanding exotic matter or extra dimensions to explain an observation, that's where the real door opens — and the only way to be standing at that door is to have done the foundational work. This track puts you there.

### Industrial vs academic
Almost entirely academic + government lab. Caltech, MIT LIGO Lab, Penn State, University of Florida, Albert Einstein Institute Potsdam/Hannover, Perimeter Institute, Cambridge DAMTP. Space: NASA Goddard + JPL for LISA prep, ESA.

### Failure mode
Getting pulled into the UAP / UFO / alternative-physics adjacent communities. The experimental literature there is noise; engagement costs credibility you'll need later. Also: theory-only paths without computational or observational grounding become untethered fast.

### First-month on-ramp project
Download a segment of public LIGO O3 data (GWOSC). Use PyCBC or Bilby to run parameter estimation on a known binary black hole event (GW150914 is the classic teaching example). Reproduce the published mass/spin posteriors. ~200 lines, heavy on the libraries. Gives you a live connection to real data and the full pipeline from strain to inference.

---

## Track E — ML for physics (cross-cutting method track)

### Core problem
Most physics simulations are stuck at "accurate but too slow" or "fast but too crude." Neural surrogates (FNO, DeepONet), physics-informed NNs (PINNs), and differentiable simulation are changing that: fusion control, weather/climate, molecular dynamics, CFD, and materials discovery all have real results now. This is a *method track* — you bring it to whichever physical domain you care about. As a standalone specialization, it's also a legitimate CS/ML research direction with active venues (ICLR, NeurIPS, MLSP).

### Physics required
Same foundations as whatever domain you apply it to — you can't do ML-for-fluids without fluids. Plus: optimization theory (Nocedal & Wright), deep learning foundations (Goodfellow or current standard), Gaussian processes and Bayesian inference, numerical analysis (because PINNs are a numerical method, not just a NN).

### Computational stack
JAX + Equinox + Optax for differentiable sim and PINNs. PyTorch for most ML research codebases. JAX-MD (molecular dynamics), Brax (rigid body), JAX-CFD (fluids), NVIDIA Modulus (commercial but capable). Neural operator libraries: neuraloperator (FNO), DeepXDE, PDEBench (benchmarks). HPC: GPU clusters, with FSDP / DeepSpeed for large models.

### Time-to-first-contribution
3–6 months to a published-at-a-workshop-level result on a well-defined benchmark. Faster than any of the other tracks, but the results can be shallower — fast iteration cuts both ways.

### 5-year frontier
- Foundation models for physics (GraphCast-style for fluids, climate, plasma)
- Neural operators as production-grade surrogates in engineering pipelines
- Differentiable simulation for inverse design (materials, antennas, thrusters, reactors)
- PINNs finally becoming competitive with classical solvers on real problems (currently they mostly aren't)
- Closed-loop RL control of real physical systems (plasma control on DIII-D, tokamak ramp-down, thermal management)

### Out of reach
Theoretical originality in ML (mostly dominated by a small number of groups); surprise breakthroughs (GPT-scale shifts) that redefine the field mid-project.

### Industrial vs academic
Both are live and the lines are blurry. Industry: DeepMind science, NVIDIA, Microsoft Research AI4Science, Meta FAIR Chemistry, Google Accelerated Science. Academic: CMU, MIT CSAIL, Stanford, Princeton, Caltech, Cambridge, ETH. Startups: Atomic AI, Isomorphic, Orbital Materials, Extropic.

### Failure mode
Becoming an ML generalist who does "physics experiments" that are mostly benchmarks. The fix: always co-advise with a domain expert, always verify against the classical solver, always report where the ML approach *fails* alongside where it succeeds. Also: PINN papers have a reputation problem (many results don't reproduce or don't beat classical methods). Be rigorous, not hypey.

### First-month on-ramp project
Implement a Fourier Neural Operator for 2D Darcy flow (the canonical FNO benchmark from Li et al. 2020). Reproduce the published error metrics. ~300 lines in JAX or PyTorch. Then do an ablation — what happens when you break the Fourier assumption, reduce resolution, change PDE? This reveals what FNO is actually doing vs. what people claim.

---

## Decision matrix

Score 1–5. "Fit" scored against *your* stated interests (energy + transportation + honest anti-gravity curiosity).

| Track | Leverage | Tractability | Fit | Crossover | Speed-to-contrib | Total |
|-------|----------|--------------|-----|-----------|------------------|-------|
| A — Fusion | 5 | 3 | 4 | 3 (energy-primary, some propulsion) | 3 | **18** |
| B — Propulsion | 4 | 4 | 5 | 2 (transport-primary) | 4 | **19** |
| C — Superconductors+power | 4 | 3 | 3 | 5 (both) | 3 | **18** |
| D — Gravitation/GR | 2 | 4 | 5 (honest anti-grav) | 1 (neither directly) | 3 | **15** |
| E — ML-for-physics | 4 | 5 | 3 | 5 (method, goes anywhere) | 5 | **22** |

The raw numbers say E (ML-for-physics) or B (propulsion) — but the matrix hides an important structural fact: **E is a method, not a domain.** The natural combination is **E + one other track**. The two strongest combined bets:

1. **A + E** — fusion with an ML-for-physics method emphasis. Largest leverage; industry desperately wants this; the hottest research area in computational plasma right now.
2. **B + E** — in-space propulsion with differentiable simulation + trajectory optimization + RL for control. Fastest path to something you can actually fly or feed to industry. Closest fit to the "transportation" half of your stated goal.

Track D stays on the table as a **parallel side-read** (see Phase 5 of the main roadmap) — the honest anti-gravity question lives there, but it's a poor primary bet if the goal is to solve energy/transport problems.

## Recommendation

**Primary: Track A (fusion) + Track E (ML-for-physics) as the method layer.**

Why:
- Highest direct leverage on the energy half of your goal.
- ML-for-physics has real, published frontier work in fusion *right now* (TAE's neural controller, DeepMind + Max Planck's tokamak control paper, fusion surrogates via FNO). You can be contributing in months, not years.
- Fusion crosses over into propulsion via direct fusion drive — Track B doesn't go away, it sits downstream.
- Track D stays in rotation as background reading; the honest anti-gravity engagement happens alongside, not instead of, productive work.

**If the transportation half pulls harder than the energy half**, swap to **Track B + Track E**: JAX-based trajectory optimization, EP plasma simulation with neural surrogates, RL for thruster control. Same method track, different domain anchor.

## Open questions for Cenni

- Which domain anchor — A (fusion) or B (propulsion) — grips you more on the gut level? (Gut-level matters; 2+ years is long.)
- Any constraint I'm missing? Need to stay employable in a specific city? Need to avoid any dual-use / ITAR-adjacent areas? Hardware access or lack thereof?
- Comfortable with the rec, want to flip to B+E, or want to push back on the framing entirely?

Once this is decided, next artifact is `log/2026-04-first-month.md` with the concrete 10–15 hr/week plan and the first on-ramp project initialized in `code/`.
