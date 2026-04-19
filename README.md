# research

A long-horizon study + work repo on **energy and transportation** problems.

Target competency: PhD-level in both software and physics, across both domains. Anti-gravity is on the list as an honest-frontier question, not a default assumption.

## How this repo is organized

- `decisions/` — decision docs (ADR-style). The first one picks a specialization track.
- `notes/` — per-topic study notes (classical mech, E&M, QM, stat mech, GR, QFT, plasma, numerics, ideas).
- `problems/` — worked textbook problem sets.
- `code/` — simulations and solvers. Each project has its own subdirectory with a `README.md` and verification test.
- `papers/` — arXiv summaries (PDFs are gitignored; only summaries + notes are tracked).
- `log/` — weekly retrospectives. Anti-stall artifact.

## Collaboration model

Working with Claude as:

- **Study partner** — work problems together, walk through derivations, catch reasoning errors.
- **Code collaborator** — build solvers together; every solver gets a verification test against an analytical solution or published benchmark before it's trusted.
- **Literature scout** — track the frontier via arXiv, summarize papers, flag the 2–3/week worth reading.
- **Idea pressure-tester** — stress-test speculative ideas against conservation laws, experimental bounds, and replication history. Blunt but constructive.

## Time budget

10–15 hours/week. First 30 days after specialization is picked: ~1 textbook chapter + 1 small coding exercise + 2–3 paper summaries per week.

## Current state

- [x] Repo skeleton
- [ ] Specialization decision (`decisions/001-specialization.md`)
- [ ] First-month plan (`log/2026-04-first-month.md`) — written after specialization is picked
- [ ] First on-ramp project in `code/`

## Anti-stall rule

If no commits for 2 weeks, re-scope instead of pushing harder. Ambitious plans fail from over-commitment, not under-effort.
