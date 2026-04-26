# AGENTS.md

## Environment
- Prefer `./run_firedrake_container ...` for solver work. It mounts the repo at `/home/firedrake/shared`, sets that as the container cwd, and pins `firedrakeproject/firedrake:2025.4.2`.
- Treat `make build/run` as stale until fixed: `Makefile` expects a top-level `Dockerfile`, but the repo only has `.devcontainer/Dockerfile`.
- If Docker/X11 plotting fails, run `xhost +local:docker` on the host.

## Repo Shape
- This is a script repo, not a packaged Python project. Run solver files from the repo root, e.g. `python3 src/chns.py`, not `python -m ...`; imports are plain `from utils import ...` and some paths are repo-root relative.
- `src/allen_cahn.py`: periodic Allen-Cahn benchmark on the unit square; supports circle and deterministic random initial data; writes `output/allen_cahn-<initial>.pvd` plus optional PNG/PDF snapshots.
- `src/ch.py`: periodic Cahn-Hilliard benchmark; writes `output/ch.pvd` and `output/ch-t*.pdf`.
- `src/chns.py`: intended canonical CHNS baseline.
- `src/square.py`, `src/stabilization.py`, `src/parallel.py`: experimental CHNS variants; do not assume they implement the same formulation.
- `src/utils.py` contains the shared `refine_bary()` helper.
- `make_allen_cahn_mp4` and `make_chns_mp4` build MP4 animations from generated PNG frames using `ffmpeg`.
- `geometry/` contains checked-in `.geo` / `.msh` assets; no scripted mesh-generation pipeline was found.
- `report/allen_cahn.tex`, `report/ch.tex`, and `report/chns.tex` are part of the deliverable. `output/ch.pdf` and `output/chns.pdf` are tracked compiled PDFs.

## Current Priorities
- Open GitHub issues are the current source of truth for project priorities: `#1-10` cover CHNS math/logic, `#11-15` cover repo/tooling/presentation work.
- Allen-Cahn and Cahn-Hilliard runs are useful reduced testbeds for debugging phase-field discretizations before touching the fully coupled CHNS solver.
- For numerical CHNS work, start with canonical single-bubble or two-bubble benchmarks before many-bubble runs.
- Fine-mesh blow-up is an active unresolved problem in this repo. Do not treat coarse-mesh success as validation.
- `src/square.py` uses many random bubbles. Make centers deterministic before using it for comparisons or paper figures.

## Verification
- There is no pytest or typecheck setup. CI only runs `pylint $(git ls-files '*.py')`.
- Smallest meaningful smoke runs are from the repo root inside the Firedrake environment:
  - `./run_firedrake_container mpiexec -n 4 python3 src/allen_cahn.py --initial circle --steps 1 --no-output --snapshot-times`
  - `./run_firedrake_container python3 src/ch.py`
  - `./run_firedrake_container python3 src/chns.py`
  - `./run_firedrake_container mpiexec -n 2 python3 src/parallel.py`
- For Allen-Cahn work, report the free energy, `phi` min/max, and whether the solution leaves the expected `[-1, 1]` range.
- When changing CHNS numerics, report phase mass, `phi` min/max, a divergence-related metric, nonlinear failure step, and bubble height/center of mass.

## Artifacts
- Do not assume `output/` is disposable: compiled report PDFs are tracked there, and solver scripts also write `.pvd`, PNG/PDF snapshot, and MP4 files into the same directory.
- `report/graphics/` contains tracked figures and animations used by the README/report.
- `src/ch.py` uses `src/presentation.mplstyle`, which enables TeX rendering. Keep plotting runs inside an environment with TeX packages installed.
- `.vscode/settings.json` sends LaTeX Workshop output to `output/`; `make clean` only removes LaTeX aux files from `output/`.

## Numerical Debugging Protocol
- Treat numerical debugging as experimental science, not generic coding. Do not jump straight to a fix.
- For any CH, Allen-Cahn, or CHNS failure, first restate the observed symptom precisely: what run, which parameters, which step/time, and what quantity fails.
- Work from hypotheses. For each plausible explanation, write down:
  - Hypothesis
  - Mechanism
  - Minimal test
  - Expected outcome
  - Interpretation
- Prefer controlled experiments over guesses. Every claim should be tied to a measurable quantity or a direct code-path inspection.
- Instrument before modifying the formulation. If diagnostics are missing, add the smallest possible prints/logs first.

## Invariants And Diagnostics
- Always check physically or numerically meaningful invariants before changing the solver.
- Allen-Cahn:
  - free energy should decay for the backward-Euler benchmark unless solver failure corrupts the step
  - `phi` should usually stay close to `[-1, 1]`; large overshoots indicate time-step or nonlinear-solve trouble
- Cahn-Hilliard:
  - phase mass should stay constant up to solver/discretization tolerance
  - free energy should be nonincreasing for the intended stable discretizations
- CHNS:
  - phase mass should stay constant up to tolerance
  - free energy / kinetic behavior should be monitored together with nonlinear convergence
  - incompressibility should be checked through a divergence metric
- If the code is not already printing the relevant quantities, instrument it first. Prefer direct prints or JSON summaries over elaborate logging frameworks.

## Reduction Strategy
- Reduce aggressively before touching the coupled solver.
- Preferred order in this repo:
  - Allen-Cahn on the periodic square
  - Cahn-Hilliard on the periodic square
  - canonical CHNS single-bubble benchmark
  - canonical CHNS two-bubble benchmark
  - only then many-bubble or experimental variants
- If a reduced problem already fails, suspect formulation, discretization, or solver setup before blaming the full coupling.
- Use coarse meshes and short runs first. If a coarse run succeeds and a fine run fails, treat that as evidence of a stability/scaling issue, not validation.

## Experiment Design
- Prefer small parameter sweeps over one-off runs.
- Sweep only one axis at a time when possible:
  - `dt`
  - mesh resolution
  - interface width `epsilon`
  - mobility scaling
  - solver tolerances / nonlinear iteration caps
- When proposing a test, predict what should happen under each competing hypothesis.
- Analyze raw outputs, tables, and time series. Avoid vague summaries like "it blows up" without step/time and diagnostics.

## Code Style For Debug Scripts
- For debugging experiments, optimize for the smallest script that tests one idea.
- Prefer:
  - one file
  - top-to-bottom execution
  - hard-coded parameters
  - direct prints of mass, energy, bounds, residuals, and convergence reasons
- Avoid building reusable frameworks, configuration systems, or abstractions for one-off numerical tests.
- Keep canonical solver changes surgical. Do not refactor working code unless the experiment shows a concrete need.

## Firedrake / Phase-Field Checks
- Explicitly inspect for:
  - wrong weak-form signs or missing terms
  - incorrect test-function pairings
  - inconsistent time discretization between coupled terms
  - missing nullspace handling where required
  - mobility or interface-width scaling mistakes
  - nonlinear solver stagnation masked as a physics failure
  - boundary-condition inconsistencies
- Common hypotheses to consider first:
  - time step too large
  - non-energy-stable discretization
  - incorrect chemical potential or Allen-Cahn reaction term
  - solver tolerance / nonlinear convergence failure
  - mesh-resolution sensitivity
  - unstable CH/NS coupling

## Expected Response Pattern
- When another agent is asked to debug a numerical problem in this repo, the default structure should be:
  - precise restatement of the issue
  - 3-6 hypotheses
  - minimal experiments to distinguish them
  - instrumentation plan if diagnostics are missing
  - only then a minimal code change
- The goal is to reduce the search space of the bug efficiently, not to produce a polished large-scale rewrite.
