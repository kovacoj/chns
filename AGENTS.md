# AGENTS.md

## Environment
- Prefer `./run_firedrake_container ...` for solver work. It mounts the repo at `/home/firedrake/shared`, sets that as the container cwd, and pins `firedrakeproject/firedrake:2025.4.2`.
- Treat `make build/run` as stale until fixed: `Makefile` expects a top-level `Dockerfile`, but the repo only has `.devcontainer/Dockerfile`.
- If Docker/X11 plotting fails, run `xhost +local:docker` on the host.

## Repo Shape
- This is a script repo, not a packaged Python project. Run solver files from the repo root, e.g. `python3 src/chns.py`, not `python -m ...`; imports are plain `from utils import ...` and some paths are repo-root relative.
- `src/ch.py`: periodic Cahn-Hilliard benchmark; writes `output/ch.pvd` and `output/ch-t*.pdf`.
- `src/chns.py`: intended canonical CHNS baseline.
- `src/square.py`, `src/stabilization.py`, `src/parallel.py`: experimental CHNS variants; do not assume they implement the same formulation.
- `src/utils.py` contains the shared `refine_bary()` helper.
- `geometry/` contains checked-in `.geo` / `.msh` assets; no scripted mesh-generation pipeline was found.
- `report/ch.tex` and `report/chns.tex` are part of the deliverable. `output/ch.pdf` and `output/chns.pdf` are tracked compiled PDFs.

## Current Priorities
- Open GitHub issues are the current source of truth for project priorities: `#1-10` cover CHNS math/logic, `#11-15` cover repo/tooling/presentation work.
- For numerical CHNS work, start with canonical single-bubble or two-bubble benchmarks before many-bubble runs.
- Fine-mesh blow-up is an active unresolved problem in this repo. Do not treat coarse-mesh success as validation.
- `src/square.py` uses many random bubbles. Make centers deterministic before using it for comparisons or paper figures.

## Verification
- There is no pytest or typecheck setup. CI only runs `pylint $(git ls-files '*.py')`.
- Smallest meaningful smoke runs are from the repo root inside the Firedrake environment:
  - `./run_firedrake_container python3 src/ch.py`
  - `./run_firedrake_container python3 src/chns.py`
  - `./run_firedrake_container mpiexec -n 2 python3 src/parallel.py`
- When changing CHNS numerics, report phase mass, `phi` min/max, a divergence-related metric, nonlinear failure step, and bubble height/center of mass.

## Artifacts
- Do not assume `output/` is disposable: compiled report PDFs are tracked there, and solver scripts also write `.pvd` files into the same directory.
- `report/graphics/` contains tracked figures and animations used by the README/report.
- `src/ch.py` uses `src/presentation.mplstyle`, which enables TeX rendering. Keep plotting runs inside an environment with TeX packages installed.
- `.vscode/settings.json` sends LaTeX Workshop output to `output/`; `make clean` only removes LaTeX aux files from `output/`.
