# Phase-Field Experiments

Firedrake experiments for the Allen-Cahn (AC), Cahn-Hilliard (CH), and Cahn-Hilliard-Navier-Stokes (CHNS) equations.

The current canonical CHNS benchmark lives in `src/chns.py`, uses a Taylor-Hood velocity-pressure pair on a rectangle, and contains the rising-bubble cases used for stabilization work. Other solver files explore pressure-robust, barycentrically refined, parallel, and many-bubble variants.

<p align="center">
  <img width="45%" src="https://github.com/jk-dot/chns/blob/main/report/graphics/bublina.gif" alt="bublina">
  <img width="45%" src="https://github.com/jk-dot/chns/blob/main/report/graphics/bubliny.gif" alt="bubliny">
</p>

## Environment

Use the Firedrake helper script from the repo root:

```bash
./run_firedrake_container <command>
```

Examples in this README assume you are running from the repo root through that helper.

For MPI runs, inspect the visible CPU topology before starting a solver:

```bash
./run_firedrake_container lscpu
./run_firedrake_container mpiexec -n 4 python3 src/chns.py --benchmark single_bubble

./run_firedrake_container mpiexec --use-hwthread-cpus -n 8 python3 src/chns.py --benchmark single_bubble
```

Open MPI counts physical cores by default when assigning slots. If you need more ranks than visible physical cores but the container exposes enough hardware threads, add `--use-hwthread-cpus`. `--oversubscribe` is only useful for correctness checks, not a real parallel run.

`make build` / `make run` are currently stale and should not be treated as the supported workflow.

## Canonical Runs

Periodic Allen-Cahn benchmark:

```bash
./run_firedrake_container mpiexec -n 4 python3 src/allen_cahn.py --initial circle
```

Periodic Cahn-Hilliard benchmark:

```bash
./run_firedrake_container python3 src/ch.py
```

Single-bubble CHNS benchmark:

```bash
./run_firedrake_container python3 src/chns.py --benchmark single_bubble
```

Two-bubble CHNS benchmark:

```bash
./run_firedrake_container python3 src/chns.py --benchmark two_bubbles
```

Useful CHNS overrides:

```bash
./run_firedrake_container python3 src/chns.py --benchmark single_bubble --nx 20 --ny 60 --dt 1e-3 --steps 1000 --output-every 20
```

Useful Allen-Cahn overrides:

```bash
./run_firedrake_container mpiexec -n 4 python3 src/allen_cahn.py --initial random --steps 400 --snapshot-every 0.02
./make_allen_cahn_mp4 random
```

## Other Solver Files

- Only `src/chns.py` should be treated as the canonical CHNS solver for new numerical work.
- `src/allen_cahn.py`: periodic Allen-Cahn benchmark with circle and random initial data.
- `src/ch.py`: periodic CH reference problem.
- `src/chns.py`: canonical CHNS rising-bubble benchmark.
- `src/parallel.py`: experimental parallel CHNS variant.
- `src/square.py`: experimental many-bubble CHNS variant.
- `src/stabilization.py`: experimental CHNS stabilization variant.

Do not assume the experimental variants implement the same formulation as `src/chns.py`.

## Outputs

- `src/allen_cahn.py` writes `output/allen_cahn-<initial>.pvd` together with optional PNG/PDF snapshots.
- `src/ch.py` writes `output/ch.pvd` and CH snapshot PDFs.
- `src/chns.py` writes benchmark-specific VTK outputs such as `output/chns-single_bubble.pvd`.
- `output/ch.pdf` and `output/chns.pdf` are tracked compiled report PDFs.

## Reports

- `report/ch.tex`: CH writeup.
- `report/allen_cahn.tex`: Allen-Cahn writeup.
- `report/chns.tex`: CHNS writeup.

## Current Focus

- stabilize the canonical CHNS solver on single-bubble and two-bubble rising benchmarks
- understand why finer meshes become unstable
- only scale back up to many-bubble runs after the canonical benchmarks behave reliably
