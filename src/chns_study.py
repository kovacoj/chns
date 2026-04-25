import argparse
import json
import sys
from itertools import product
from pathlib import Path
from mpi4py import MPI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a mesh/timestep sweep for the canonical CHNS rising-bubble benchmark."
    )
    parser.add_argument("--benchmark", choices=("single_bubble", "two_bubbles"), default="single_bubble")
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=("20x60", "30x90", "40x120"),
        help="Mesh sizes as NXxNY, e.g. 20x60 30x90",
    )
    parser.add_argument("--dts", nargs="+", type=float, default=(1e-3, 5e-4, 2.5e-4))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-every", type=int, default=20)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--write-output", action="store_true")
    output_group.add_argument("--no-output", action="store_true")
    parser.add_argument(
        "--summary-json",
        default="output/chns-study-summary.json",
        help="Where to write the study summary JSON",
    )
    return parser.parse_args()


def parse_mesh(spec):
    nx_str, ny_str = spec.lower().split("x", 1)
    return int(nx_str), int(ny_str)


def summarize_history(history, benchmark, nx, ny, dt, steps):
    final = history[-1]
    return {
        "benchmark": benchmark,
        "nx": nx,
        "ny": ny,
        "dt": dt,
        "requested_steps": steps,
        "completed_steps": final["step"],
        "final_time": final["time"],
        "final_reason": final["reason"],
        "phase_mass": final["phase_mass"],
        "phi_min": final["phi_min"],
        "phi_max": final["phi_max"],
        "div_l2": final["div_l2"],
        "com_y": final["com_y"],
        "max_iterations": max(entry["iterations"] for entry in history),
        "stable": final["reason"] >= 0 and final["step"] == steps,
    }


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    is_root = comm.rank == 0
    sys.argv = [sys.argv[0]]
    from chns import CahnHilliardNavierStokes

    results = []

    for mesh_spec, dt in product(args.meshes, args.dts):
        nx, ny = parse_mesh(mesh_spec)
        if is_root:
            print(f"Running {args.benchmark} on {nx}x{ny} with dt={dt:g}")
        model = CahnHilliardNavierStokes(
            benchmark=args.benchmark,
            nx=nx,
            ny=ny,
            dt=dt,
            steps=args.steps,
            output_every=args.output_every,
            write_output=args.write_output,
        )
        history = model.run()
        summary = summarize_history(history, args.benchmark, nx, ny, dt, args.steps)
        results.append(summary)
        if is_root:
            print(
                f"  completed_steps={summary['completed_steps']} "
                f"stable={summary['stable']} "
                f"phi=[{summary['phi_min']:.3e}, {summary['phi_max']:.3e}] "
                f"div={summary['div_l2']:.3e}"
            )

    if is_root:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote study summary to {summary_path}")


if __name__ == "__main__":
    main()
