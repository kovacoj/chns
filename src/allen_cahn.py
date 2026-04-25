import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


def _parse_main_args(argv):
    parser = argparse.ArgumentParser(description="Run a periodic Allen-Cahn benchmark in Firedrake.")
    parser.add_argument("--initial", choices=("circle", "random"), default="circle")
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output-every", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=2e-2)
    parser.add_argument("--mobility", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument(
        "--snapshot-times",
        nargs="*",
        type=float,
        default=(0.0, 0.02, 0.04, 0.06, 0.08, 0.1),
    )
    parser.add_argument("--snapshot-every", type=float, default=None)
    parser.add_argument("--snapshot-vmin", type=float, default=-1.0)
    parser.add_argument("--snapshot-vmax", type=float, default=1.0)
    parser.add_argument(
        "--snapshot-size",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(4.0, 4.0),
    )
    parser.add_argument("--snapshot-dpi", type=int, default=200)
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    CLI_ARGS, REMAINING_ARGV = _parse_main_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *REMAINING_ARGV]
else:
    CLI_ARGS = None

from tqdm import tqdm
import firedrake as fd
from firedrake.utility_meshes import PeriodicRectangleMesh
from firedrake.output import VTKFile
from functools import cached_property
from mpi4py import MPI

plt.style.use("./src/presentation.mplstyle")


class AllenCahn:
    def __init__(
        self,
        initial="circle",
        nx=80,
        ny=80,
        dt=2.5e-4,
        steps=400,
        output_every=20,
        epsilon=2e-2,
        mobility=1.0,
        seed=0,
        write_output=True,
        snapshot_times=(),
        snapshot_vmin=-1.0,
        snapshot_vmax=1.0,
        snapshot_size=(4.0, 4.0),
        snapshot_dpi=200,
    ):
        self.initial = initial
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.n_steps = steps
        self.output_every = output_every
        self.epsilon = epsilon
        self.mobility_constant = mobility
        self.seed = seed
        self.write_output = write_output
        self.snapshot_times = tuple(snapshot_times)
        self.snapshot_vmin = snapshot_vmin
        self.snapshot_vmax = snapshot_vmax
        self.snapshot_size = tuple(snapshot_size)
        self.snapshot_dpi = snapshot_dpi

        if self.snapshot_vmax <= self.snapshot_vmin:
            raise ValueError("snapshot_vmax must be greater than snapshot_vmin")
        if len(self.snapshot_size) != 2 or any(length <= 0 for length in self.snapshot_size):
            raise ValueError("snapshot_size must contain two positive values")
        if self.snapshot_dpi <= 0:
            raise ValueError("snapshot_dpi must be positive")

        self.file = VTKFile(f"output/allen_cahn-{initial}.pvd") if write_output else None
        self.solver_params = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_max_it": 50,
            "snes_rtol": 1e-8,
            "snes_atol": 1e-8,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    def __str__(self):
        return self.describe()

    def describe(self):
        mesh_lines = [
            f"                - Cells: {self.global_num_cells}",
            f"                - Cell Diameter: {self.cell_size:.4f}",
        ]
        if self.comm.size == 1:
            mesh_lines.insert(1, f"                - Vertices: {self.mesh.num_vertices()}")
        else:
            mesh_lines.insert(1, f"                - MPI ranks: {self.comm.size}")
        mesh_summary = "\n".join(mesh_lines)

        return f"""
            Allen-Cahn Model:
            · Initial condition = {self.initial}
            · Function Space {{φ}} dim(Q) = {self.FunctionSpace.dim()}
            · Physical Parameters:
                - ε = {self.epsilon}, M = {self.mobility_constant}
                - dt = {self.dt}, steps = {self.n_steps}
                - seed = {self.seed}
                - write_output = {self.write_output}
                - snapshot_times = {self.snapshot_times}
                - snapshot_range = ({self.snapshot_vmin}, {self.snapshot_vmax})
                - snapshot_size = {self.snapshot_size}
                - snapshot_dpi = {self.snapshot_dpi}
            · Mesh:
{mesh_summary}
        """

    @property
    def comm(self):
        return self.mesh.comm

    @property
    def is_root(self):
        return self.comm.rank == 0

    @cached_property
    def global_num_cells(self):
        return self.comm.allreduce(self.mesh.num_cells(), op=MPI.SUM)

    @cached_property
    def cell_size(self):
        cell_diameter = fd.CellSize(self.mesh)
        h = fd.Function(fd.FunctionSpace(self.mesh, "DG", 0))
        h.interpolate(cell_diameter)
        local_max = h.dat.data_ro.max() if h.dat.data_ro.size else 0.0
        return self.comm.allreduce(local_max, op=MPI.MAX)

    @cached_property
    def mesh(self):
        return PeriodicRectangleMesh(self.nx, self.ny, 1.0, 1.0, quadrilateral=False)

    @cached_property
    def FunctionSpace(self):
        return fd.FunctionSpace(self.mesh, "CG", 1)

    @cached_property
    def mesh_coordinate_values(self):
        return np.array(self.mesh.coordinates.dat.data_ro, copy=True)

    @staticmethod
    def potential(phase):
        return 0.25 * (phase**2 - 1.0) ** 2

    @staticmethod
    def potential_derivative(phase):
        return phase**3 - phase

    def phase_bounds(self, phase):
        values = phase.dat.data_ro
        local_min = values.min() if values.size else float("inf")
        local_max = values.max() if values.size else -float("inf")
        return (
            self.comm.allreduce(local_min, op=MPI.MIN),
            self.comm.allreduce(local_max, op=MPI.MAX),
        )

    def energy(self, phase):
        return fd.assemble(
            (
                self.potential(phase) / self.epsilon**2
                + 0.5 * fd.inner(fd.grad(phase), fd.grad(phase))
            )
            * fd.dx
        )

    def collect_diagnostics(self, phase):
        phase_min, phase_max = self.phase_bounds(phase)
        return {
            "phi_min": phase_min,
            "phi_max": phase_max,
            "energy": self.energy(phase),
        }

    def _circle_initial_phase(self):
        x, y = fd.SpatialCoordinate(self.mesh)
        radius = 0.2
        center_x = center_y = 0.5
        return fd.conditional((x - center_x) ** 2 + (y - center_y) ** 2 < radius**2, 1.0, -1.0)

    def _random_initial_phase(self):
        phase = fd.Function(self.FunctionSpace)
        coordinates = self.mesh_coordinate_values
        hashed = np.sin(
            12.9898 * coordinates[:, 0] + 78.233 * coordinates[:, 1] + 37.719 * self.seed
        ) * 43758.5453123
        phase.dat.data[:] = 2.0 * (hashed - np.floor(hashed)) - 1.0
        return phase

    @cached_property
    def initial_phase(self):
        if self.initial == "circle":
            return fd.Function(self.FunctionSpace).interpolate(self._circle_initial_phase())
        if self.initial == "random":
            return self._random_initial_phase()
        raise ValueError(f"Unsupported initial condition: {self.initial}")

    def plot_phase_snapshot(self, phase, time):
        local_points = self.mesh_coordinate_values
        local_values = np.array(phase.dat.data_ro, copy=True)
        local_count = min(len(local_points), len(local_values))

        gathered_points = self.comm.gather(local_points[:local_count], root=0)
        gathered_values = self.comm.gather(local_values[:local_count], root=0)

        if not self.is_root:
            return

        points = np.concatenate(gathered_points, axis=0)
        values = np.concatenate(gathered_values, axis=0)

        rounded = np.round(points, decimals=12)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        unique_idx.sort()
        points = points[unique_idx]
        values = values[unique_idx]

        x_coords = np.unique(points[:, 0])
        y_coords = np.unique(points[:, 1])
        grid = np.full((len(x_coords), len(y_coords)), np.nan)

        x_index = {value: i for i, value in enumerate(np.round(x_coords, decimals=12))}
        y_index = {value: i for i, value in enumerate(np.round(y_coords, decimals=12))}

        for point, value in zip(np.round(points, decimals=12), values):
            grid[x_index[point[0]], y_index[point[1]]] = value

        fig, axes = plt.subplots(figsize=self.snapshot_size)
        axes.imshow(
            grid.T,
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            vmin=self.snapshot_vmin,
            vmax=self.snapshot_vmax,
            interpolation="bicubic",
            cmap="RdYlBu",
        )

        axes.set_aspect("equal")
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_xticks([])
        axes.set_yticks([])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        for collection in axes.collections:
            collection.set_edgecolor("face")

        base_path = f"output/allen_cahn-{self.initial}-t{time:g}"
        fig.savefig(f"{base_path}.png", dpi=self.snapshot_dpi)
        fig.savefig(f"{base_path}.pdf", dpi=self.snapshot_dpi)
        plt.close(fig)

    def run(self):
        phase = fd.Function(self.FunctionSpace)
        phase_old = fd.Function(self.FunctionSpace)

        phase.rename("phase")
        phase_old.assign(self.initial_phase)
        phase.assign(phase_old)

        psi = fd.TestFunction(self.FunctionSpace)
        dt = self.dt
        total_time = self.n_steps * dt

        F = (
            fd.inner((phase - phase_old) / dt, psi)
            + self.mobility_constant * fd.inner(fd.grad(phase), fd.grad(psi))
            + self.mobility_constant / self.epsilon**2 * fd.inner(self.potential_derivative(phase), psi)
        ) * fd.dx

        J = fd.derivative(F, phase)
        problem = fd.NonlinearVariationalProblem(F, phase, J=J)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=self.solver_params)

        history = []
        pending_snapshots = list(self.snapshot_times)
        initial_diagnostics = self.collect_diagnostics(phase)

        if self.file is not None:
            self.file.write(phase, time=0.0)
        while pending_snapshots and pending_snapshots[0] <= 0.0:
            self.plot_phase_snapshot(phase, pending_snapshots.pop(0))
        history.append({"step": 0, "time": 0.0, "iterations": 0, "reason": 0, **initial_diagnostics})

        with tqdm(
            total=total_time,
            desc="Time Evolution",
            unit="s",
            dynamic_ncols=True,
            disable=not self.is_root,
            bar_format="{l_bar}{bar}| {n:.0e}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for step in range(1, self.n_steps + 1):
                solver.solve()
                phase_old.assign(phase)

                t = step * dt
                diagnostics = self.collect_diagnostics(phase)

                if self.file is not None and step % self.output_every == 0:
                    self.file.write(phase, time=t)

                while pending_snapshots and abs(t - pending_snapshots[0]) <= 0.5 * dt:
                    self.plot_phase_snapshot(phase, pending_snapshots.pop(0))

                snes = solver.snes
                iterations = snes.getIterationNumber()
                converged_reason = snes.getConvergedReason()

                history.append(
                    {
                        "step": step,
                        "time": t,
                        "iterations": iterations,
                        "reason": converged_reason,
                        **diagnostics,
                    }
                )

                if self.is_root and (step % self.output_every == 0 or converged_reason < 0):
                    tqdm.write(
                        f"step={step:04d} t={t:.3e} its={iterations:02d} "
                        f"phi=[{diagnostics['phi_min']:.3e}, {diagnostics['phi_max']:.3e}] "
                        f"energy={diagnostics['energy']:.6f} reason={converged_reason}"
                    )

                if converged_reason < 0:
                    break

                if self.is_root:
                    pbar.update(t - pbar.n)
                    pbar.set_postfix_str(
                        f"t={t:.2e} phi=[{diagnostics['phi_min']:.2e}, {diagnostics['phi_max']:.2e}]"
                    )

        return history


def build_snapshot_times(total_time, explicit_times=(), every=None):
    snapshot_times = list(explicit_times)
    if every is not None and every > 0:
        count = int(total_time / every)
        snapshot_times.extend(every * step for step in range(1, count + 1))

    unique_times = []
    for time in sorted(snapshot_times):
        if 0.0 <= time <= total_time and (not unique_times or abs(time - unique_times[-1]) > 1e-12):
            unique_times.append(time)
    return tuple(unique_times)


if __name__ == "__main__":
    total_time = CLI_ARGS.dt * CLI_ARGS.steps
    model = AllenCahn(
        initial=CLI_ARGS.initial,
        nx=CLI_ARGS.nx,
        ny=CLI_ARGS.ny,
        dt=CLI_ARGS.dt,
        steps=CLI_ARGS.steps,
        output_every=CLI_ARGS.output_every,
        epsilon=CLI_ARGS.epsilon,
        mobility=CLI_ARGS.mobility,
        seed=CLI_ARGS.seed,
        write_output=not CLI_ARGS.no_output,
        snapshot_times=build_snapshot_times(total_time, CLI_ARGS.snapshot_times, CLI_ARGS.snapshot_every),
        snapshot_vmin=CLI_ARGS.snapshot_vmin,
        snapshot_vmax=CLI_ARGS.snapshot_vmax,
        snapshot_size=CLI_ARGS.snapshot_size,
        snapshot_dpi=CLI_ARGS.snapshot_dpi,
    )

    summary = model.describe()
    if model.is_root:
        print(summary)

    model.run()
