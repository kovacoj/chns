from tqdm import tqdm, TqdmWarning
import firedrake as fd
from firedrake.utility_meshes import PeriodicRectangleMesh
from firedrake.output import VTKFile
import numpy as np
from functools import cached_property
from mpi4py import MPI  # Import MPI

from queue import deque

from firedrake.pyplot import tricontourf, triplot
import matplotlib.pyplot as plt
plt.style.use('./src/presentation.mplstyle')


class CahnHilliard:
    def __init__(self):
        self.file = fd.VTKFile("output/ch.pvd")
        self.theta = 0.5 # time-evolution param

        self.sigma = 1e-1
        self.epsilon = 1e-1 # 4 * self.cell_size

        self.solver_params = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_max_it": 200,
            "snes_rtol": 1e-8,  # Tighter tolerances for better convergence
            "snes_atol": 1e-8,
            "ksp_type": "gmres",
            "ksp_max_it": 1000,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0",  # Correct split: field 0 (phi)
            "pc_fieldsplit_1_fields": "1",  # Correct split: field 1 (mu)
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "pc_factor_shift_type": "nonzero",  # Stability for LU
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",  # AMG for Schur complement
                "pc_hypre_boomeramg_max_iter": 1,
                "pc_hypre_boomeramg_strong_threshold": 0.7,
            },
        }

    def __str__(self):
        ## TODO: ptp add domain shape & size
        return f'''
            Cahn-Hilliard Model:
            · Function Space {{φ, μ}} dim(W) = {self.FunctionSpace.dim()}
            · Phase field space dim(Q) = {self.FunctionSpace[0].dim()}
            · Chemical potential space dim(M) = {self.FunctionSpace[1].dim()}
            · Physical Parameters:
                - σ = {self.sigma}, ε = {self.epsilon}
            · Mesh:
                - Cells: {self.mesh.num_cells()}
                - Vertices: {self.mesh.num_vertices()}
                - Cell Diameter: {self.cell_size:.4f}
        '''

    @property
    def comm(self):
        return self.mesh.comm

    @property
    def is_root(self):
        return self.comm.rank == 0

    @cached_property
    def cell_size(self):
        # calculate the cell diameter
        cell_diameter = fd.CellSize(self.mesh)
        h = fd.Function(fd.FunctionSpace(self.mesh, 'DG', 0))
        h.interpolate(cell_diameter)

        local_max = h.dat.data_ro.max() if h.dat.data_ro.size else 0.0
        return self.comm.allreduce(local_max, op=MPI.MAX)

    @cached_property
    def mesh(self):
        Lx, Ly = 1, 1
        nx, ny = 50, 50
        mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

        return mesh

    @cached_property
    def FunctionSpace(self):
        k = 1
        Q = M = fd.FunctionSpace(self.mesh, "CG", k)  # Phase field (φ)

        return Q * M  # Mixed function space

    @staticmethod
    def potential(x):
        return (1 - x)**2 * x**2

    @staticmethod
    def potential_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    def mobility(self, phase):
        return self.potential(phase) + 1e-3

    def mass(self, phase):
        return fd.assemble(phase * fd.dx)

    def center_of_mass(self, phase):
        x = fd.SpatialCoordinate(self.mesh)
        mass = self.mass(phase)
        if abs(mass) <= 1e-12:
            return [0.0 for _ in range(len(x))]

        return [fd.assemble(phase * x[i] * fd.dx) / mass for i in range(len(x))]

    @cached_property
    def initial_phase(self):
        x, y = fd.SpatialCoordinate(self.mesh)
        initial_phase = 0.3*fd.sin(4*fd.pi*x) * fd.sin(2*fd.pi*y) + 0.1

        return fd.Function(self.FunctionSpace[0]).interpolate(initial_phase)

    @cached_property
    def initial_chempot(self):
        # return fd.Function(self.FunctionSpace[1])
        # Time evolution seems to converge only when the chemical potential is correctly initialized
        phi = self.initial_phase
        mu = fd.Function(self.FunctionSpace[1])
        nu = fd.TestFunction(self.FunctionSpace[1])

        F = (
            fd.inner(mu, nu) - self.epsilon*self.sigma*fd.inner(fd.grad(phi), fd.grad(nu))
             - self.sigma/self.epsilon*fd.inner(self.potential_derivative(phi), nu)
        ) * fd.dx

        problem = fd.NonlinearVariationalProblem(F, mu)
        solver = fd.NonlinearVariationalSolver(problem)
        solver.solve()

        return mu

    def initialize(self, *functions):
        initial_conditions = {
            'phase': self.initial_phase,
            'chempot': self.initial_chempot
        }

        for func, name in zip(functions, initial_conditions):
            func.rename(name)
            func.interpolate(initial_conditions[name])

    def plot(self, fn, time):
        if self.comm.size > 1:
            return

        fig, axes = plt.subplots()
        levels = np.linspace(fn.dat.data.min(), fn.dat.data.max(), 200)

        contours = tricontourf(fn, levels=levels, axes=axes)
        
        axes.set_aspect("equal")
        # fig.colorbar(contours)
        plt.xticks([]); plt.yticks([])
        for c in axes.collections:
            c.set_edgecolor("face")
        # plt.gca().set_aspect('equal')
        plt.savefig(f"output/ch-t{time}.pdf", bbox_inches="tight")

    def run(self):
        w = fd.Function(self.FunctionSpace)
        w_ = fd.Function(self.FunctionSpace)

        self.initialize(*w.subfunctions)
        w_.assign(w)

        phi, mu = fd.split(w)
        phi_, mu_ = fd.split(w_)

        psi, nu = fd.TestFunctions(self.FunctionSpace)

        dt = 5e-3
        n = 1000
        total_time = n * dt

        phase = lambda phi, mu: (
            self.mobility(phi) * fd.inner(fd.grad(mu), fd.grad(psi))
        )

        F = (
            fd.inner((phi - phi_) / dt, psi) 
            + self.theta * phase(phi, mu) + (1 - self.theta) * phase(phi_, mu_)
            + fd.inner(mu, nu) - self.epsilon*self.sigma*fd.inner(fd.grad(phi), fd.grad(nu))
            - self.sigma/self.epsilon*fd.inner(self.potential_derivative(phi), nu)
        ) * fd.dx

        J = fd.derivative(F, w)
        problem = fd.NonlinearVariationalProblem(F, w, J=J)
        solver = fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_params
        )

        snapshots = deque([0, 0.5, 1., 1.5, 2, 2.5, 3, 4])
        t = 0.0

        self.file.write(*w.subfunctions, time=t)
        while snapshots and abs(snapshots[0] - t) <= 0.5 * dt:
            self.plot(w.subfunctions[0], time=snapshots.popleft())

        with tqdm(
            total=total_time,
            desc="Time Evolution",
            unit="s",
            dynamic_ncols=True,
            disable=not self.is_root,
            bar_format="{l_bar}{bar}| {n:.0e}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for step in range(n):
                solver.solve()
                w_.assign(w)

                t += dt

                self.file.write(*w.subfunctions, time=t)

                # Get solver convergence information
                snes = solver.snes
                iterations = snes.getIterationNumber()
                converged_reason = snes.getConvergedReason()

                # Check for solver convergence issues
                if converged_reason < 0 and self.is_root:
                    tqdm.write(f"Warning: Solver failed to converge at step {step}, t={t:.0e} (Reason: {converged_reason})")

                if self.is_root:
                    pbar.update(dt)
                    pbar.set_postfix_str(f"t={t:.0e}")

                while snapshots and abs(snapshots[0] - t) <= 0.5 * dt:
                    self.plot(w.subfunctions[0], time=snapshots.popleft())

        # maybe return something about convergence
        # return self


class CahnHilliardParallel:
    def __init__(self):
        self.file = fd.VTKFile("output/ch-p.pvd")
        self.theta = 0.5  # time-evolution param

        self.sigma = 1e-1
        self.epsilon = 1e-1  # 4 * self.cell_size

        self.solver_params = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_max_it": 200,
            "snes_rtol": 1e-8,  # Tighter tolerances for better convergence
            "snes_atol": 1e-8,
            "ksp_type": "gmres",
            "ksp_max_it": 1000,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0",  # Correct split: field 0 (phi)
            "pc_fieldsplit_1_fields": "1",  # Correct split: field 1 (mu)
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "pc_factor_shift_type": "nonzero",  # Stability for LU
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",  # AMG for Schur complement
                "pc_hypre_boomeramg_max_iter": 1,
                "pc_hypre_boomeramg_strong_threshold": 0.7,
            },
        }

    def __str__(self):
        return f'''
            Cahn-Hilliard Model:
            · Function Space {{φ, μ}} dim(W) = {self.FunctionSpace.dim()}
            · Phase field space dim(Q) = {self.FunctionSpace[0].dim()}
            · Chemical potential space dim(M) = {self.FunctionSpace[1].dim()}
            · Physical Parameters:
                - σ = {self.sigma}, ε = {self.epsilon}
            · Mesh:
                - Cells: {self.mesh.num_cells()}
                - Vertices: {self.mesh.num_vertices()}
                - Cell Diameter: {self.cell_size:.4f}
        '''

    @cached_property
    def cell_size(self):
        cell_diameter = fd.CellSize(self.mesh)
        h = fd.Function(fd.FunctionSpace(self.mesh, 'DG', 0))
        h.interpolate(cell_diameter)
        return h.dat.data_ro.max()

    @cached_property
    def mesh(self):
        Lx, Ly = 1, 1
        nx, ny = 50, 50
        mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        return mesh

    @cached_property
    def FunctionSpace(self):
        k = 1
        Q = M = fd.FunctionSpace(self.mesh, "CG", k)  # Phase field (φ)
        return Q * M  # Mixed function space

    @staticmethod
    def potential(x):
        return (1 - x)**2 * x**2

    @staticmethod
    def potential_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    def mobility(self, phase):
        return self.potential(phase) + 1e-3

    def mass(self, phase):
        return fd.assemble(phase * fd.dx)

    def center_of_mass(self, phase):
        x = fd.SpatialCoordinate(self.mesh)
        mass = self.mass(phase)
        if abs(mass) <= 1e-12:
            return [0.0 for _ in range(len(x))]

        return [fd.assemble(phase * x[i] * fd.dx) / mass for i in range(len(x))]

    @cached_property
    def initial_phase(self):
        x, y = fd.SpatialCoordinate(self.mesh)
        initial_phase = 0.3 * fd.sin(4 * fd.pi * x) * fd.sin(2 * fd.pi * y) + 0.1
        return fd.Function(self.FunctionSpace[0]).interpolate(initial_phase)

    @cached_property
    def initial_chempot(self):
        phi = self.initial_phase
        mu = fd.Function(self.FunctionSpace[1])
        nu = fd.TestFunction(self.FunctionSpace[1])

        F = (
            fd.inner(mu, nu) - self.epsilon * self.sigma * fd.inner(fd.grad(phi), fd.grad(nu))
            - self.sigma / self.epsilon * fd.inner(self.potential_derivative(phi), nu)
        ) * fd.dx

        problem = fd.NonlinearVariationalProblem(F, mu)
        solver = fd.NonlinearVariationalSolver(problem)
        solver.solve()

        return mu

    def initialize(self, *functions):
        initial_conditions = {
            'phase': self.initial_phase,
            'chempot': self.initial_chempot
        }

        for func, name in zip(functions, initial_conditions):
            func.rename(name)
            func.interpolate(initial_conditions[name])

    def run(self):
        w = fd.Function(self.FunctionSpace)
        w_ = fd.Function(self.FunctionSpace)

        self.initialize(*w.subfunctions)
        w_.assign(w)

        phi, mu = fd.split(w)
        phi_, mu_ = fd.split(w_)

        psi, nu = fd.TestFunctions(self.FunctionSpace)

        dt = 5e-3
        n = 2000
        total_time = n * dt

        phase = lambda phi, mu: (
            self.mobility(phi) * fd.inner(fd.grad(mu), fd.grad(psi))
        )

        F = (
            fd.inner((phi - phi_) / dt, psi)
            + self.theta * phase(phi, mu) + (1 - self.theta) * phase(phi_, mu_)
            + fd.inner(mu, nu) - self.epsilon * self.sigma * fd.inner(fd.grad(phi), fd.grad(nu))
            - self.sigma / self.epsilon * fd.inner(self.potential_derivative(phi), nu)
        ) * fd.dx

        J = fd.derivative(F, w)
        problem = fd.NonlinearVariationalProblem(F, w, J=J)
        solver = fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_params
        )

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            pbar = tqdm(total=total_time, desc="Time Evolution", unit="s", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n:.0e}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        else:
            pbar = None

        t = 0.0

        for step in range(n):
            solver.solve()
            w_.assign(w)

            t += dt

            self.file.write(*w.subfunctions, time=t)

            snes = solver.snes
            iterations = snes.getIterationNumber()
            converged_reason = snes.getConvergedReason()

            if converged_reason < 0 and rank == 0:
                tqdm.write(f"Warning: Solver failed to converge at step {step}, t={t:.0e} (Reason: {converged_reason})")

            if rank == 0:
                pbar.update(dt)
                pbar.set_postfix_str(f"t={t:.0e}")

        if rank == 0:
            pbar.close()



if __name__ == '__main__':
    model = CahnHilliard()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(model)

    model.run()
