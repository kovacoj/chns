import argparse
import sys

CLI_PARSER = argparse.ArgumentParser(description="Run the canonical CHNS rising-bubble benchmark.")
CLI_PARSER.add_argument("--benchmark", choices=("single_bubble", "two_bubbles"), default="single_bubble")
CLI_PARSER.add_argument("--nx", type=int, default=20)
CLI_PARSER.add_argument("--ny", type=int, default=60)
CLI_PARSER.add_argument("--dt", type=float, default=1e-3)
CLI_PARSER.add_argument("--steps", type=int, default=1000)
CLI_PARSER.add_argument("--output-every", type=int, default=20)
CLI_ARGS, REMAINING_ARGV = CLI_PARSER.parse_known_args()
sys.argv = [sys.argv[0], *REMAINING_ARGV]

from tqdm import tqdm
import firedrake as fd
from firedrake.utility_meshes import RectangleMesh
from firedrake.output import VTKFile
from functools import cached_property


class CahnHilliardNavierStokes:
    def __init__(self, benchmark="single_bubble", nx=20, ny=60, dt=1e-3, steps=1000, output_every=20):
        self.benchmark = benchmark
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.n_steps = steps
        self.output_every = output_every

        self.file = fd.VTKFile(f"output/chns-{benchmark}.pvd")
        self.theta = 1.0

        self.rho1, self.rho2 = 10, 1
        self.nu1 = self.nu2 = 1 # paper says should be the same
        self.sigma = 1e-1
        self.gravity = fd.Constant((0., -9.81))
        self.epsilon = 5e-2
        self.m0 = 1e-4

        self.solver_params = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_max_it": 200,
            # half digit precission :-/
            "snes_rtol": 1e-4,
            "snes_atol": 1e-4,
            "ksp_type": "gmres",
            # "snes_monitor": None,
            # "ksp_monitor": None,
            "ksp_max_it": 1000,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0,1",
            "pc_fieldsplit_1_fields": "2,3",
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "pc_factor_shift_type": "nonzero",
                "mat_mumps_icntl_14": 500
            },
            "fieldsplit_1": {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "ksp_rtol": 1e-2,  # Looser tolerance for ill-conditioned subsystem
                "mat_mumps_icntl_14": 500
            },
        }


    def __str__(self):
        return f'''
            Cahn-Hilliard Navier-Stokes Model:
            · Benchmark = {self.benchmark}
            · Function Space {{u, p, φ, μ}} dim(W) = {self.FunctionSpace.dim()}
            · Velocity space dim(V) = {self.FunctionSpace[0].dim()}
            · Pressure space dim(P) = {self.FunctionSpace[1].dim()}
            · Phase field space dim(Q) = {self.FunctionSpace[2].dim()}
            · Chemical potential space dim(M) = {self.FunctionSpace[3].dim()}
            · Physical Parameters:
                - Density: ρ₁ = {self.rho1}, ρ₂ = {self.rho2}
                - Viscosity: ν₁ = {self.nu1}, ν₂ = {self.nu2}
                - Mobility: m0 = {self.m0}
                - σ = {self.sigma}, ε = {self.epsilon}
                - dt = {self.dt}, steps = {self.n_steps}
            · Mesh:
                - Cells: {self.mesh.num_cells()}
                - Vertices: {self.mesh.num_vertices()}
                - Cell Diameter: {self.cell_size:.4f}
        '''

    @cached_property
    def cell_size(self):
        # calculate the cell diameter
        cell_diameter = fd.CellSize(self.mesh)
        h = fd.Function(fd.FunctionSpace(self.mesh, 'DG', 0))
        h.interpolate(cell_diameter)
        
        return h.dat.data_ro.max()

    @cached_property
    def mesh(self):
        return RectangleMesh(self.nx, self.ny, 1.0, 3.0, quadrilateral=False)

    @cached_property
    def FunctionSpace(self):
        # Taylor-Hood velocity-pressure pair with scalar CG1 phase fields.
        k = 2
        V = fd.VectorFunctionSpace(self.mesh, "CG", k)  # Velocity
        P = fd.FunctionSpace(self.mesh, "CG", k-1)      # Pressure
        Q = M = fd.FunctionSpace(self.mesh, "CG", k-1)  # Phase field (φ)

        return V * P * Q * M  # Mixed function space

    @cached_property
    def pressure_nullspace(self):
        return fd.MixedVectorSpaceBasis(
            self.FunctionSpace,
            [
                self.FunctionSpace.sub(0),
                fd.VectorSpaceBasis(constant=True, comm=self.mesh.comm),
                self.FunctionSpace.sub(2),
                self.FunctionSpace.sub(3),
            ],
        )

    @staticmethod
    def potential(x):
        return (1 - x)**2 * x**2

    @staticmethod
    def potential_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    def density(self, phase):
        return fd.conditional(
            phase < 0,
            self.rho1,
            fd.conditional(
                phase > 1, 
                self.rho2, 
                self.rho1 + phase*(self.rho2 - self.rho1))
        )

    def mobility(self, phase):
        return self.m0 * self.potential(phase) + 1e-6

    def viscosity(self, phase):
        return self.nu2 * phase + self.nu1 * (1.0 - phase)
   
    def phase_mass(self, phase):
        return fd.assemble(phase * fd.dx)

    def phase_center_of_mass(self, phase):
        x = fd.SpatialCoordinate(self.mesh)
        mass = self.phase_mass(phase)
        if abs(mass) <= 1e-12:
            return (0.0, 0.0)

        return tuple(fd.assemble(phase * x[i] * fd.dx) / mass for i in range(len(x)))

    def phase_bounds(self, phase):
        values = phase.dat.data_ro
        return values.min(), values.max()

    def divergence_metric(self, velocity):
        return fd.assemble(fd.div(velocity) * fd.div(velocity) * fd.dx) ** 0.5

    def collect_diagnostics(self, velocity, phase):
        phi_min, phi_max = self.phase_bounds(phase)
        _, center_y = self.phase_center_of_mass(phase)
        return {
            "phase_mass": self.phase_mass(phase),
            "phi_min": phi_min,
            "phi_max": phi_max,
            "div_l2": self.divergence_metric(velocity),
            "com_y": center_y,
        }

    def energy(self, w):
        u, p, phi, mu = w.split()

        kinetic = 0.5*fd.assemble(self.density(phi)*fd.inner(u, u)*fd.dx)
        interface = self.sigma * fd.assemble(
            self.epsilon * fd.inner(fd.grad(phi), fd.grad(phi)) + self.potential(phi) / self.epsilon * fd.dx
        )

        return kinetic, interface

    @cached_property
    def initial_velocity(self):
        # generalize using mesh dim for vector
        return fd.Function(self.FunctionSpace[0]).interpolate(fd.Constant((0., 0.)))

    @cached_property
    def initial_pressure(self):
        # should initialize to zero ?
        return fd.Function(self.FunctionSpace[1])

    @cached_property
    def initial_phase(self):
        coordinates = fd.SpatialCoordinate(self.mesh)

        if self.benchmark == "single_bubble":
            radius = 0.16
            centers = ((0.5, 0.8),)
        elif self.benchmark == "two_bubbles":
            radius = 0.14
            centers = ((0.5, 0.65), (0.5, 1.05))
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")

        interface_width = 0.5 * self.epsilon
        initial_phase = 0.0

        # Use a smooth diffuse interface so refinement does not turn the
        # benchmark into a sharper and stiffer problem by construction.
        for center_x, center_y in centers:
            distance = fd.sqrt((coordinates[0] - center_x)**2 + (coordinates[1] - center_y)**2 + 1e-12)
            bubble = 0.5 * (1.0 - fd.tanh((distance - radius) / interface_width))
            initial_phase = initial_phase + bubble

        return fd.Function(self.FunctionSpace[2]).interpolate(initial_phase)

    @cached_property
    def initial_chempot(self):
        # return fd.Function(self.FunctionSpace[3])
        # Time evolution seems to converge only when the chemical potential is correctly initialized
        phi = self.initial_phase
        mu = fd.Function(self.FunctionSpace[3])
        nu = fd.TestFunction(self.FunctionSpace[3])

        F = (
            fd.inner(mu, nu) - self.epsilon*self.sigma*fd.inner(fd.grad(phi), fd.grad(nu))
             - self.sigma/self.epsilon*fd.inner(self.potential_derivative(phi), nu)
        ) * fd.dx

        problem = fd.NonlinearVariationalProblem(F, mu)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters={
                                                            "snes_type": "newtonls",
                                                            "snes_rtol": 1e-6,
                                                            "snes_atol": 1e-6,
                                                            "ksp_type": "preonly",
                                                            "pc_type": "lu"
                                                        })
        solver.solve()

        return mu

    def initialize(self, *functions):
        initial_conditions = {
            'velocity': self.initial_velocity,
            'pressure': self.initial_pressure,
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

        u, p, phi, mu = fd.split(w)
        u_, p_, phi_, mu_ = fd.split(w_)

        v, q, psi, nu = fd.TestFunctions(self.FunctionSpace)

        dt = self.dt
        n = self.n_steps
        total_time = n * dt

        momentum = lambda u, p, phi, mu: (
            self.density(phi)*fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
            + self.viscosity(phi)*fd.inner(fd.grad(u), fd.grad(v))
            - self.density(phi)*fd.dot(self.gravity, v)
            - phi*fd.inner(fd.grad(mu), v) - p*fd.div(v)
        )

        phase = lambda u, p, phi, mu: (
            fd.inner(fd.dot(u, fd.grad(phi)), psi)
            + self.mobility(phi) * fd.inner(fd.grad(mu), fd.grad(psi))
        )

        F = (
            fd.inner((phi - phi_) / dt, psi) 
            + (self.theta*self.density(phi) + (1 - self.theta) * self.density(phi_)) * fd.inner((u - u_) / dt, v)
            + self.theta * momentum(u, p, phi, mu) + self.theta * phase(u, p, phi, mu)
            + (1 - self.theta) * momentum(u_, p_, phi_, mu_) + (1 - self.theta) * phase(u_, p_, phi_, mu_)
            + q * fd.div(u) + fd.inner(mu, nu) - self.epsilon*self.sigma*fd.inner(fd.grad(phi), fd.grad(nu)) - self.sigma/self.epsilon*fd.inner(self.potential_derivative(phi), nu)
        ) * fd.dx() # fd.dx(degree=6)

        # possibly not needed since using pressure-robust method
        # R space does not even work with this setting, so....
        # F += (p * s + r * q) * fd.dx # pressure stabilization
        # F += 0.1 * fd.div(u) * fd.div(v) * fd.dx # stabilization ?

        bcs = [
            fd.DirichletBC(self.FunctionSpace.sub(0), fd.Constant((0, 0)), "on_boundary"),
        ]

        J = fd.derivative(F, w)
        problem = fd.NonlinearVariationalProblem(F, w, J=J, bcs=bcs)
        solver = fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_params,
            nullspace=self.pressure_nullspace,
            transpose_nullspace=self.pressure_nullspace,
        )

        history = []
        velocity_fn, _, phase_fn, _ = w.subfunctions
        initial_diagnostics = self.collect_diagnostics(velocity_fn, phase_fn)
        self.file.write(*w.subfunctions, time=0.0)
        history.append({"step": 0, "time": 0.0, "iterations": 0, "reason": 0, **initial_diagnostics})

        with tqdm(
            total=total_time, desc="Time Evolution", unit="s", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n:.0e}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            t = 0.0

            for step in range(1, n + 1):
                solver.solve()
                w_.assign(w)

                t += dt

                velocity_fn, _, phase_fn, _ = w.subfunctions
                diagnostics = self.collect_diagnostics(velocity_fn, phase_fn)

                if step % self.output_every == 0:
                    self.file.write(*w.subfunctions, time=t)

                snes = solver.snes
                iterations = snes.getIterationNumber()
                converged_reason = snes.getConvergedReason()

                history.append({
                    "step": step,
                    "time": t,
                    "iterations": iterations,
                    "reason": converged_reason,
                    **diagnostics,
                })

                if step % self.output_every == 0 or converged_reason < 0:
                    tqdm.write(
                        f"step={step:04d} t={t:.3e} its={iterations:02d} "
                        f"phi=[{diagnostics['phi_min']:.3e}, {diagnostics['phi_max']:.3e}] "
                        f"mass={diagnostics['phase_mass']:.6f} div={diagnostics['div_l2']:.3e} "
                        f"com_y={diagnostics['com_y']:.3e} reason={converged_reason}"
                    )

                if converged_reason < 0:
                    break

                pbar.update(dt)
                pbar.set_postfix_str(
                    f"t={t:.2e} phi=[{diagnostics['phi_min']:.2e}, {diagnostics['phi_max']:.2e}] com_y={diagnostics['com_y']:.2e}"
                )

        return history


if __name__ == '__main__':
    model = CahnHilliardNavierStokes(
        benchmark=CLI_ARGS.benchmark,
        nx=CLI_ARGS.nx,
        ny=CLI_ARGS.ny,
        dt=CLI_ARGS.dt,
        steps=CLI_ARGS.steps,
        output_every=CLI_ARGS.output_every,
    )

    print(model)

    model.run()
