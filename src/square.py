from tqdm import tqdm, TqdmWarning

# import warnings
# warnings.filterwarnings("ignore",  TqdmWarning)

import firedrake as fd
from firedrake.utility_meshes import RectangleMesh
from firedrake.output import VTKFile
import numpy as np

from utils import refine_bary
from functools import cached_property

# Experimental many-bubble variant; the canonical capillary coupling lives in
# `src/chns.py` and this file may intentionally diverge while ideas are tested.

class CahnHilliardNavierStokes:
    def __init__(self):
        self.file = fd.VTKFile("output/bubbles.pvd")
        self.theta = 1 # time-evolution param

        self.rho1, self.rho2 = 10, 1
        self.nu1 = self.nu2 = 1 # paper says should be the same
        self.sigma = 1e-1
        self.gravity = fd.Constant((0., -9.81))
        self.epsilon = 1e-1 # 4 * self.cell_size
        self.m0 = 1e-4 # mobility factor, needs to compensate the gradient of mu :-/

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
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "pc_factor_shift_type": "nonzero",
                "mat_mumps_icntl_14": 500
            },
        }

    def __str__(self):
        ## TODO: ptp add domain shape & size
        return f'''
            Cahn-Hilliard Navier-Stokes Model:
            · Function Space {{u, p, φ, μ}} dim(W) = {self.FunctionSpace.dim()}
            · Velocity space dim(V) = {self.FunctionSpace[0].dim()}
            · Pressure space dim(P) = {self.FunctionSpace[1].dim()}
            · Phase field space dim(Q) = {self.FunctionSpace[2].dim()}
            · Chemical potential space dim(M) = {self.FunctionSpace[3].dim()}
            · Physical Parameters:
                - Density: ρ₁ = {self.rho1}, ρ₂ = {self.rho2}
                - Viscosity: ν₁ = {self.nu1}, ν₂ = {self.nu2}
                - σ = {self.sigma}, ε = {self.epsilon}
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
        Lx, Ly = 4.0, 8.0
        nx, ny = 100, 200
        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

        # barycentric refinement using Alfeld split
        mesh = refine_bary(mesh)

        return mesh

    @cached_property
    def FunctionSpace(self):
        # Scott-Vogelius pressure-robust
        k = 2
        V = fd.VectorFunctionSpace(self.mesh, "CG", k)  # Velocity
        P = fd.FunctionSpace(self.mesh, "DG", k-1)      # Pressure
        Q = M = fd.FunctionSpace(self.mesh, "CG", k)  # Phase field (φ)

        return V * P * Q * M  # Mixed function space
        # return fd.MixedFunctionSpace([V, P, Q, M])

    @staticmethod
    def potential(x):
        return (1 - x)**2 * x**2

    @staticmethod
    def potential_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    def density(self, phase):
        # return self.rho1 + (self.rho2 - self.rho1) * phase
        return fd.conditional(
            phase < 0,
            self.rho1,
            fd.conditional(
                phase > 1, 
                self.rho2, 
                self.rho1 + phase*(self.rho2 - self.rho1))
        )

    def mobility(self, phase):
        return self.m0 * self.potential(phase)
        # return self.m0 * (1 - phase)**2 * phase**2

    def viscosity(self, phase):
        return self.nu2 * phase + self.nu1 * (1.0 - phase)
   
    def mass(self, phase):
        return fd.assemble(self.density(phase) * fd.dx)

    def center_of_mass(self, phase):
        x = fd.SpatialCoordinate(self.mesh)        

        return [
            fd.assemble(self.density(phase) * x[i] * fd.dx) / self.mass(phase)
            for i in range(len(x))
        ]

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

        mesh_coords = self.mesh.coordinates.dat.data_ro
        domain_size = np.ptp(mesh_coords, axis=0)

        radius = 0.1
        n_bubbles = 42
        centers = np.random.rand(n_bubbles, len(domain_size)) * domain_size

        # centers = np.array([[0.4, 0.3], [0.6, 0.8]])
        # radius = 0.2

        initial_phase = fd.Constant(0.)

        for center in centers:
            diff = [(coordinates[i] - center[i])**2 for i in range(len(coordinates))]
            distance = fd.sqrt(sum(diff))

            initial_phase = fd.max_value(
                initial_phase,
                fd.conditional(distance <= radius, 1. ,0.)
            )

        return fd.Function(self.FunctionSpace[2]).interpolate(initial_phase)

    @cached_property
    def initial_chempot(self):
        return fd.Function(self.FunctionSpace[3])
        # Time evolution seems to converge only when the chemical potential is correctly initialized
        # phi = self.initial_phase
        # mu = fd.Function(self.FunctionSpace[3])
        # nu = fd.TestFunction(self.FunctionSpace[3])

        # F = (
        #     fd.inner(mu, nu) - self.epsilon*self.sigma*fd.inner(fd.grad(phi), fd.grad(nu))
        #      - self.sigma/self.epsilon*fd.inner(self.potential_derivative(phi), nu)
        # ) * fd.dx

        # problem = fd.NonlinearVariationalProblem(F, mu)
        # solver = fd.NonlinearVariationalSolver(problem)
        # solver.solve()

        # return mu

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

        # dt = min(
        #     0.25 * self.cell_size,  # CFL condition
        #     0.5 * self.cell_size**2 / max(self.nu1, self.nu2),  # Diffusion stability
        #     0.5 * self.epsilon**2 / self.m0  # Interface dynamics
        # )

        dt = 1e-2
        n = 10**3
        total_time = n * dt

        momentum = lambda u, p, phi, mu: (
            self.density(phi)*fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
            + self.viscosity(phi)*fd.inner(fd.grad(u), fd.grad(v))
            - self.density(phi)*fd.dot(self.gravity, v)
            - p*fd.div(v) - phi*mu*fd.div(v)
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
        ) * fd.dx

        # possibly not needed since using pressure-robust method
        # R space does not even work with this setting, so....
        # F += (p * s + r * q) * fd.dx # pressure stabilization

        bcs = [
            fd.DirichletBC(self.FunctionSpace.sub(0), fd.Constant((0, 0)), 'on_boundary')
        ]

        J = fd.derivative(F, w)
        problem = fd.NonlinearVariationalProblem(F, w, bcs=bcs, J=J)
        solver = fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_params
        )

        with tqdm(
            total=total_time, desc="Time Evolution", unit="s", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n:.0e}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            t = 0.0
            
            for step in range(n):
                solver.solve()
                w_.assign(w)

                self.file.write(*w.subfunctions, time=t)

                # Get solver convergence information
                snes = solver.snes
                iterations = snes.getIterationNumber()
                converged_reason = snes.getConvergedReason()

                # Check for solver convergence issues
                if converged_reason < 0:
                    tqdm.write(f"Warning: Solver failed to converge at step {step}, t={t:.0e} (Reason: {converged_reason})")

                t += dt

                pbar.update(dt)
                pbar.set_postfix_str(f"t={t:.0e}")

        # maybe return something about convergence
        # return self

if __name__ == '__main__':
    model = CahnHilliardNavierStokes()

    print(model)

    model.run()
