from tqdm import tqdm
import firedrake as fd
from firedrake import COMM_WORLD
from firedrake.utility_meshes import RectangleMesh
from firedrake.output import VTKFile
import numpy as np

from utils import refine_bary
from functools import cached_property

class CahnHilliardNavierStokes:
    def __init__(self):
        self.file = VTKFile("output/parallel.pvd", "w")
        self.theta = 1.0  # Backward Euler while the CHNS formulation is still being stabilized.

        # Physical parameters
        self.rho1, self.rho2 = 1000.0, 1100.0  # kg/m^3
        self.nu1, self.nu2 = 1e-3, 1e-2       # Dynamic viscosity (Pa·s)
        self.sigma = 0                   # Surface tension (N/m)
        self.gravity = fd.Constant((0.0, -9.81))  # m/s^2
        self.epsilon = 4 * self.cell_size                   # Interface width
        self.m0 = 0                     # Mobility factor

        # Solver parameters
        self.solver_params = {
            "mat_type": "aij",
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_max_it": 50,
            "snes_rtol": 1e-8,
            "snes_atol": 1e-8,
            "ksp_type": "gmres",
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
                "mat_mumps_icntl_14": 200,
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_14": 200,
            },
        }

    def __str__(self):
        if COMM_WORLD.rank == 0:
            return f'''Cahn-Hilliard Navier-Stokes (Parallel on {COMM_WORLD.size} cores):
                Mesh: {self.mesh.num_cells()} cells, {self.mesh.num_vertices()} vertices
                Parameters: ρ₁={self.rho1}, ρ₂={self.rho2}, ν₁={self.nu1}, ν₂={self.nu2}
                σ={self.sigma}, ε={self.epsilon}, M₀={self.m0}'''
        return ""

    @cached_property
    def cell_size(self):
        # calculate the cell diameter
        cell_diameter = fd.CellSize(self.mesh)
        h = fd.Function(fd.FunctionSpace(self.mesh, 'DG', 0))
        h.interpolate(cell_diameter)
        
        return h.dat.data_ro.max()

    @cached_property
    def mesh(self):
        Lx, Ly = 1.0, 2.0
        nx, ny = 30, 60
        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

        # barycentric refinement using Alfeld's split
        # mesh = refine_bary(mesh)

        return mesh

    @cached_property
    def FunctionSpace(self):
        # Scott-Vogelius pressure-robust
        k = 2
        V = fd.VectorFunctionSpace(self.mesh, "CG", k)  # Velocity
        P = fd.FunctionSpace(self.mesh, "DG", k-1)      # Pressure
        Q = M = fd.FunctionSpace(self.mesh, "CG", k)    # Phase field (φ)

        return V * P * Q * M

    @staticmethod
    def potential(x):
        return (1 - x)**2 * x**2

    @staticmethod
    def potential_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    def density(self, phase):
        return self.rho1 + (self.rho2 - self.rho1) * phase
        # return fd.conditional(
        #     phase < 0,
        #     self.rho1,
        #     fd.conditional(
        #         phase > 1, 
        #         self.rho2, 
        #         self.rho1 + phase*(self.rho2 - self.rho1))
        # )

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

        # radius = 0.05
        # n_bubbles = 42
        # centers = np.random.rand(n_bubbles, len(domain_size)) * domain_size

        centers = [[0.5, 1]]
        radius = 0.2

        interface_width = 0.5 * self.epsilon
        initial_phase = 0.0

        for center in centers:
            diff = [(coordinates[i] - center[i])**2 for i in range(len(coordinates))]
            distance = fd.sqrt(sum(diff) + 1e-12)
            bubble = 0.5 * (1.0 - fd.tanh((distance - radius) / interface_width))

            initial_phase = 1.0 - (1.0 - initial_phase) * (1.0 - bubble)

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
        solver = fd.NonlinearVariationalSolver(problem)
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

    def phase_mass(self, phi):
        return fd.assemble(phi * fd.dx)

    def phase_com(self, phi):
        x = fd.SpatialCoordinate(self.mesh)
        mass = self.phase_mass(phi)
        com_x = fd.assemble(phi * x[0] * fd.dx)
        com_y = fd.assemble(phi * x[1] * fd.dx)
        return (com_x/mass, com_y/mass) if abs(mass) > 1e-12 else (0.0, 0.0)

    def run(self):
        w = fd.Function(self.FunctionSpace)
        w_ = fd.Function(self.FunctionSpace)

        self.initialize(*w.subfunctions)
        w_.assign(w)

        u, p, phi, mu = fd.split(w)
        u_, p_, phi_, mu_ = fd.split(w_)

        v, q, psi, nu = fd.TestFunctions(self.FunctionSpace)

        # Setup time parameters
        dt = 1e-4

        momentum = lambda u, p, phi, mu: (
            self.density(phi)*fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
            + self.viscosity(phi)*fd.inner(fd.grad(u), fd.grad(v))
            + self.density(phi)*fd.dot(self.gravity, v)
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
        ) * fd.dx

        bcs = [fd.DirichletBC(self.FunctionSpace.sub(0), fd.Constant((0, 0)), "on_boundary")]

        problem = fd.NonlinearVariationalProblem(F, w, bcs=bcs)
        solver = fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_params
        )
        
        # Time stepping
        total_time = 1.0
        nsteps = int(total_time/dt)
        t = 0.0

        if COMM_WORLD.rank == 0:
            pbar = tqdm(total=total_time, desc="Time Evolution")

        for _ in range(nsteps):
            solver.solve()
            w_.assign(w)
            t += dt

            if COMM_WORLD.rank == 0:
                pbar.update(dt)
                pbar.set_postfix_str(f"t={t:.1f}")

            self.file.write(*w.subfunctions, time=t)

        if COMM_WORLD.rank == 0:
            pbar.close()

if __name__ == '__main__':
    model = CahnHilliardNavierStokes()
    if COMM_WORLD.rank == 0:
        print(model)
    model.run()
