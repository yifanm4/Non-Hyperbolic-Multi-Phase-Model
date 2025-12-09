import numpy as np
import matplotlib.pyplot as plt

class FullPressureTwoFluidSolver:
    def __init__(self, n_cells=100, L=12.0, t_final=0.5):
        self.N = n_cells
        self.L = L
        self.dx = L / n_cells
        self.t_final = t_final
        self.x = np.linspace(self.dx/2, L - self.dx/2, self.N)

        # Physics Constants
        self.rho_l = 1000.0
        self.g = 9.81

        # Initial Conditions (Ransom Faucet)
        self.alpha0 = 0.2
        self.v0 = 10.0

        # State Arrays (Initialized to t=0 condition)
        self.alpha = np.ones(self.N) * self.alpha0
        self.u_l = np.ones(self.N) * self.v0

    def van_leer(self, r):
        return (r + np.abs(r)) / (1.0 + np.abs(r))

    def reconstruct_interface(self, Q, scheme):
        """
        Reconstruct values Q_L at the left side of every face i.
        Face i is between cell i-1 and i.
        """
        Q_L = np.zeros(self.N + 1)

        if scheme == 'Upwind':
            # 1st Order Upwind: Value at face i is simply value of cell i-1
            Q_L[1:] = Q[:]
            Q_L[0]  = Q[0] # Inlet

        elif scheme == 'MUSCL':
            # 2nd Order with Van Leer Limiter
            for i in range(1, self.N):
                if i < 2 or i > self.N-2:
                    Q_L[i] = Q[i-1] # Fallback to 1st order near boundaries
                else:
                    d_minus = Q[i-1] - Q[i-2]
                    d_plus  = Q[i] - Q[i-1]

                    if abs(d_minus) < 1e-10: r = 0.0
                    else: r = d_plus / d_minus

                    phi = self.van_leer(r)
                    Q_L[i] = Q[i-1] + 0.5 * phi * d_minus

            Q_L[0] = Q[0]  # Inlet
            Q_L[-1] = Q[-1] # Outlet

        return Q_L

    def solve(self, scheme='Upwind'):
        # Reset State for clean run
        self.alpha = np.ones(self.N) * self.alpha0
        self.u_l = np.ones(self.N) * self.v0

        t = 0.0
        CFL = 0.1

        while t < self.t_final:
            # 1. Adaptive Time Step
            max_v = np.max(np.abs(self.u_l)) + 1e-6
            dt = CFL * self.dx / max_v
            if t + dt > self.t_final: dt = self.t_final - t

            # 2. Reconstruct Velocities and Alpha at Faces
            # This handles the difference between Upwind vs MUSCL
            u_L = self.reconstruct_interface(self.u_l, scheme)
            alpha_L = self.reconstruct_interface(self.alpha, scheme)

            # 3. Solve Momentum (Predictor)
            # Flux Calculation: (rho_part * u * u)_face
            flux_mom = self.rho_l * (1.0 - alpha_L) * u_L * u_L

            # Boundary Fluxes
            flux_mom[0] = self.rho_l * (1.0 - self.alpha0) * self.v0**2 # Inlet
            flux_mom[-1] = flux_mom[-2] # Outlet (Transmissive)

            # Divergence of Momentum Flux (Vectorized)
            div_flux = (flux_mom[1:] - flux_mom[:-1]) / self.dx

            # Forces
            # Gravity: acts on LIQUID mass (rho_part * g)
            force_grav = (1.0 - self.alpha) * self.rho_l * self.g

            # Pressure Gradient (Free Jet: P_grad = 0)
            grad_P = 0.0
            force_press = -(1.0 - self.alpha) * grad_P

            # Update Momentum
            mass_l = (1.0 - self.alpha) * self.rho_l
            mom_l = mass_l * self.u_l

            rhs_mom = -div_flux + force_grav + force_press
            mom_l_new = mom_l + dt * rhs_mom

            # 4. Solve Continuity (Transport Alpha)
            # Flux Mass = (rho_part * u)_face
            flux_mass = self.rho_l * (1.0 - alpha_L) * u_L

            # Boundary Fluxes
            flux_mass[0] = self.rho_l * (1.0 - self.alpha0) * self.v0
            flux_mass[-1] = flux_mass[-2]

            # Divergence of Mass Flux
            div_mass = (flux_mass[1:] - flux_mass[:-1]) / self.dx

            mass_l_new = mass_l - dt * div_mass

            # 5. Update Primitives
            self.alpha = 1.0 - (mass_l_new / self.rho_l)

            self.u_l = np.zeros_like(mass_l_new)
            mask = mass_l_new > 1e-6
            self.u_l[mask] = mom_l_new[mask] / mass_l_new[mask]

            t += dt

        return self.x, self.alpha, self.u_l

    def analytical_solution_transient(self, t_current):
        alpha_exact = np.zeros_like(self.x)
        vel_exact = np.zeros_like(self.x)
        x_front = self.v0 * t_current + 0.5 * self.g * t_current**2

        for i, xi in enumerate(self.x):
            if xi < x_front:
                # Region 1: Steady State
                vel_exact[i] = np.sqrt(self.v0**2 + 2 * self.g * xi)
                term = ((1.0 - self.alpha0) * self.v0) / vel_exact[i]
                alpha_exact[i] = 1.0 - term
            else:
                # Region 2: Falling Plug
                vel_exact[i] = self.v0 + self.g * t_current
                alpha_exact[i] = self.alpha0
        return self.x, alpha_exact, vel_exact

# --- RUN SIMULATIONS ---
solver = FullPressureTwoFluidSolver(n_cells=100, t_final=0.5)

# 1. Analytical
x_ref, alpha_ref, u_ref = solver.analytical_solution_transient(0.5)

# 2. Simple Upwind
_, alpha_up, u_up = solver.solve(scheme='Upwind')

# 3. MUSCL
_, alpha_muscl, u_muscl = solver.solve(scheme='MUSCL')

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Void Fraction
ax1.set_title("Comparison: Void Fraction (Full Model)")
ax1.plot(x_ref, alpha_ref, 'k--', linewidth=2, label='Exact Analytical')
ax1.plot(solver.x, alpha_up, 'g-o', markersize=4, label='1st Order Upwind', alpha=0.6)
ax1.plot(solver.x, alpha_muscl, 'r-s', markersize=4, label='MUSCL (Van Leer)', alpha=0.8)
ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Void Fraction")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Velocity
ax2.set_title("Comparison: Liquid Velocity")
ax2.plot(x_ref, u_ref, 'k--', label='Exact Analytical')
ax2.plot(solver.x, u_up, 'g-', label='1st Order Upwind')
ax2.plot(solver.x, u_muscl, 'r-', label='MUSCL (Van Leer)')
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Velocity (m/s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
