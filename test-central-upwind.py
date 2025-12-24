import numpy as np
import matplotlib.pyplot as plt

class RansomFaucetSolver:
    def __init__(self, n_cells=100, length=12.0, t_end=0.6, cfl=0.4):
        self.N = n_cells
        self.L = length
        self.dx = length / n_cells
        self.t_end = t_end
        self.CFL = cfl

        # Physics constants
        self.g = 9.81
        self.alpha_0 = 0.8  # Initial liquid fraction (Void fraction = 0.2)
        self.v_0 = 10.0     # Initial velocity

        # Initialize Grid (Cell centers)
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.N)

        # State Vector U = [alpha, alpha*v] (Liquid mass, Liquid momentum)
        # We assume constant density for liquid, so we drop rho_l scaling
        self.U = np.zeros((self.N, 2))

        # Initial Conditions (Uniform flow initially)
        self.U[:, 0] = self.alpha_0
        self.U[:, 1] = self.alpha_0 * self.v_0

    def get_primitive(self, U):
        """Recover alpha and velocity from conserved variables."""
        alpha = U[:, 0]
        # Avoid division by zero
        alpha_safe = np.where(alpha < 1e-6, 1e-6, alpha)
        v = U[:, 1] / alpha_safe
        return alpha, v

    def flux(self, U):
        """Compute Flux F(U) = [alpha*v, alpha*v^2]."""
        alpha, v = self.get_primitive(U)
        F = np.zeros_like(U)
        F[:, 0] = U[:, 1]           # Mass flux
        F[:, 1] = U[:, 1] * v       # Momentum flux
        return F

    def source(self, U):
        """Compute Source S(U) = [0, alpha*g]."""
        alpha, _ = self.get_primitive(U)
        S = np.zeros_like(U)
        S[:, 0] = 0.0
        S[:, 1] = alpha * self.g    # Gravity force
        return S

    def minmod(self, a, b):
        """The MinMod slope limiter to prevent oscillations."""
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    def reconstruct_interfaces(self, U):
        """
        Reconstruct values at cell interfaces (U_L, U_R) using
        Piecewise Linear Reconstruction with MinMod limiter.
        """
        # Calculate slopes
        dU = np.zeros_like(U)
        # Central difference for internal nodes
        delta_central = 0.5 * (U[2:, :] - U[:-2, :])
        # One-sided for boundaries (simplified)
        dU[1:-1, :] = self.minmod((U[1:-1, :] - U[:-2, :]), (U[2:, :] - U[1:-1, :]))

        # Reconstruct Left and Right states at interface i+1/2
        # U_L[i] is the value at right boundary of cell i
        # U_R[i] is the value at left boundary of cell i+1

        U_L = U + 0.5 * dU
        U_R = U - 0.5 * dU

        # Shift to align interfaces:
        # Interface j is between cell j and j+1.
        # Left state at interface j is U_L[j]
        # Right state at interface j is U_R[j+1]

        # We prepare arrays for interfaces 0 to N (N+1 interfaces)
        # Pad with boundary conditions
        U_L_face = np.vstack([self.boundary_inlet(U[0]), U_L])
        U_R_face = np.vstack([self.boundary_inlet(U[0]), U_R])

        # Shift for correct indexing:
        # Face i connects Cell i-1 and Cell i.
        # But commonly we calculate flux at i+1/2.
        # Let's align: Flux[i] is flux at right face of cell i.

        # Left side of interface i+1/2 comes from cell i
        U_interface_L = U_L
        # Right side of interface i+1/2 comes from cell i+1
        U_interface_R = np.roll(U_R, -1, axis=0)

        # Apply Boundary Conditions to the "Ghost" values
        # Inlet (Left) - Fixed
        U_inlet = np.array([self.alpha_0, self.alpha_0 * self.v_0])
        # Outlet (Right) - Zero Gradient (Extrapolation)
        U_outlet = U[-1]

        # Fix the rolled array at the end
        U_interface_R[-1] = U_outlet

        return U_interface_L, U_interface_R, U_inlet

    def boundary_inlet(self, u_first):
        return np.array([self.alpha_0, self.alpha_0 * self.v_0])

    def kurganov_flux(self, U):
        """
        Compute the Central-Upwind Fluxes.
        """
        # 1. Reconstruction
        # UL: State at the LEFT of the interface (from cell i)
        # UR: State at the RIGHT of the interface (from cell i+1)
        # Note: We compute fluxes for interfaces 0 to N-1 (right faces of all cells)

        # Get slopes
        theta = 1.5 # Minmod parameter (1=dissipative, 2=compressive)
        slopes = np.zeros_like(U)
        diff_up = U - np.roll(U, 1, axis=0)
        diff_down = np.roll(U, -1, axis=0) - U

        # Apply Minmod limiter component-wise
        slopes = self.minmod(theta*diff_up, theta*diff_down)

        # Reconstruct values at Right face of cell i (Interface i+1/2)
        U_L = U + 0.5 * slopes

        # Reconstruct values at Left face of cell i+1 (Interface i+1/2)
        U_R = np.roll(U - 0.5 * slopes, -1, axis=0)

        # Fix Boundary Conditions for reconstruction
        # At the last cell, U_R needs to be extrapolated
        U_R[-1] = U[-1]

        # 2. Compute Physics at Interfaces
        F_L = self.flux(U_L)
        F_R = self.flux(U_R)

        _, v_L = self.get_primitive(U_L)
        _, v_R = self.get_primitive(U_R)

        # 3. Estimate Local Wave Speeds (Eigenvalues)
        # For this system, eigenvalue approx equals velocity v
        # a_plus = max(v_R, v_L, 0)
        # a_minus = min(v_R, v_L, 0)

        # We need to broadcast max/min over the state vector dimension for vectorized math
        a_plus = np.maximum(np.maximum(v_R, v_L), 0)[:, np.newaxis]
        a_minus = np.minimum(np.minimum(v_R, v_L), 0)[:, np.newaxis]

        # Avoid division by zero if a_plus == a_minus (stagnant)
        epsilon = 1e-12
        denom = a_plus - a_minus
        denom[denom < epsilon] = epsilon

        # 4. Kurganov-Tadmor Flux Formula
        # H = (a+ F_L - a- F_R) / (a+ - a-) + (a+ a-)/(a+ - a-) * (U_R - U_L)
        H = (a_plus * F_L - a_minus * F_R) / denom + \
            (a_plus * a_minus) / denom * (U_R - U_L)

        return H

    def analytical_solution(self, t):
        """Exact solution for the Ransom Faucet."""
        x_analytical = np.linspace(0, self.L, 500)
        alpha_analytical = np.zeros_like(x_analytical)

        # The location of the wave front
        x_front = self.v_0 * t + 0.5 * self.g * t**2

        for i, x in enumerate(x_analytical):
            if x <= x_front:
                # Steady state profile
                v_x = np.sqrt(self.v_0**2 + 2 * self.g * x)
                alpha_analytical[i] = self.alpha_0 * self.v_0 / v_x
            else:
                # Initial condition state
                alpha_analytical[i] = self.alpha_0

        return x_analytical, alpha_analytical

    def solve(self):
        t = 0.0

        print(f"Starting simulation. Target Time: {self.t_end}s")

        while t < self.t_end:
            # Dynamic Time Step
            _, v = self.get_primitive(self.U)
            max_v = np.max(np.abs(v)) + 1e-6
            dt = self.CFL * self.dx / max_v
            if t + dt > self.t_end:
                dt = self.t_end - t

            # Runge-Kutta 2nd Order (Heun's Method)

            # Step 1: Predictor
            # Compute Fluxes at cell RIGHT faces
            H = self.kurganov_flux(self.U)
            # Flux entering from left (H_{i-1/2}) is just H rolled
            H_left = np.roll(H, 1, axis=0)

            # Fix Inlet Flux BC (Exact Inflow)
            U_inlet = np.array([self.alpha_0, self.alpha_0 * self.v_0])
            F_inlet = self.flux(U_inlet[np.newaxis, :])[0]
            H_left[0] = F_inlet # Overwrite first face flux

            # Source Terms
            S = self.source(self.U)

            # Forward Euler Step
            dAdt = -(H - H_left)/self.dx + S
            U_star = self.U + dt * dAdt

            # Step 2: Corrector
            H_star = self.kurganov_flux(U_star)
            H_left_star = np.roll(H_star, 1, axis=0)
            H_left_star[0] = F_inlet

            S_star = self.source(U_star)

            dAdt_star = -(H_star - H_left_star)/self.dx + S_star

            self.U = 0.5 * self.U + 0.5 * (U_star + dt * dAdt_star)

            t += dt

        return self.x, self.get_primitive(self.U)

# --- Execute and Plot ---
solver = RansomFaucetSolver(n_cells=1000, t_end=0.6)
x_num, (alpha_num, v_num) = solver.solve()
x_ref, alpha_ref = solver.analytical_solution(0.6)

# Plot Void Fraction (Alpha Gas = 1 - Alpha Liquid)
plt.figure(figsize=(10, 6))
plt.plot(x_ref, 1.0 - alpha_ref, 'k-', linewidth=2, label='Analytical Solution')
plt.plot(x_num, 1.0 - alpha_num, 'r', markersize=4, label='Central-Upwind (Kurganov)')

plt.title(f"Ransom Faucet Problem: Void Fraction at t=0.6s")
plt.xlabel("Pipe Length (m)")
plt.ylabel("Gas Void Fraction")
plt.ylim(0.15, 0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
