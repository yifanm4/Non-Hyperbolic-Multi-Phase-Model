import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

class TwoFluidRoeSolver:
    def __init__(self, n_cells=100, length=12.0, t_end=0.6, cfl=0.5):
        self.N = n_cells
        self.L = length
        self.dx = length / n_cells
        self.t_end = t_end
        self.CFL = cfl

        # Grid
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.N)

        # Physics Constants (Stiffened Gas EOS)
        self.g = 9.81
        self.rho0_l = 1000.0
        self.c_l = 100.0   # Artificial sound speed for liquid (reduced for stiffness)
        self.rho0_g = 1.0
        self.c_g = 100.0   # Artificial sound speed for gas
        self.p0 = 1e5

        # State Vector: U = [alpha_l*rho_l, alpha_g*rho_g, alpha_l*rho_l*u_l, alpha_g*rho_g*u_g]
        self.U = np.zeros((self.N, 4))

        # Initialize (Water Faucet)
        self.init_water_faucet()

    def eos_pressure(self, rho_l, rho_g):
        """
        Simplified Stiffened Gas EOS.
        We assume mechanical equilibrium P_l = P_g.
        P = c^2 * (rho - rho0) + P0
        """
        # Note: In a full code, you iterate to match pressures.
        # Here we approximate P based on the dominant liquid phase or average.
        pl = self.c_l**2 * (rho_l - self.rho0_l) + self.p0
        return pl

    def get_primitive(self, U):
        """
        Convert Conservative (U) to Primitive (V).
        U = [m_l, m_g, mom_l, mom_g]
        V = [alpha_l, P, u_l, u_g]
        """
        epsilon = 1e-8
        m_l, m_g = U[0], U[1]
        mom_l, mom_g = U[2], U[3]

        # 1. Recover alpha (Simplification for Stiffened Gas)
        # alpha_l * rho_l + alpha_g * rho_g = total_density? No.
        # We solve: m_l/rho_l(P) + m_g/rho_g(P) = 1.
        # For this demo, we assume approximate constant density for alpha recovery
        # to avoid non-linear root finding at every cell (too slow for python script).

        # Approx: alpha_l approx m_l / rho0_l
        # This works well for water faucet where water is nearly incompressible.
        al = m_l / self.rho0_l
        al = np.clip(al, 1e-6, 1.0 - 1e-6)
        ag = 1.0 - al

        # 2. Recover densities
        rho_l = m_l / al
        rho_g = m_g / ag

        # 3. Recover velocities
        ul = mom_l / (m_l + epsilon)
        ug = mom_g / (m_g + epsilon)

        # 4. Recover Pressure
        P = self.eos_pressure(rho_l, rho_g)

        return np.array([al, P, ul, ug])

    def conservative_from_primitive(self, V):
        al, P, ul, ug = V
        ag = 1.0 - al

        rho_l = (P - self.p0)/(self.c_l**2) + self.rho0_l
        rho_g = (P - self.p0)/(self.c_g**2) + self.rho0_g

        U = np.zeros(4)
        U[0] = al * rho_l           # m_l
        U[1] = ag * rho_g           # m_g
        U[2] = al * rho_l * ul      # mom_l
        U[3] = ag * rho_g * ug      # mom_g
        return U

    def flux_function(self, U):
        """Physical Flux F(U)"""
        m_l, m_g, mom_l, mom_g = U
        # Recover Primitives needed for flux
        V = self.get_primitive(U)
        al, P, ul, ug = V
        ag = 1.0 - al

        F = np.zeros(4)
        F[0] = mom_l                            # m_l * u_l
        F[1] = mom_g                            # m_g * u_g
        F[2] = mom_l * ul + al * P              # m_l u_l^2 + al P
        F[3] = mom_g * ug + ag * P              # m_g u_g^2 + ag P
        return F

    def get_jacobian_num(self, U_state):
        """
        Numerically compute the Jacobian Matrix A = dF/dU + NonCons_Terms.
        Standard Roe requires A. Here we include the non-conservative p*grad(alpha)
        into the matrix for the Riemann solver (Path Conservative approach).
        """
        epsilon = 1e-5
        nd = 4
        A = np.zeros((nd, nd))

        F_base = self.flux_function(U_state)
        V_base = self.get_primitive(U_state)
        P_base = V_base[1]

        # Compute Jacobian via Finite Difference column by column
        for i in range(nd):
            U_pert = U_state.copy()
            U_pert[i] += epsilon

            F_pert = self.flux_function(U_pert)

            # Derivative of Flux
            dFdU = (F_pert - F_base) / epsilon
            A[:, i] = dFdU

            # --- Non-Conservative Term Correction ---
            # System: dU/dt + dF/dx + H d(alpha)/dx = 0
            # We fold H d(alpha)/dx into A * dU/dx
            # H = [0, 0, -P, P]^T approx? (Interfacial pressure terms)
            # Actually, standard TFM momentum eq: d(alpha P)/dx - P d(alpha)/dx
            # The flux F contains d(alpha P)/dx.
            # We need to Subtract P d(alpha)/dx term.
            # Term is:  - P_i * d(alpha_l)/dx

            # Find d(alpha)/dU_i
            V_pert = self.get_primitive(U_pert)
            dalpha_dU = (V_pert[0] - V_base[0]) / epsilon

            # Interfacial pressure (assume P_int = P)
            # Source vector H for alpha gradient: [0, 0, -P, +P]?
            # Liquid Mom eq: ... + alpha dp/dx. (Conservative form has d(alpha p)/dx).
            # d(alpha p)/dx = alpha dp/dx + p dalpha/dx.
            # TFM requires: alpha dp/dx.
            # So we must Subtract p dalpha/dx from the conservative flux derivative.

            # H_vector representing -P * dalpha/dx
            H = np.zeros(4)
            H[2] = -P_base * dalpha_dU  # Liquid
            H[3] = +P_base * dalpha_dU  # Gas (d_alpha_g = -d_alpha_l)

            A[:, i] += H

        return A

    def roe_flux(self, U_L, U_R):
        """
        Roe Flux: F_roe = 0.5*(F_L + F_R) - 0.5 * |A_roe| * (U_R - U_L)
        """
        # 1. Roe Average State (Simple arithmetic avg for TFM usually sufficient)
        # sqrt_rho weighting is ideal for Euler, but complex for 4-eq TFM
        U_avg = 0.5 * (U_L + U_R)

        # 2. Compute Physical Fluxes
        F_L = self.flux_function(U_L)
        F_R = self.flux_function(U_R)

        # 3. Compute Jacobian at Average State
        A_roe = self.get_jacobian_num(U_avg)

        # 4. Eigendecomposition
        # A = R * L * R^-1
        evals, R = eig(A_roe)
        evals = np.real(evals)
        R = np.real(R)

        # Entropy Fix (prevent zero wave speed)
        abs_evals = np.abs(evals)
        delta = 0.2 * (np.abs(np.max(evals)) + 1.0)
        abs_evals = np.where(abs_evals < delta,
                             (evals**2 + delta**2)/(2*delta),
                             abs_evals)

        # Construct Dissipation Matrix |A| = R * |L| * R^-1
        Lambda_abs = np.diag(abs_evals)

        try:
            R_inv = np.linalg.inv(R)
            Abs_A = R @ Lambda_abs @ R_inv
        except np.linalg.LinAlgError:
            # Fallback to Rusanov (Lax-Friedrichs) if singular
            max_wave = np.max(abs_evals)
            Abs_A = max_wave * np.eye(4)

        # 5. Final Flux
        F_num = 0.5 * (F_L + F_R) - 0.5 * Abs_A @ (U_R - U_L)

        return F_num

    def source_terms(self, U):
        """Gravity Source"""
        V = self.get_primitive(U)
        al, P, ul, ug = V
        ag = 1.0 - al

        rho_l = (P - self.p0)/self.c_l**2 + self.rho0_l
        rho_g = (P - self.p0)/self.c_g**2 + self.rho0_g

        S = np.zeros(4)
        # Mass eq: 0
        # Mom eq: alpha * rho * g
        S[2] = al * rho_l * self.g
        S[3] = ag * rho_g * self.g
        return S

    def init_water_faucet(self):
        # Ransom Water Faucet IC
        # Tube length 12m.
        # t=0: alpha_l = 0.8, u_l = 10 m/s, u_g = 0.
        # Inlet (x=0): Fixed at IC values.

        for i in range(self.N):
            al = 0.8
            P = self.p0
            ul = 10.0
            ug = 0.0

            V = np.array([al, P, ul, ug])
            self.U[i] = self.conservative_from_primitive(V)

    def solve(self):
        t = 0.0
        print(f"Starting Roe Solver (N={self.N})...")

        while t < self.t_end:
            # 1. Adaptive Time Step
            # Get max wave speed approx
            max_ws = 0.0
            for i in range(0, self.N, 5): # sample grid for speed
                V = self.get_primitive(self.U[i])
                ws = max(abs(V[2])+self.c_l, abs(V[3])+self.c_g)
                if ws > max_ws: max_ws = ws

            dt = self.CFL * self.dx / (max_ws + 1e-6)
            if t + dt > self.t_end: dt = self.t_end - t

            # 2. Reconstruction (1st Order for robustness in this demo)
            # U_L[i] = U[i], U_R[i] = U[i+1]
            # We calculate Fluxes at faces i+1/2

            Fluxes = np.zeros((self.N + 1, 4))

            for i in range(self.N + 1):
                # Boundary Conditions (Ghost Cells)
                if i == 0:
                    # Inlet: Dirichlet (Fixed)
                    U_L = self.U[0].copy() # Simplification: Flux calculated using boundary val
                    U_R = self.U[0].copy()
                elif i == self.N:
                    # Outlet: Transmissive (Zero Gradient)
                    U_L = self.U[self.N-1].copy()
                    U_R = self.U[self.N-1].copy()
                else:
                    # Interior
                    U_L = self.U[i-1]
                    U_R = self.U[i]

                Fluxes[i] = self.roe_flux(U_L, U_R)

            # 3. Update Conservative Variables
            U_new = np.zeros_like(self.U)
            for i in range(self.N):
                # Flux Divergence
                dF = Fluxes[i+1] - Fluxes[i]

                # Source Term
                S = self.source_terms(self.U[i])

                U_new[i] = self.U[i] - (dt/self.dx) * dF + dt * S

            self.U = U_new
            t += dt

        print("Simulation Complete.")
        return self.get_results()

    def get_results(self):
        # Extract Alpha profile
        alpha_res = np.zeros(self.N)
        for i in range(self.N):
            V = self.get_primitive(self.U[i])
            alpha_res[i] = V[0]
        return self.x, alpha_res

# --- Analytical Solution Helper ---
def analytical_solution(x, t, g=9.81, v0=10.0, alpha0=0.8):
    """
    Analytical solution for Water Faucet.
    Fluid accelerates: v(x) = sqrt(v0^2 + 2gx)
    Mass conservation: alpha * v = alpha0 * v0
    """
    # Location of the wave front at time t
    x_front = v0 * t + 0.5 * g * t**2

    alpha_ana = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] <= x_front:
            # Steady state profile
            v = np.sqrt(v0**2 + 2*g*x[i])
            alpha_ana[i] = alpha0 * v0 / v
        else:
            # Initial state
            alpha_ana[i] = alpha0

    return alpha_ana

# --- Main Execution ---
if __name__ == "__main__":
    # Parameters
    L = 12.0
    N = 100
    T_final = 0.6

    # Run Solver
    solver = TwoFluidRoeSolver(n_cells=N, length=L, t_end=T_final)
    x, alpha_num = solver.solve()

    # Get Analytical
    alpha_exact = analytical_solution(x, T_final)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, alpha_exact, 'k-', linewidth=2, label='Analytical')
    plt.plot(x, alpha_num, 'ro-', markersize=4, label="Roe's Method (Num. Jacobian)")

    plt.title(f"Water Faucet Problem (T={T_final}s) - Roe's Method")
    plt.xlabel("Position (m)")
    plt.ylabel("Liquid Volume Fraction")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.show()
