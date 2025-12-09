import numpy as np
import matplotlib.pyplot as plt

class CoupledTwoFluidSolver:
    def __init__(self, n_cells=100, L=12.0, t_final=0.5):
        self.N = n_cells
        self.L = L
        self.dx = L / n_cells
        self.t_final = t_final
        self.x = np.linspace(self.dx/2, L - self.dx/2, self.N)

        # Physics Constants (Water Faucet)
        self.rho_l_const = 1000.0  # Liquid is Incompressible
        self.R_gas = 287.0         # Gas Constant
        self.T_gas = 300.0         # Isothermal T
        self.g = 9.81              # Gravity
        self.P_outlet = 100000.0   # Boundary Pressure

        # Drag Coefficient (Coupling)
        # Tuned to reproduce the slight lag between phases
        self.Cd = 10.0

        # Initial Conditions (Ransom Faucet)
        self.alpha0 = 0.2
        self.vl0 = 10.0
        self.vg0 = 0.0
        self.P0 = 100000.0

        # --- Conserved Variables (U) ---
        # U[0] = Gas Mass Density     (alpha * rho_g)
        # U[1] = Liquid Mass Density  ( (1-alpha) * rho_l )
        # U[2] = Gas Momentum         (alpha * rho_g * u_g)
        # U[3] = Liquid Momentum      ( (1-alpha) * rho_l * u_l )

        self.U = np.zeros((self.N, 4))

        # Initialize
        # 1. Derived Quantities
        rho_g_init = self.P0 / (self.R_gas * self.T_gas)

        self.U[:, 0] = self.alpha0 * rho_g_init
        self.U[:, 1] = (1.0 - self.alpha0) * self.rho_l_const
        self.U[:, 2] = self.U[:, 0] * self.vg0
        self.U[:, 3] = self.U[:, 1] * self.vl0

    def get_primitives(self, U):
        """ Recover alpha, P, ug, ul from Conserved U """
        # 1. Recover Liquid Fraction & Alpha
        # m_l = (1-alpha)*rho_l_const  => (1-alpha) = m_l / rho_l_const
        # alpha = 1 - (m_l / rho_l_const)

        m_g = U[:, 0]
        m_l = U[:, 1]
        p_g = U[:, 2]
        p_l = U[:, 3]

        # Safety clamp to prevent vacuum or overfilling
        alpha = 1.0 - (m_l / self.rho_l_const)
        alpha = np.clip(alpha, 1e-4, 0.9999)

        # 2. Recover Densities
        # rho_g_actual = m_g / alpha
        rho_g = m_g / alpha

        # 3. Recover Pressure (Equation of State: Ideal Gas)
        P = rho_g * self.R_gas * self.T_gas

        # 4. Recover Velocities
        u_g = np.zeros_like(m_g)
        mask_g = m_g > 1e-8
        u_g[mask_g] = p_g[mask_g] / m_g[mask_g]

        u_l = np.zeros_like(m_l)
        mask_l = m_l > 1e-8
        u_l[mask_l] = p_l[mask_l] / m_l[mask_l]

        return alpha, P, u_g, u_l

    def van_leer(self, r):
        return (r + np.abs(r)) / (1.0 + np.abs(r))

    def reconstruct(self, q):
        """ MUSCL Reconstruction of a variable q """
        q_L = np.zeros(self.N + 1)
        q_R = np.zeros(self.N + 1)

        # Slopes
        dq = np.zeros(self.N)
        # Central difference for internal
        for i in range(1, self.N-1):
            d_minus = q[i] - q[i-1]
            d_plus = q[i+1] - q[i]
            if abs(d_minus) < 1e-10: r = 0.0
            else: r = d_plus/d_minus
            phi = self.van_leer(r)
            dq[i] = 0.5 * phi * d_minus # Limited Slope

        # Reconstruct Face Values
        for i in range(1, self.N):
            q_L[i] = q[i-1] + dq[i-1] # Right side of cell i-1
            q_R[i] = q[i]   - dq[i]   # Left side of cell i

        # Boundaries (First Order)
        q_L[0], q_R[0] = q[0], q[0]
        q_L[-1], q_R[-1] = q[-1], q[-1]

        return q_L, q_R

    def get_flux(self, rho_L, rho_R, u_L, u_R):
        """ HLL-style Diffusive Flux or Simple Upwind """
        # Simple Upwind Advection Flux for Robustness in Two-Fluid
        # F = rho * u
        # If u > 0: F = rho_L * u_L

        flux = np.zeros_like(rho_L)

        # Vectorized Upwind
        vel_avg = 0.5*(u_L + u_R)

        # Positive Flow
        mask_pos = vel_avg > 0
        flux[mask_pos] = rho_L[mask_pos] * u_L[mask_pos]

        # Negative Flow
        mask_neg = ~mask_pos
        flux[mask_neg] = rho_R[mask_neg] * u_R[mask_neg]

        return flux

    def solve(self):
        t = 0.0
        # Small CFL for compressible sound waves (gas sound speed ~300m/s)
        CFL = 0.1

        while t < self.t_final:
            # 1. Get Primitives
            alpha, P, ug, ul = self.get_primitives(self.U)

            # 2. Time Step
            c_sound = np.sqrt(self.R_gas * self.T_gas)
            max_vel = np.max(np.abs(ug)) + c_sound
            dt = CFL * self.dx / max_vel
            if t + dt > self.t_final: dt = self.t_final - t

            # 3. Reconstruct Primitives for Fluxes
            # Conserved variables are advected
            m_g = self.U[:, 0]
            m_l = self.U[:, 1]
            p_g = self.U[:, 2]
            p_l = self.U[:, 3]

            mg_L, mg_R = self.reconstruct(m_g)
            ml_L, ml_R = self.reconstruct(m_l)
            # Velocities needed for upwinding direction
            ug_L, ug_R = self.reconstruct(ug)
            ul_L, ul_R = self.reconstruct(ul)

            # Also reconstruct Momentum for advection
            pg_L, pg_R = self.reconstruct(p_g)
            pl_L, pl_R = self.reconstruct(p_l)

            # 4. Compute Advective Fluxes (Method of Lines)
            # Mass Fluxes
            F_mg = self.get_flux(mg_L, mg_R, ug_L, ug_R)
            F_ml = self.get_flux(ml_L, ml_R, ul_L, ul_R)

            # Momentum Fluxes (Convection part only: rho*u*u)
            # We treat Momentum as the scalar being advected by velocity u
            F_pg = self.get_flux(pg_L, pg_R, ug_L, ug_R)
            F_pl = self.get_flux(pl_L, pl_R, ul_L, ul_R)

            # Boundaries (Inlet Fixed, Outlet Fixed Pressure/Backflow)
            # Inlet (Top, i=0 face)
            rho_g_in = self.P0 / (self.R_gas * self.T_gas)
            F_mg[0] = self.alpha0 * rho_g_in * self.vg0
            F_ml[0] = (1-self.alpha0) * self.rho_l_const * self.vl0
            F_pg[0] = F_mg[0] * self.vg0
            F_pl[0] = F_ml[0] * self.vl0

            # Outlet (Bottom, i=N face) - Transmissive / Zero Gradient
            F_mg[-1] = F_mg[-2]
            F_ml[-1] = F_ml[-2]
            F_pg[-1] = F_pg[-2]
            F_pl[-1] = F_pl[-2]

            # 5. Calculate RHS (Flux Divergence)
            rhs_mg = -(F_mg[1:] - F_mg[:-1]) / self.dx
            rhs_ml = -(F_ml[1:] - F_ml[:-1]) / self.dx
            rhs_pg = -(F_pg[1:] - F_pg[:-1]) / self.dx
            rhs_pl = -(F_pl[1:] - F_pl[:-1]) / self.dx

            # 6. Source Terms (The Coupling)
            # Gravity
            grav_g = m_g * self.g
            grav_l = m_l * self.g

            # Pressure Gradient (Central Difference)
            # Staggered-like gradient for stability: (P_i+1 - P_i-1)/2dx
            grad_P = np.zeros(self.N)
            grad_P[1:-1] = (P[2:] - P[:-2]) / (2*self.dx)
            grad_P[0]    = (P[1] - P[0]) / self.dx   # Boundary approx
            grad_P[-1]   = (P[-1] - P[-2]) / self.dx

            # Term: - alpha * grad_P
            src_press_g = - alpha * grad_P
            src_press_l = - (1.0 - alpha) * grad_P

            # Drag Force (Interfacial Friction)
            # F_drag = Cd * (ul - ug)
            # Adding standard Drag stabilizes the relative velocity
            drag = self.Cd * (ul - ug) # Positive on gas if ul > ug

            # 7. Update Conserved Variables
            self.U[:, 0] += dt * rhs_mg
            self.U[:, 1] += dt * rhs_ml

            self.U[:, 2] += dt * (rhs_pg + grav_g + src_press_g + drag)
            self.U[:, 3] += dt * (rhs_pl + grav_l + src_press_l - drag)

            # Update time
            print(f"Time: {t:.3f}s, dt: {dt:.5f}s")
            t += dt

        return self.x, alpha, ug, ul, P

# --- RUN ---
solver = CoupledTwoFluidSolver(n_cells=100, t_final=0.05)
x, alpha, ug, ul, P = solver.solve()

# --- PLOTTING (Matching Reference Graph Style) ---
fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

# 1. Pressure
axs[0].plot(x, P, 'tab:purple', marker='o', markersize=3, label='Pressure')
axs[0].set_ylabel('P (Pa)')
axs[0].grid(True)
axs[0].set_title(f'Coupled Two-Fluid Model Results (t=0.5s)')

# 2. Void Fraction (Alpha)
axs[1].plot(x, alpha, 'tab:brown', marker='o', markersize=3, label='Alpha')
axs[1].set_ylabel('alpha (-)')
axs[1].grid(True)
# Note: Reference graph shows alpha rising to ~0.4.
# Physics: Liquid accelerates -> column narrows -> Gas fraction MUST increase.
# This plot should confirm that behavior.

# 3. Liquid Velocity
axs[2].plot(x, ul, 'tab:red', marker='o', markersize=3, label='U_liq')
axs[2].set_ylabel('ul (m/s)')
axs[2].grid(True)

# 4. Gas Velocity
axs[3].plot(x, ug, 'tab:blue', marker='o', markersize=3, label='U_gas')
axs[3].set_ylabel('ug (m/s)')
axs[3].set_xlabel('Position (m)')
axs[3].grid(True)

plt.tight_layout()
plt.show()
