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
        self.alpha0 = 0.2
        self.v0 = 10.0

        # State Arrays
        self.alpha = np.ones(self.N) * self.alpha0
        self.u_l = np.ones(self.N) * self.v0

        # History for Entropy Viscosity
        self.alpha_hist = [self.alpha.copy()] * 3

    def compute_entropy_viscosity(self, dt, v_field):
        # 1. Update History
        self.alpha_hist.append(self.alpha.copy())
        if len(self.alpha_hist) > 3: self.alpha_hist.pop(0)

        if len(self.alpha_hist) < 3:
            return np.zeros(self.N + 1)

        # 2. Compute Residual: | d(alpha)/dt + div(flux) |
        # Safety: Check for tiny dt to avoid divide-by-zero if simulation crashes
        if dt < 1e-12: return np.zeros(self.N + 1)

        d_dt = (3*self.alpha_hist[-1] - 4*self.alpha_hist[-2] + self.alpha_hist[-3]) / (2*dt)

        f = self.alpha * v_field
        df_dx = np.zeros(self.N)
        df_dx[1:-1] = (f[2:] - f[:-2]) / (2*self.dx)
        df_dx[0] = (f[1] - f[0]) / self.dx
        df_dx[-1] = (f[-1] - f[-2]) / self.dx

        # Residual
        Residual = np.abs(d_dt + df_dx)

        # 3. Normalization
        norm = np.max(np.abs(self.alpha - np.mean(self.alpha))) + 1e-10

        # 4. Entropy Viscosity
        c_E = 1.0
        nu_E = c_E * (self.dx**2) * Residual / norm

        # 5. Limiters
        c_max = 0.5
        nu_max = c_max * self.dx * np.abs(v_field)
        nu_cell = np.minimum(nu_max, nu_E)

        # Map to Faces
        nu_face = np.zeros(self.N + 1)
        nu_face[1:-1] = 0.5 * (nu_cell[1:] + nu_cell[:-1])
        nu_face[0] = nu_cell[0]
        nu_face[-1] = nu_cell[-1]

        return nu_face

    def check_stability(self):
        # Returns True if simulation has blown up
        if np.any(np.isnan(self.u_l)) or np.any(np.isinf(self.u_l)):
            return False
        if np.any(np.isnan(self.alpha)) or np.any(np.isinf(self.alpha)):
            return False
        if np.max(np.abs(self.u_l)) > 1e4: # Velocity sanity check
            return False
        return True

    def solve(self, scheme='Upwind1'):
        # Reset State
        self.alpha = np.ones(self.N) * self.alpha0
        self.u_l = np.ones(self.N) * self.v0
        self.alpha_hist = [self.alpha.copy()] * 3

        t = 0.0
        CFL = 0.1

        visc_record = np.zeros(self.N + 1)

        while t < self.t_final:
            # 0. Safety Check
            if not self.check_stability():
                print(f"  [Warning] Simulation blew up (NaN/Inf) at t={t:.4f}s with scheme {scheme}")
                break

            # 1. Adaptive Time Step
            max_v = np.max(np.abs(self.u_l)) + 1e-6
            dt = CFL * self.dx / max_v
            if t + dt > self.t_final: dt = self.t_final - t

            # --- 2. Compute Viscosity (if EVM enabled) ---
            if 'EVM' in scheme:
                nu_visc = self.compute_entropy_viscosity(dt, self.u_l)
                visc_record = np.maximum(visc_record, nu_visc)
            else:
                nu_visc = np.zeros(self.N + 1)

            # --- 3. Compute Fluxes at Faces ---
            flux_mass = np.zeros(self.N + 1)
            flux_mom  = np.zeros(self.N + 1)

            for i in range(1, self.N):
                # Basic Neighbors
                rho_L = (1 - self.alpha[i-1])*self.rho_l
                rho_R = (1 - self.alpha[i])*self.rho_l
                u_L = self.u_l[i-1]
                u_R = self.u_l[i]

                # --- FLUX RECONSTRUCTION ---

                # A. 1st Order Upwind
                if 'Upwind1' in scheme:
                    rho_face = rho_L
                    u_face   = u_L

                # B. Central Difference
                elif 'Central' in scheme:
                    rho_face = 0.5 * (rho_L + rho_R)
                    u_face   = 0.5 * (u_L + u_R)

                # C. 2nd Order Upwind (Linear Extrapolation)
                elif 'Upwind2' in scheme:
                    # Q_face = 1.5*Q_{i-1} - 0.5*Q_{i-2}
                    if i >= 2:
                        rho_face = 1.5*rho_L - 0.5*((1 - self.alpha[i-2])*self.rho_l)
                        u_face   = 1.5*u_L   - 0.5*self.u_l[i-2]
                    else:
                        rho_face = rho_L
                        u_face   = u_L

                # Convective Flux
                f_mass_conv = rho_face * u_face
                f_mom_conv  = rho_face * u_face**2

                # --- ADD VISCOSITY ---
                # Diffusive Flux = - nu * (Q_R - Q_L)/dx
                grad_rho = (rho_R - rho_L) / self.dx
                grad_mom = (rho_R*u_R - rho_L*u_L) / self.dx

                flux_mass[i] = f_mass_conv - nu_visc[i] * grad_rho
                flux_mom[i]  = f_mom_conv  - nu_visc[i] * grad_mom

            # Boundary Fluxes (Inlet Fixed, Outlet Transmissive)
            rho_in = (1.0 - self.alpha0)*self.rho_l
            flux_mass[0] = rho_in * self.v0
            flux_mom[0]  = rho_in * self.v0**2
            flux_mass[-1] = flux_mass[-2]
            flux_mom[-1]  = flux_mom[-2]

            # --- 4. Update Equations ---

            # Divergences
            div_mass = (flux_mass[1:] - flux_mass[:-1]) / self.dx
            div_mom  = (flux_mom[1:] - flux_mom[:-1]) / self.dx

            # Forces
            force_grav = (1.0 - self.alpha) * self.rho_l * self.g

            mass_new = (1-self.alpha)*self.rho_l - dt * div_mass
            mom_new  = (1-self.alpha)*self.rho_l*self.u_l + dt * (-div_mom + force_grav)

            # Recover Primitives
            self.alpha = 1.0 - (mass_new / self.rho_l)

            # Safety: Clamp alpha to avoid numerical vacuum during instabilities
            self.alpha = np.clip(self.alpha, 0.0, 1.0)

            self.u_l = np.zeros_like(mass_new)
            mask = mass_new > 1e-6
            self.u_l[mask] = mom_new[mask] / mass_new[mask]

            t += dt

        # FIX FOR SHAPE ERROR:
        # visc_record is N+1 (Faces), x is N (Centers).
        # We average viscosity to centers for plotting.
        visc_centers = 0.5 * (visc_record[1:] + visc_record[:-1])

        return self.x, self.alpha, visc_centers

    def analytical_solution(self, t_current):
        alpha_exact = np.zeros_like(self.x)
        x_front = self.v0 * t_current + 0.5 * self.g * t_current**2
        for i, xi in enumerate(self.x):
            if xi < x_front:
                v = np.sqrt(self.v0**2 + 2 * self.g * xi)
                alpha_exact[i] = 1.0 - ((1.0 - self.alpha0) * self.v0) / v
            else:
                alpha_exact[i] = self.alpha0
        return self.x, alpha_exact

# --- RUNNING THE 5 CASES ---
solver = FullPressureTwoFluidSolver(n_cells=200, t_final=0.5)

# Analytical
x_ref, alpha_ref = solver.analytical_solution(0.5)

results = {}

print("1. Running 1st Order Upwind...")
results['Up1'] = solver.solve(scheme='Upwind1')

print("2. Running 2nd Order Upwind (No EVM)...")
results['Up2'] = solver.solve(scheme='Upwind2')

print("3. Running Central (No EVM)...")
results['Cen'] = solver.solve(scheme='Central')

print("4. Running Central + EVM...")
results['Cen_EVM'] = solver.solve(scheme='Central_EVM')

print("5. Running 2nd Order Upwind + EVM...")
results['Up2_EVM'] = solver.solve(scheme='Upwind2_EVM')

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Void Fraction
ax1.set_title("Water Faucet: Flux Scheme Comparison")
ax1.plot(x_ref, alpha_ref, 'k--', linewidth=2, label='Analytical')

# The Baselines (Dashed/Dotted)
ax1.plot(solver.x, results['Up1'][1], 'g--', label='1st Order Upwind')
ax1.plot(solver.x, results['Up2'][1], 'y:',  label='2nd Order Upwind (Unstable?)')
ax1.plot(solver.x, results['Cen'][1], 'm:',  label='Central (Unstable?)')

# The Stabilized Schemes (Solid)
ax1.plot(solver.x, results['Cen_EVM'][1], 'b-', linewidth=2, label='Central + EVM')
ax1.plot(solver.x, results['Up2_EVM'][1], 'r-', linewidth=2, label='2nd Order Upwind + EVM')

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Void Fraction")
ax1.set_ylim(0.0, 0.60)
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Viscosity Profile
ax2.set_title("Artificial Viscosity Activation")
ax2.plot(solver.x, results['Cen_EVM'][2], 'b-', label='Viscosity (Central Base)')
ax2.plot(solver.x, results['Up2_EVM'][2], 'r-', label='Viscosity (Upwind2 Base)')
ax2.fill_between(solver.x, 0, results['Up2_EVM'][2], color='r', alpha=0.1)

ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Viscosity (m^2/s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
