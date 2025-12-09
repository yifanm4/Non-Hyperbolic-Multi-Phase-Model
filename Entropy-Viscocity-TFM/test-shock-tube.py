import numpy as np
import matplotlib.pyplot as plt

class FullShockTubeSolver:
    def __init__(self, n_cells=50, L=100.0, t_final=0.1):
        self.N = n_cells
        self.L = L
        self.dx = L / n_cells
        self.t_final = t_final
        self.x = np.linspace(self.dx/2, L - self.dx/2, self.N)

        # Physics Constants
        self.rho_g = 1.0       # Gas Density
        self.rho_l = 1000.0    # Liquid Density
        self.g = 0.0           # Gravity
        self.P_init = 265000.0 # Pressure

        # Initial Conditions
        self.midpoint = 50.0

        # Initialize State Arrays
        self.alpha = np.zeros(self.N)
        self.u_g = np.zeros(self.N)
        self.u_l = np.zeros(self.N) # Liquid Velocity

        # History for Entropy Viscosity (alpha based)
        self.alpha_hist = [np.zeros(self.N)] * 3

    def van_leer(self, r):
        # Van Leer Slope Limiter
        return (r + np.abs(r)) / (1.0 + np.abs(r))

    def check_stability(self):
        # Safety check
        if np.any(np.isnan(self.u_g)) or np.any(np.isinf(self.u_g)): return False
        if np.any(np.isnan(self.alpha)) or np.any(np.isinf(self.alpha)): return False
        if np.max(np.abs(self.u_g)) > 1e4: return False
        return True

    def compute_entropy_viscosity(self, dt, v_field):
        # 1. Update History
        self.alpha_hist.append(self.alpha.copy())
        if len(self.alpha_hist) > 3: self.alpha_hist.pop(0)

        if len(self.alpha_hist) < 3 or dt < 1e-12:
            return np.zeros(self.N + 1)

        # 2. Residual on Void Fraction
        d_dt = (3*self.alpha_hist[-1] - 4*self.alpha_hist[-2] + self.alpha_hist[-3]) / (2*dt)

        f = self.alpha * v_field
        df_dx = np.zeros(self.N)
        df_dx[1:-1] = (f[2:] - f[:-2]) / (2*self.dx)
        df_dx[0] = (f[1] - f[0]) / self.dx
        df_dx[-1] = (f[-1] - f[-2]) / self.dx

        Residual = np.abs(d_dt + df_dx)

        # 3. Normalization
        norm = np.max(np.abs(self.alpha - np.mean(self.alpha))) + 1e-10

        # 4. Coefficients
        #c_E = 0.5
        c_E = 5.0
        nu_E = c_E * (self.dx**2) * Residual / norm

        c_max = 0.5
        #c_max = 1.0
        nu_max = c_max * self.dx * np.abs(v_field)
        nu_cell = np.minimum(nu_max, nu_E)

        # Map to Faces
        nu_face = np.zeros(self.N + 1)
        nu_face[1:-1] = 0.5 * (nu_cell[1:] + nu_cell[:-1])
        nu_face[0], nu_face[-1] = nu_cell[0], nu_cell[-1]

        return nu_face

    def reconstruct_MUSCL(self, Q):
        Q_L = np.zeros(self.N + 1)
        for i in range(1, self.N):
            if i < 2:
                Q_L[i] = Q[i-1]
            else:
                d_minus = Q[i-1] - Q[i-2]
                d_plus  = Q[i] - Q[i-1]
                if abs(d_minus) < 1e-10: r = 0.0
                else: r = d_plus / d_minus
                phi = self.van_leer(r)
                Q_L[i] = Q[i-1] + 0.5 * phi * d_minus
        Q_L[0], Q_L[-1] = Q[0], Q[-1]
        return Q_L

    def solve(self, scheme='Upwind1'):
        # --- 1. Set Initial Conditions ---
        self.alpha = np.zeros(self.N)
        self.u_g = np.zeros(self.N)
        self.u_l = np.zeros(self.N)

        for i, xi in enumerate(self.x):
            if xi < self.midpoint:
                self.alpha[i] = 0.29
                self.u_g[i]   = 65.0
                self.u_l[i]   = 1.0
            else:
                self.alpha[i] = 0.30
                self.u_g[i]   = 50.0
                self.u_l[i]   = 1.0

        self.alpha_hist = [self.alpha.copy()] * 3

        t = 0.0
        CFL = 0.1
        visc_record = np.zeros(self.N + 1)

        while t < self.t_final:
            if not self.check_stability():
                print(f"  [Warning] Simulation blew up at t={t:.4f}s with scheme {scheme}")
                break

            max_v = np.max(np.abs(self.u_g)) + 1e-6
            dt = CFL * self.dx / max_v
            if t + dt > self.t_final: dt = self.t_final - t

            # --- Viscosity (Based on Gas Velocity field) ---
            if 'EVM' in scheme:
                nu_visc = self.compute_entropy_viscosity(dt, self.u_g)
                visc_record = np.maximum(visc_record, nu_visc)
            else:
                nu_visc = np.zeros(self.N + 1)

            # --- FLUX CALCULATION ---
            # We solve for 3 equations: Gas Mass, Gas Mom, Liquid Mom
            # Liquid Mass is inferred from alpha_l = 1 - alpha_g

            # 1. Reconstructions
            if 'MUSCL' in scheme:
                # Gas
                rho_g_muscl = self.reconstruct_MUSCL(self.alpha * self.rho_g)
                ug_muscl    = self.reconstruct_MUSCL(self.u_g)
                # Liquid
                rho_l_muscl = self.reconstruct_MUSCL((1.0-self.alpha) * self.rho_l)
                ul_muscl    = self.reconstruct_MUSCL(self.u_l)

            flux_Mg = np.zeros(self.N+1) # Mass Gas
            flux_Pg = np.zeros(self.N+1) # Momentum Gas
            flux_Pl = np.zeros(self.N+1) # Momentum Liquid

            for i in range(1, self.N):
                # Neighbors
                ag_L, ag_R = self.alpha[i-1], self.alpha[i]
                al_L, al_R = 1.0 - ag_L, 1.0 - ag_R

                rhog_L, rhog_R = ag_L*self.rho_g, ag_R*self.rho_g
                rhol_L, rhol_R = al_L*self.rho_l, al_R*self.rho_l

                ug_L, ug_R = self.u_g[i-1], self.u_g[i]
                ul_L, ul_R = self.u_l[i-1], self.u_l[i]

                # --- Scheme Selection ---
                if 'Upwind1' in scheme:
                    fg_rho, fg_u = rhog_L, ug_L
                    fl_rho, fl_u = rhol_L, ul_L
                elif 'Central' in scheme:
                    fg_rho, fg_u = 0.5*(rhog_L+rhog_R), 0.5*(ug_L+ug_R)
                    fl_rho, fl_u = 0.5*(rhol_L+rhol_R), 0.5*(ul_L+ul_R)
                elif 'MUSCL' in scheme:
                    fg_rho, fg_u = rho_g_muscl[i], ug_muscl[i]
                    fl_rho, fl_u = rho_l_muscl[i], ul_muscl[i]

                # --- Convective Fluxes ---
                flux_Mg[i] = fg_rho * fg_u
                flux_Pg[i] = fg_rho * fg_u**2
                flux_Pl[i] = fl_rho * fl_u**2

                # --- Artificial Diffusion (EVM) ---
                # Applied to all conserved variables
                flux_Mg[i] -= nu_visc[i] * (rhog_R - rhog_L) / self.dx
                flux_Pg[i] -= nu_visc[i] * (rhog_R*ug_R - rhog_L*ug_L) / self.dx
                flux_Pl[i] -= nu_visc[i] * (rhol_R*ul_R - rhol_L*ul_L) / self.dx

            # Boundaries (Transmissive)
            flux_Mg[0], flux_Mg[-1] = flux_Mg[1], flux_Mg[-2]
            flux_Pg[0], flux_Pg[-1] = flux_Pg[1], flux_Pg[-2]
            flux_Pl[0], flux_Pl[-1] = flux_Pl[1], flux_Pl[-2]

            # --- UPDATES ---
            # 1. Update Gas Mass
            div_Mg = (flux_Mg[1:] - flux_Mg[:-1]) / self.dx
            mass_g_new = (self.alpha * self.rho_g) - dt * div_Mg

            # Recover Alpha
            self.alpha = mass_g_new / self.rho_g
            self.alpha = np.clip(self.alpha, 0.0, 1.0)

            # 2. Update Gas Momentum
            div_Pg = (flux_Pg[1:] - flux_Pg[:-1]) / self.dx
            mom_g_new = (mass_g_new * self.u_g) - dt * div_Pg # No forces

            # Recover Gas Velocity
            self.u_g = np.zeros_like(mass_g_new)
            mask_g = mass_g_new > 1e-9
            self.u_g[mask_g] = mom_g_new[mask_g] / mass_g_new[mask_g]

            # 3. Update Liquid Momentum
            # Note: Liquid Mass is derived from (1-alpha)
            mass_l_new = (1.0 - self.alpha) * self.rho_l
            div_Pl = (flux_Pl[1:] - flux_Pl[:-1]) / self.dx

            # Current Mom
            mom_l_old = (1.0 - self.alpha) * self.rho_l * self.u_l # Use alpha_old? Approximated.
            # Ideally we update M_l explicitly too, but for advection test this is okay.
            mom_l_new = mom_l_old - dt * div_Pl

            # Recover Liquid Velocity
            self.u_l = np.zeros_like(mass_l_new)
            mask_l = mass_l_new > 1e-9
            self.u_l[mask_l] = mom_l_new[mask_l] / mass_l_new[mask_l]

            t += dt

        visc_centers = 0.5 * (visc_record[1:] + visc_record[:-1])
        return self.x, self.alpha, self.u_l, visc_centers

# --- RUNNING CASES ---
NCells = 1000
solver = FullShockTubeSolver(n_cells=NCells, t_final=0.1)

results = {}
print("1. Running 1st Order Upwind...")
results['Up1'] = solver.solve(scheme='Upwind1')

print("2. Running Central (Unstable)...")
results['Cen'] = solver.solve(scheme='Central')

print("3. Running MUSCL...")
results['MUSCL'] = solver.solve(scheme='MUSCL')

print("4. Running Central + EVM...")
results['Cen_EVM'] = solver.solve(scheme='Central_EVM')

print("5. Running MUSCL + EVM...")
results['MUSCL_EVM'] = solver.solve(scheme='MUSCL_EVM')

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Gas Void Fraction
ax1.set_title("Shock Tube: Gas Void Fraction")
ax1.plot(solver.x, np.zeros_like(solver.x)+0.29, 'k:', alpha=0.3)
ax1.plot(solver.x, np.zeros_like(solver.x)+0.30, 'k:', alpha=0.3)

ax1.plot(solver.x, results['Up1'][1], 'g--', label='Upwind1')
#ax1.plot(solver.x, results['Cen'][1], 'm:', label='Central')
ax1.plot(solver.x, results['MUSCL'][1], 'y-', label='MUSCL')
#ax1.plot(solver.x, results['Cen_EVM'][1], 'b-', linewidth=2, label='Central + EVM')
ax1.plot(solver.x, results['MUSCL_EVM'][1], 'r-', linewidth=2, label='MUSCL + EVM')

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Void Fraction (-)")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Liquid Velocity (Checking Consistency)
ax2.set_title("Liquid Velocity Stability")
ax2.plot(solver.x, results['Up1'][2], 'g--', label='Upwind1')
ax2.plot(solver.x, results['MUSCL_EVM'][2], 'r-', label='MUSCL + EVM')
# It should stay near 1.0 if stable
ax2.set_ylim(0.95, 1.05)
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Liquid Velocity (m/s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"shock_tube_results_ncells_{NCells}.png", dpi=300)
plt.show()
