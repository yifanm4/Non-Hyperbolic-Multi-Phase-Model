import numpy as np
import matplotlib.pyplot as plt

# ==========================================
#        PHYSICS & FLUX FUNCTIONS
# ==========================================

def primitives_from_conservatives(beta, omega, rho_g, rho_l, U_mix, eps=1e-8):
    beta = np.asarray(beta)
    omega = np.asarray(omega)
    drho = rho_g - rho_l
    if abs(drho) < eps: drho = eps

    alpha_g = (beta - rho_l) / drho
    alpha_l = 1.0 - alpha_g

    # Robust clipping
    alpha_g = np.clip(alpha_g, 1e-5, 1.0 - 1e-5)
    alpha_l = np.clip(alpha_l, 1e-5, 1.0 - 1e-5)

    beta_eff = alpha_g * rho_g + alpha_l * rho_l

    num_g = U_mix * rho_l + (beta_eff - rho_l) * omega / drho
    num_l = U_mix * rho_g + (beta_eff - rho_g) * omega / drho

    u_g = num_g / beta_eff
    u_l = num_l / beta_eff

    return alpha_g, alpha_l, u_g, u_l

def get_flux_and_source(beta, omega, rho_g, rho_l, U_mix, g=9.81):
    alpha_g, alpha_l, u_g, u_l = primitives_from_conservatives(
        beta, omega, rho_g, rho_l, U_mix
    )
    F1 = rho_g * alpha_g * u_g + rho_l * alpha_l * u_l
    F2 = 0.5 * (rho_g * u_g**2 - rho_l * u_l**2)

    S1 = np.zeros_like(beta)
    with np.errstate(divide='ignore'):
        S2 = (rho_g * g / alpha_g) - (rho_l * g / alpha_l)

    return np.array([F1, F2]), np.array([S1, S2])

def characteristic_speed(beta, omega, rho_g, rho_l, U_mix, eps=1e-8):
    beta_safe = np.clip(beta, eps, None)
    drho = rho_g - rho_l
    lam_num = rho_g * rho_l * (U_mix * (rho_l - rho_g) + beta_safe * omega)
    lam_den = (beta_safe ** 2) * (rho_l - rho_g)
    return np.abs(lam_num / lam_den)

def analytical_solution(x_eval, t, alphag_ini, ul_ini, g):
    x_front = ul_ini * t + 0.5 * g * t**2
    v_steady = np.sqrt(ul_ini**2 + 2 * g * x_eval)
    al_steady = (1 - alphag_ini) * ul_ini / v_steady
    ag_steady = 1.0 - al_steady
    ag_initial = alphag_ini
    return np.where(x_eval < x_front, ag_steady, ag_initial)

# ==========================================
#          LIMITERS & FIXES
# ==========================================

def van_leer_limiter(dq1, dq2):
    eps = 1e-12
    abs_dq1 = np.abs(dq1)
    abs_dq2 = np.abs(dq2)
    den = abs_dq1 + abs_dq2 + eps
    return (dq1 * abs_dq2 + abs_dq1 * dq2) / den

def harten_entropy_fix(lam, delta=1.0):
    """
    Harten's Entropy Fix:
    Prevents wave speed from approaching zero too closely.
    If |lambda| < delta, replace with (lambda^2 + delta^2) / (2*delta).
    """
    lam_abs = np.abs(lam)
    mask = lam_abs < delta
    # Where mask is true, modify lambda
    lam_fixed = lam_abs.copy()
    lam_fixed[mask] = (lam_abs[mask]**2 + delta**2) / (2 * delta)
    return lam_fixed

def compute_entropy_viscosity(beta, F_beta, dx, max_vel):
    """
    Simple Entropy Viscosity Model (EVM).
    Uses the residual of the mass equation as the 'entropy' indicator.
    """
    # 1. Compute Entropy Residual D = d(beta)/dt + d(F)/dx
    # Approximate d(beta)/dt ~ (beta - beta_old)/dt...
    # Or simpler: Deviation from mean

    # Standard simplified EVM for scalar conservation:
    # nu_E = C_E * h^2 * |Residual| / |Normalization|

    # Calculate local jump in beta
    dbeta = np.abs(beta[1:] - beta[:-1]) # length N-1

    # Max viscosity (First order scaling)
    nu_max = 0.5 * max_vel * dx

    # Entropy viscosity (Local scaling)
    # Using simple gradient based switch for robustness in this specific model
    # nu_E ~ dx * |jump|
    nu_E = 0.5 * dbeta * dx

    # Limit
    nu = np.minimum(nu_max, nu_E)
    return nu

def path_jump_G(bL, oL, bR, oR, rho_g, rho_l):
    """
    Placeholder path-integral jump for nonconservative parts.
    For this model the governing equations are handled as conservative+source,
    so we return zeros; kept for extensibility to true path terms.
    """
    return np.zeros(2)

# ==========================================
#            MAIN SOLVER CLASS
# ==========================================

class MultiSolver:
    def __init__(self, ncells=200):
        self.rhog = 1.0
        self.rhol = 1000.0
        self.L = 12.0
        self.ncells = ncells
        self.dx = self.L / ncells
        self.nghost = 2
        self.nx_tot = ncells + 2 * self.nghost

        self.ul_ini = 10.0
        self.ug_ini = 0.0
        self.alphag_ini = 0.2
        self.g = 9.81

        self.beta_ini = self.alphag_ini * self.rhog + (1 - self.alphag_ini) * self.rhol
        self.omega_ini = self.rhog * self.ug_ini - self.rhol * self.ul_ini
        self.U_mix = self.alphag_ini * self.ug_ini + (1 - self.alphag_ini) * self.ul_ini

        # Arrays
        self.beta = np.zeros(self.nx_tot)
        self.omega = np.zeros(self.nx_tot)

    def reset(self):
        self.beta[:] = self.beta_ini
        self.omega[:] = self.omega_ini

    def apply_bcs(self):
        # Dirichlet Inlet
        self.beta[0:self.nghost] = self.beta_ini
        self.omega[0:self.nghost] = self.omega_ini
        # Neumann Outlet
        self.beta[-self.nghost:] = self.beta[-self.nghost-1]
        self.omega[-self.nghost:] = self.omega[-self.nghost-1]

    def solve(self, method, tEnd, dt):
        self.reset()
        nt = int(tEnd / dt)
        print(f"Running {method}...")

        for _ in range(nt):
            self.apply_bcs()
            if method == "1st_Upwind":
                self.step_rusanov(dt)
            elif method == "1st_Central_LxF":
                self.step_lxf(dt)
            elif method == "1st_Central_EVM":
                self.step_central_evm(dt)
            elif method == "2nd_Upwind_VanLeer":
                self.step_muscl(dt, use_harten=False)
            elif method == "2nd_Upwind_Harten":
                self.step_muscl(dt, use_harten=True)
            elif method == "Central_Upwind":
                self.step_central_upwind(dt, use_path=False)
            elif method == "Path_Central_Upwind":
                self.step_central_upwind(dt, use_path=True, kappa=1.0)

        start, end = self.nghost, self.nghost + self.ncells
        ag, _, _, _ = primitives_from_conservatives(
            self.beta[start:end], self.omega[start:end],
            self.rhog, self.rhol, self.U_mix
        )
        return ag

    # --- 1. First Order Upwind (Rusanov) ---
    def step_rusanov(self, dt):
        idx = slice(self.nghost, self.nghost + self.ncells)
        F_faces = np.zeros((2, self.nx_tot))

        for i in range(self.nghost, self.ncells + self.nghost + 1):
            bL, oL = self.beta[i-1], self.omega[i-1]
            bR, oR = self.beta[i],   self.omega[i]

            FL, _ = get_flux_and_source(bL, oL, self.rhog, self.rhol, self.U_mix, self.g)
            FR, _ = get_flux_and_source(bR, oR, self.rhog, self.rhol, self.U_mix, self.g)

            lam = max(characteristic_speed(bL, oL, self.rhog, self.rhol, self.U_mix),
                      characteristic_speed(bR, oR, self.rhog, self.rhol, self.U_mix))

            F_faces[:, i] = 0.5*(FL + FR) - 0.5*lam*np.array([bR-bL, oR-oL])

        self.update(dt, F_faces, idx)

    # --- 2. First Order Central (Lax-Friedrichs) ---
    def step_lxf(self, dt):
        idx = slice(self.nghost, self.nghost + self.ncells)
        F_faces = np.zeros((2, self.nx_tot))

        # LxF Flux: 0.5(FL + FR) - 0.5 * (dx/dt) * (UR - UL)
        grid_speed = self.dx / dt

        for i in range(self.nghost, self.ncells + self.nghost + 1):
            bL, oL = self.beta[i-1], self.omega[i-1]
            bR, oR = self.beta[i],   self.omega[i]
            FL, _ = get_flux_and_source(bL, oL, self.rhog, self.rhol, self.U_mix, self.g)
            FR, _ = get_flux_and_source(bR, oR, self.rhog, self.rhol, self.U_mix, self.g)

            F_faces[:, i] = 0.5*(FL + FR) - 0.5 * grid_speed * np.array([bR-bL, oR-oL])

        self.update(dt, F_faces, idx)

    # --- 3. First Order Central + Entropy Viscosity ---
    def step_central_evm(self, dt):
        idx = slice(self.nghost, self.nghost + self.ncells)
        F_faces = np.zeros((2, self.nx_tot))

        # Compute Entropy Viscosity field
        # We use a simplified jump-based sensor at interfaces

        for i in range(self.nghost, self.ncells + self.nghost + 1):
            bL, oL = self.beta[i-1], self.omega[i-1]
            bR, oR = self.beta[i],   self.omega[i]

            FL, _ = get_flux_and_source(bL, oL, self.rhog, self.rhol, self.U_mix, self.g)
            FR, _ = get_flux_and_source(bR, oR, self.rhog, self.rhol, self.U_mix, self.g)

            # Pure Central Flux (Unstable!)
            F_central = 0.5 * (FL + FR)

            # Entropy Viscosity Coefficient
            # nu_max = 0.5 * max_wave_speed * dx
            lam = max(characteristic_speed(bL, oL, self.rhog, self.rhol, self.U_mix),
                      characteristic_speed(bR, oR, self.rhog, self.rhol, self.U_mix))
            nu_max = 0.5 * lam * self.dx

            # Entropy Sensor: Normalized jump in beta
            # Ideally this uses the residual, but jump is a good proxy for shocks
            jump = abs(bR - bL) / (abs(0.5*(bR+bL)) + 1e-6)

            # Tune C_E approx 0.5 to 1.0
            nu_E = 1.0 * self.dx * lam * jump

            nu = min(nu_max, nu_E)

            # Diffusive Flux
            F_visc = nu * np.array([bR-bL, oR-oL]) / self.dx

            F_faces[:, i] = F_central - F_visc

        self.update(dt, F_faces, idx)

    # --- 4 & 5. MUSCL (with Van Leer) +/- Harten Fix ---
    def step_muscl(self, dt, use_harten=False):
        # Slopes
        d_beta = self.beta[1:] - self.beta[:-1]
        d_omega = self.omega[1:] - self.omega[:-1]
        slope_b = np.zeros_like(self.beta)
        slope_o = np.zeros_like(self.omega)
        for i in range(1, self.nx_tot-1):
            slope_b[i] = van_leer_limiter(d_beta[i-1], d_beta[i])
            slope_o[i] = van_leer_limiter(d_omega[i-1], d_omega[i])

        # Half Step (Predictor)
        F_c, S_c = get_flux_and_source(self.beta, self.omega, self.rhog, self.rhol, self.U_mix, self.g)
        bL = self.beta - 0.5*slope_b
        bR = self.beta + 0.5*slope_b
        oL = self.omega - 0.5*slope_o
        oR = self.omega + 0.5*slope_o

        FLv, _ = get_flux_and_source(bL, oL, self.rhog, self.rhol, self.U_mix, self.g)
        FRv, _ = get_flux_and_source(bR, oR, self.rhog, self.rhol, self.U_mix, self.g)
        dF = FRv - FLv
        k = dt / (2*self.dx)

        b_bound_R = bR - k*dF[0] + 0.5*dt*S_c[0]
        b_bound_L = bL - k*dF[0] + 0.5*dt*S_c[0]
        o_bound_R = oR - k*dF[1] + 0.5*dt*S_c[1]
        o_bound_L = oL - k*dF[1] + 0.5*dt*S_c[1]

        # Riemann (Corrector)
        F_faces = np.zeros((2, self.nx_tot))
        for i in range(self.nghost, self.ncells+self.nghost+1):
            bL_s, oL_s = b_bound_R[i-1], o_bound_R[i-1]
            bR_s, oR_s = b_bound_L[i],   o_bound_L[i]

            FL, _ = get_flux_and_source(bL_s, oL_s, self.rhog, self.rhol, self.U_mix, self.g)
            FR, _ = get_flux_and_source(bR_s, oR_s, self.rhog, self.rhol, self.U_mix, self.g)

            lam_L = characteristic_speed(bL_s, oL_s, self.rhog, self.rhol, self.U_mix)
            lam_R = characteristic_speed(bR_s, oR_s, self.rhog, self.rhol, self.U_mix)
            a = max(lam_L, lam_R)

            if use_harten:
                # Apply Harten fix to wave speed 'a'
                # Prevents vanishing dissipation at sonic points
                delta = 0.2 * abs(self.U_mix) # heuristic delta
                if a < delta:
                    a = (a**2 + delta**2)/(2*delta)

            F_faces[:, i] = 0.5*(FL + FR) - 0.5*a*np.array([bR_s-bL_s, oR_s-oL_s])

        self.update(dt, F_faces, slice(self.nghost, self.nghost+self.ncells))

    # --- 6. Central-Upwind (Kurganov-type) ---
    def cu_face_flux(self, bL, oL, bR, oR, use_path=False, kappa=1.0):
        UL = np.array([bL, oL])
        UR = np.array([bR, oR])

        FL, _ = get_flux_and_source(bL, oL, self.rhog, self.rhol, self.U_mix, self.g)
        FR, _ = get_flux_and_source(bR, oR, self.rhog, self.rhol, self.U_mix, self.g)

        lam_L = characteristic_speed(bL, oL, self.rhog, self.rhol, self.U_mix)
        lam_R = characteristic_speed(bR, oR, self.rhog, self.rhol, self.U_mix)
        a = max(lam_L, lam_R)
        a_plus = a
        a_minus = -a
        denom = a_plus - a_minus + 1e-12

        H = (a_plus * FL + a_minus * FR) / denom + (a_plus * a_minus) / denom * (UR - UL)

        if use_path:
            G = path_jump_G(bL, oL, bR, oR, self.rhog, self.rhol)
            H = H - kappa * (a_plus * a_minus) / denom * G
        return H

    def step_central_upwind(self, dt, use_path=False, kappa=1.0):
        idx = slice(self.nghost, self.nghost + self.ncells)
        F_faces = np.zeros((2, self.nx_tot))

        for i in range(self.nghost, self.ncells + self.nghost + 1):
            bL, oL = self.beta[i-1], self.omega[i-1]
            bR, oR = self.beta[i],   self.omega[i]
            F_faces[:, i] = self.cu_face_flux(bL, oL, bR, oR, use_path=use_path, kappa=kappa)

        self.update(dt, F_faces, idx)

    def update(self, dt, F_faces, idx):
        # Helper to apply flux diff + source
        F_R = F_faces[:, self.nghost+1 : self.ncells + self.nghost + 1]
        F_L = F_faces[:, self.nghost   : self.ncells + self.nghost]
        _, S = get_flux_and_source(self.beta[idx], self.omega[idx], self.rhog, self.rhol, self.U_mix, self.g)

        self.beta[idx]  += dt * (-(F_R[0] - F_L[0])/self.dx + S[0])
        self.omega[idx] += dt * (-(F_R[1] - F_L[1])/self.dx + S[1])


# ==========================================
#                  RUNNER
# ==========================================

def main():
    ncells = 100
    tEnd = 0.5
    dt = 5e-5 # Small timestep for LxF stability

    sim = MultiSolver(ncells)

    results = {}
    solvers = [
        "1st_Upwind",
        "1st_Central_LxF",
        "1st_Central_EVM",
        "2nd_Upwind_VanLeer",
        "2nd_Upwind_Harten",
        "Central_Upwind",
        "Path_Central_Upwind"
    ]

    for s in solvers:
        results[s] = sim.solve(s, tEnd, dt)

    # Analytical
    x = np.linspace(sim.dx/2, sim.L - sim.dx/2, ncells)
    res_exact = analytical_solution(x, tEnd, sim.alphag_ini, sim.ul_ini, sim.g)

    # Plot
    plt.figure(figsize=(12, 8))



    # Reference
    plt.plot(x, res_exact, 'k--', label='Analytical', linewidth=2)

    # 1st Order Family
    plt.plot(x, results["1st_Upwind"], label='1st Upwind (Rusanov)', alpha=0.6)
    plt.plot(x, results["1st_Central_LxF"], label='1st Central (LxF)', linestyle='dotted')
    plt.plot(x, results["1st_Central_EVM"], label='1st Central + EVM', linestyle='-.')

    # 2nd Order Family
    plt.plot(x, results["2nd_Upwind_VanLeer"], label='2nd Upwind (MUSCL)', linewidth=2)
    plt.plot(x, results["2nd_Upwind_Harten"], label='2nd Upwind + Harten Fix', linestyle='--')

    plt.title(f"Comparison of 5 Solvers: Water Faucet Case (t={tEnd}s)")
    plt.xlabel("Position (m)")
    plt.ylabel("Gas Void Fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.ylim(0.18, 0.45)
    plt.tight_layout()
    plt.savefig("5_solver_comparison.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
