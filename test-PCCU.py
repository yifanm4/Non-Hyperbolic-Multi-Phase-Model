import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.linalg import eig

# --- CONFIGURATION ---
L_PIPE = 12.0
N_CELLS = 100
T_END = 0.5
CFL = 0.4
GRAVITY = 9.81
VEL_CLIP = 200.0
RHO_MIN = 1e-6
RHO_MAX = 1e6
P_MAX = 1e9

# --- EOS PARAMETERS (Stiffened Gas) ---
# Liquid (Water-like)
RHO0_L = 1000.0
C_L = 300.0
P0_L = 1.0e5
# Gas (Air-like)
RHO0_G = 1.0
C_G = 300.0
P0_G = 1.0e5

class TwoFluidSolver:
    def __init__(self, method='PCCU', n_cells=N_CELLS):
        self.method = method
        self.nx = n_cells
        self.dx = L_PIPE / n_cells
        self.nghost = 2

        self.x = np.linspace(self.dx/2, L_PIPE - self.dx/2, self.nx)

        # U = [m_l, m_g, mom_l, mom_g]
        self.U = np.zeros((self.nx + 2*self.nghost, 4))

        # IC: Water Faucet
        self.al_in = 0.8
        self.ul_in = 10.0
        self.ug_in = 0.0
        self.ag_in = 1.0 - self.al_in

        self.init_domain()

    # --- THERMODYNAMICS (FIXED) ---
    def get_p_from_rho(self, rho, phase):
        # Use np.maximum to handle both scalars and arrays safely
        rho = np.clip(rho, RHO_MIN, RHO_MAX)
        if phase == 'l': return C_L**2 * (rho - RHO0_L) + P0_L
        else:            return C_G**2 * (rho - RHO0_G) + P0_G

    def get_rho_from_p(self, p, phase):
        # Use np.maximum to handle both scalars and arrays safely
        p = np.clip(p, 100.0, P_MAX)
        if phase == 'l': return (p - P0_L)/(C_L**2) + RHO0_L
        else:            return (p - P0_G)/(C_G**2) + RHO0_G

    def recover_primitives(self, U_slice):
        """Recover [alpha_l, P, u_l, u_g] from Conservative U."""
        m_l = U_slice[:, 0]
        m_g = U_slice[:, 1]
        mom_l = U_slice[:, 2]
        mom_g = U_slice[:, 3]

        N = len(m_l)
        alpha_l = np.zeros(N)
        P = np.zeros(N)

        # Robust Newton Loop
        for i in range(N):
            ml_i = max(m_l[i], 1e-8)
            mg_i = max(m_g[i], 1e-8)

            def p_diff(al):
                al_safe = max(1e-6, min(al, 1.0-1e-6))
                rho_l = ml_i / al_safe
                rho_g = mg_i / (1.0 - al_safe)
                pl = self.get_p_from_rho(rho_l, 'l')
                pg = self.get_p_from_rho(rho_g, 'g')
                return pl - pg

            try:
                al_sol = newton(p_diff, 0.8, maxiter=20, tol=1e-3)
            except (RuntimeError, ValueError):
                # Fallback check
                v0 = p_diff(0.01)
                v1 = p_diff(0.99)
                if np.isfinite(v0) and np.isfinite(v1) and np.sign(v0) != np.sign(v1):
                    from scipy.optimize import brentq
                    al_sol = brentq(p_diff, 0.01, 0.99)
                else:
                    al_sol = 0.8

            if not np.isfinite(al_sol):
                al_sol = 0.8
            al_sol = max(1e-6, min(al_sol, 1.0-1e-6))
            alpha_l[i] = al_sol

            # Recompute Pressure
            rho_l_final = ml_i / al_sol
            P[i] = self.get_p_from_rho(rho_l_final, 'l')

        u_l = np.zeros_like(mom_l)
        u_g = np.zeros_like(mom_g)
        idx_l = m_l > 1e-9; u_l[idx_l] = mom_l[idx_l]/m_l[idx_l]
        idx_g = m_g > 1e-9; u_g[idx_g] = mom_g[idx_g]/m_g[idx_g]

        return alpha_l, P, u_l, u_g

    # --- INITIALIZATION & BCs ---
    def init_domain(self):
        rho_l = self.get_rho_from_p(1e5, 'l')
        rho_g = self.get_rho_from_p(1e5, 'g')

        idx_s = self.nghost; idx_e = self.nx + self.nghost
        self.U[idx_s:idx_e, 0] = self.al_in * rho_l
        self.U[idx_s:idx_e, 1] = self.ag_in * rho_g
        self.U[idx_s:idx_e, 2] = self.al_in * rho_l * self.ul_in
        self.U[idx_s:idx_e, 3] = self.ag_in * rho_g * self.ug_in
        self.apply_bc()

    def enforce_bounds(self):
        """Keep masses positive and velocities bounded to avoid overflow."""
        self.U[:, 0] = np.clip(self.U[:, 0], RHO_MIN, RHO_MAX)
        self.U[:, 1] = np.clip(self.U[:, 1], RHO_MIN, RHO_MAX)

        u_l = np.zeros_like(self.U[:, 2])
        u_g = np.zeros_like(self.U[:, 3])
        mask_l = self.U[:, 0] > 0
        mask_g = self.U[:, 1] > 0
        u_l[mask_l] = self.U[:, 2][mask_l] / self.U[:, 0][mask_l]
        u_g[mask_g] = self.U[:, 3][mask_g] / self.U[:, 1][mask_g]
        u_l = np.clip(u_l, -VEL_CLIP, VEL_CLIP)
        u_g = np.clip(u_g, -VEL_CLIP, VEL_CLIP)
        self.U[:, 2] = self.U[:, 0] * u_l
        self.U[:, 3] = self.U[:, 1] * u_g

    def apply_bc(self):
        rho_l = self.get_rho_from_p(1e5, 'l'); rho_g = self.get_rho_from_p(1e5, 'g')
        U_in = np.array([self.al_in*rho_l, self.ag_in*rho_g,
                         self.al_in*rho_l*self.ul_in, self.ag_in*rho_g*self.ug_in])
        for i in range(self.nghost): self.U[i] = U_in

        for i in range(self.nghost):
            self.U[self.nx+self.nghost+i] = self.U[self.nx+self.nghost-1]

    # --- FLUX HELPERS ---
    def get_flux_vec(self, U_slice):
        al, P, ul, ug = self.recover_primitives(U_slice)
        ag = 1.0 - al
        m_l, m_g = U_slice[:,0], U_slice[:,1]
        mom_l, mom_g = U_slice[:,2], U_slice[:,3]

        F = np.zeros_like(U_slice)
        F[:,0] = mom_l
        F[:,1] = mom_g
        F[:,2] = mom_l * ul + al * P
        F[:,3] = mom_g * ug + ag * P
        return F, al, P, ul, ug

    # --- SOLVER METHODS ---
    def solve_roe(self, U_L, U_R):
        U_avg = 0.5 * (U_L + U_R)
        eps = 1e-6
        A = np.zeros((4, 4))

        F_base, al_b, P_b, _, _ = self.get_flux_vec(U_avg.reshape(1,4))
        F_base = F_base[0]

        for i in range(4):
            U_p = U_avg.copy(); U_p[i] += eps
            F_p, al_p, _, _, _ = self.get_flux_vec(U_p.reshape(1,4))
            F_p = F_p[0]
            col = (F_p - F_base)/eps
            dal_dU = (al_p[0] - al_b[0])/eps
            col[2] -= P_b[0] * dal_dU
            col[3] -= P_b[0] * (-dal_dU)
            A[:, i] = col

        if not np.isfinite(A).all():
            Abs_A = np.eye(4) * 1000.0
        else:
            try:
                evals, R = eig(A)
                evals = np.real(evals); R = np.real(R)

                # Robust Entropy Fix
                max_wave = np.max(np.abs(evals))
                delta = 0.2 * (max_wave + 1.0)
                abs_evals = np.where(np.abs(evals)<delta, (evals**2+delta**2)/(2*delta), np.abs(evals))

                if np.abs(np.linalg.det(R)) < 1e-10: raise ValueError

                Abs_A = R @ np.diag(abs_evals) @ np.linalg.inv(R)
            except:
                # Fallback
                Abs_A = np.eye(4) * 1000.0

        F_L_phys, _, _, _, _ = self.get_flux_vec(U_L.reshape(1,4))
        F_R_phys, _, _, _, _ = self.get_flux_vec(U_R.reshape(1,4))
        return 0.5*(F_L_phys[0] + F_R_phys[0]) - 0.5 * Abs_A @ (U_R - U_L)

    def solve_hllc(self, U_L, U_R):
        al_L, P_L, ul_L, ug_L = self.reconstruct_one(U_L)
        al_R, P_R, ul_R, ug_R = self.reconstruct_one(U_R)
        F_L_val, _, _, _, _ = self.get_flux_vec(U_L.reshape(1,4))
        F_R_val, _, _, _, _ = self.get_flux_vec(U_R.reshape(1,4))
        F_L = F_L_val[0]; F_R = F_R_val[0]

        # Use np.minimum/maximum for scalar safety
        S_L = np.minimum(ul_L - C_L, ug_L - C_G)
        S_R = np.maximum(ul_R + C_L, ug_R + C_G)

        rho_L = U_L[0]+U_L[1]; rho_R = U_R[0]+U_R[1]
        u_L = (U_L[2]+U_L[3])/max(rho_L, 1e-6)
        u_R = (U_R[2]+U_R[3])/max(rho_R, 1e-6)

        den = rho_L*(S_L - u_L) - rho_R*(S_R - u_R)
        if abs(den) < 1e-8: S_star = 0.5*(u_L + u_R)
        else:
            num = P_R - P_L + rho_L*u_L*(S_L - u_L) - rho_R*u_R*(S_R - u_R)
            S_star = num / den

        if 0 <= S_L: return F_L
        elif S_L < 0 <= S_star:
            return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L + 1e-10)
        elif S_star < 0 <= S_R:
            return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L + 1e-10)
        else: return F_R

    def reconstruct_one(self, U):
        al, P, ul, ug = self.recover_primitives(U.reshape(1,4))
        return al[0], P[0], ul[0], ug[0]

    def step(self, dt):
        self.apply_bc()

        if self.method == 'PCCU':
            U_c = self.U[1:-1]; U_l = self.U[:-2]; U_r = self.U[2:]
            dp = U_r - U_c; dm = U_c - U_l
            slope = 0.5*(np.sign(dp)+np.sign(dm))*np.minimum(np.abs(dp), np.abs(dm))

            U_L = self.U + 0.5*np.pad(slope, ((1,1),(0,0)))
            U_R = self.U - 0.5*np.pad(slope, ((1,1),(0,0)))

            idx_L = self.nghost - 1; idx_R = self.nx + self.nghost - 1
            U_L_int = U_L[idx_L:idx_R+1]; U_R_int = U_R[idx_L+1:idx_R+2]

            F_L, al_L, P_L, ul_L, ug_L = self.get_flux_vec(U_L_int)
            F_R, al_R, P_R, ul_R, ug_R = self.get_flux_vec(U_R_int)

            S_L = np.maximum(np.abs(ul_L)+C_L, np.abs(ug_L)+C_G)
            S_R = np.maximum(np.abs(ul_R)+C_L, np.abs(ug_R)+C_G)
            a = np.maximum(S_L, S_R)

            denom = 2*a + 1e-10
            H = (a[:,None]*F_L + a[:,None]*F_R)/denom[:,None] - (a[:,None]**2)/denom[:,None]*(U_R_int - U_L_int)

            P_avg = 0.5*(P_L + P_R); d_al = al_R - al_L
            B = np.zeros_like(F_L)
            B[:,2] = P_avg * d_al; B[:,3] = P_avg * (-d_al)

            dF = (H[1:] - H[:-1]) / self.dx
            S_PC = (0.5*B[1:] + 0.5*B[:-1]) / self.dx

            U_in = self.U[self.nghost:self.nx+self.nghost]
            al_in, P_in, _, _ = self.recover_primitives(U_in)

            # Use fixed EOS functions that handle arrays
            rho_l = self.get_rho_from_p(P_in, 'l')
            rho_g = self.get_rho_from_p(P_in, 'g')

            S_grav = np.zeros_like(U_in)
            S_grav[:,2] = al_in * rho_l * GRAVITY; S_grav[:,3] = (1-al_in) * rho_g * GRAVITY

            self.U[self.nghost:self.nx+self.nghost] += dt * (-dF + S_PC + S_grav)

        else:
            Fluxes = np.zeros((self.nx+1, 4))
            for i in range(self.nx+1):
                U_L = self.U[self.nghost+i-1]; U_R = self.U[self.nghost+i]
                if self.method == 'Roe': Fluxes[i] = self.solve_roe(U_L, U_R)
                elif self.method == 'HLLC': Fluxes[i] = self.solve_hllc(U_L, U_R)

            for i in range(self.nx):
                idx = self.nghost + i
                dF = (Fluxes[i+1] - Fluxes[i]) / self.dx

                U_cell = self.U[idx]
                al, P, _, _ = self.recover_primitives(U_cell.reshape(1,4))

                # Use fixed EOS functions that handle scalars (numpy aware)
                rho_l = self.get_rho_from_p(P[0], 'l')
                rho_g = self.get_rho_from_p(P[0], 'g')

                S_grav = np.zeros(4)
                S_grav[2] = al[0] * rho_l * GRAVITY
                S_grav[3] = (1-al[0]) * rho_g * GRAVITY

                U_r = self.U[idx+1]; U_l = self.U[idx-1]
                al_r, _, _, _ = self.recover_primitives(U_r.reshape(1,4))
                al_l, _, _, _ = self.recover_primitives(U_l.reshape(1,4))
                dal_dx = (al_r[0] - al_l[0])/(2*self.dx)
                S_nc = np.zeros(4); S_nc[2] = P[0] * dal_dx; S_nc[3] = P[0] * (-dal_dx)

                self.U[idx] += dt * (-dF + S_grav + S_nc)
        self.enforce_bounds()

    def run(self):
        t = 0
        while t < T_END:
            dt = CFL * self.dx / C_L
            if t+dt > T_END: dt = T_END - t
            self.step(dt)
            t += dt
            print(f"{self.method} t={t:.2f}", end='\r')

        U_in = self.U[self.nghost:self.nx+self.nghost]
        al, P, ul, ug = self.recover_primitives(U_in)
        return self.x, 1.0-al, ul, ug, P

# --- ANALYTICAL SOLUTION ---
def get_analytical():
    x = np.linspace(0, L_PIPE, 200)
    v0 = 10.0; a0 = 0.8; P0 = 1e5
    x_front = v0*T_END + 0.5*GRAVITY*T_END**2
    v = np.sqrt(v0**2 + 2*GRAVITY*x)
    al = np.where(x < x_front, a0*v0/v, a0)
    ag = 1.0 - al
    ul = v
    ug = np.zeros_like(x)
    P = np.ones_like(x) * P0
    return x, ag, ul, ug, P

# --- EXECUTION & PLOTTING ---
if __name__ == "__main__":
    results = {}
    for scheme in ['PCCU', 'Roe', 'HLLC']:
        s = TwoFluidSolver(scheme)
        results[scheme] = s.run()

    x_ref, ag_ref, ul_ref, ug_ref, P_ref = get_analytical()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 1. Alpha Gas
    ax = axes[0,0]
    ax.plot(x_ref, ag_ref, 'k-', linewidth=2, label='Analytical')
    ax.plot(results['PCCU'][0], results['PCCU'][1], 'r-', label='PCCU')
    ax.plot(results['Roe'][0], results['Roe'][1], 'g--', label='Roe')
    ax.plot(results['HLLC'][0], results['HLLC'][1], 'b:', label='HLLC')
    ax.set_title('Gas Volume Fraction (alpha_g)')
    ax.set_ylim(0, 1)
    ax.legend()

    # 2. Liquid Velocity
    ax = axes[0,1]
    ax.plot(x_ref, ul_ref, 'k-', linewidth=2)
    ax.plot(results['PCCU'][0], results['PCCU'][2], 'r-')
    ax.plot(results['Roe'][0], results['Roe'][2], 'g--')
    ax.plot(results['HLLC'][0], results['HLLC'][2], 'b:')
    ax.set_title('Liquid Velocity (u_l)')

    # 3. Gas Velocity
    ax = axes[1,0]
    ax.plot(x_ref, ug_ref, 'k-', linewidth=2)
    ax.plot(results['PCCU'][0], results['PCCU'][3], 'r-')
    ax.plot(results['Roe'][0], results['Roe'][3], 'g--')
    ax.plot(results['HLLC'][0], results['HLLC'][3], 'b:')
    ax.set_title('Gas Velocity (u_g)')

    # 4. Pressure
    ax = axes[1,1]
    ax.plot(x_ref, P_ref, 'k-', linewidth=2)
    ax.plot(results['PCCU'][0], results['PCCU'][4], 'r-')
    ax.plot(results['Roe'][0], results['Roe'][4], 'g--')
    ax.plot(results['HLLC'][0], results['HLLC'][4], 'b:')
    ax.set_title('Pressure (P)')

    plt.tight_layout()
    plt.show()
