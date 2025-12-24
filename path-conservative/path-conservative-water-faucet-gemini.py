'''
Comments: It runs ok, but the high resolution comes with high computational cost. And the results are not significantly better than standard CU.
Path-Conservative Central-Upwind (PCCU) scheme for the Water Faucet problem.
'''
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physics & Constants ---
G = 9.81
L_PIPE = 12.0
N_CELLS = 500
T_END = 0.5
DT_FIXED = 1e-4

# Stiffened Gas EOS Parameters
RHO_L_0 = 1000.0
C_L = 120.0
RHO_G_0 = 1.0
C_G = 120.0

def get_pressure(rho_l, rho_g):
    return C_L**2 * (rho_l - RHO_L_0)

# --- 2. Solver Class ---
class WaterFaucetSolver:
    def __init__(self, scheme='PCCU', n_cells=N_CELLS):
        self.scheme = scheme
        self.N = n_cells
        self.nghost = 2
        self.dx = L_PIPE / n_cells

        # Grid Center Points (Internal)
        self.x = np.linspace(self.dx/2, L_PIPE - self.dx/2, self.N)

        # State Vector U with Ghost Cells
        # Structure: [Padding_Left, Internal_Domain, Padding_Right]
        self.U = np.zeros((self.N + 2*self.nghost, 4))

        # Define Fixed Inlet State (Water Faucet IC)
        # alpha_l = 0.8, u_l = 10.0, u_g = 0.0
        self.al_in = 0.8
        self.ag_in = 0.2
        self.ul_in = 10.0
        self.ug_in = 0.0

        # Initial Condition: Uniform everywhere matching inlet
        self.initialize_domain()

    def initialize_domain(self):
        # Set internal cells
        idx_start = self.nghost
        idx_end = self.nghost + self.N

        self.U[idx_start:idx_end, 0] = self.al_in * RHO_L_0          # m_l
        self.U[idx_start:idx_end, 1] = self.al_in * RHO_L_0 * self.ul_in  # mom_l
        self.U[idx_start:idx_end, 2] = self.ag_in * RHO_G_0          # m_g
        self.U[idx_start:idx_end, 3] = self.ag_in * RHO_G_0 * self.ug_in  # mom_g

        # Apply BCs immediately to fill ghosts
        self.apply_bcs()

    def apply_bcs(self):
        # LEFT BOUNDARY (Fixed Inlet)
        # Force ghost cells to exact Inlet State
        for i in range(self.nghost):
            self.U[i, 0] = self.al_in * RHO_L_0
            self.U[i, 1] = self.al_in * RHO_L_0 * self.ul_in
            self.U[i, 2] = self.ag_in * RHO_G_0
            self.U[i, 3] = self.ag_in * RHO_G_0 * self.ug_in

        # RIGHT BOUNDARY (Open / Extrapolation)
        # Copy last internal cell to right ghost cells
        idx_last_internal = self.nghost + self.N - 1
        for i in range(self.nghost):
            self.U[idx_last_internal + 1 + i, :] = self.U[idx_last_internal, :]

    def get_prim(self, U):
        eps = 1e-10
        m_l, mom_l = U[:, 0], U[:, 1]
        m_g, mom_g = U[:, 2], U[:, 3]

        al = m_l / RHO_L_0
        al = np.clip(al, 1e-4, 1.0-1e-4)
        ag = 1.0 - al

        rl = m_l / al
        rg = m_g / ag

        ul = mom_l / (m_l + eps)
        ug = mom_g / (m_g + eps)

        P = get_pressure(rl, rg)
        return al, ag, rl, rg, ul, ug, P

    def get_flux(self, U):
        al, ag, rl, rg, ul, ug, P = self.get_prim(U)
        F = np.zeros_like(U)
        F[:, 0] = U[:, 0] * ul
        F[:, 1] = U[:, 0] * ul**2 + al * P
        F[:, 2] = U[:, 2] * ug
        F[:, 3] = U[:, 2] * ug**2 + ag * P
        return F

    def minmod(self, a, b):
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    def reconstruct(self):
        # We reconstruct states at interfaces i+1/2
        # We need slopes for cells [nghost-1] to [nghost+N]

        # Calculate slopes using central difference of neighbors
        # dU[i] uses U[i+1] and U[i-1]

        # Slice for calculation (exclude extreme edges)
        U_center = self.U[1:-1]
        U_left   = self.U[:-2]
        U_right  = self.U[2:]

        diff_p = U_right - U_center
        diff_m = U_center - U_left

        slope = np.zeros_like(self.U)
        # Slope at index i corresponds to cell i
        slope[1:-1] = self.minmod(diff_p, diff_m)

        if self.scheme == 'Upwind':
            slope[:] = 0.0 # Force 1st order

        # Reconstruct at interfaces
        # U_L[i] is state at Right Face of Cell i
        # U_R[i] is state at Left Face of Cell i

        U_L_face = self.U + 0.5 * slope
        U_R_face = self.U - 0.5 * slope

        # We need pairs (U_L, U_R) at the interface between i and i+1
        # Interface j is between cell j and j+1
        # Left state: U_L_face[j]
        # Right state: U_R_face[j+1]

        return U_L_face[:-1], U_R_face[1:]

    def get_non_cons_term(self, U_L, U_R):
        """PCCU Path Term: P_avg * Delta_Alpha"""
        al_L, _, _, _, _, _, P_L = self.get_prim(U_L)
        al_R, _, _, _, _, _, P_R = self.get_prim(U_R)

        P_avg = 0.5 * (P_L + P_R)
        d_alpha_l = al_R - al_L
        d_alpha_g = -d_alpha_l

        B = np.zeros_like(U_L)
        B[:, 1] = P_avg * d_alpha_l
        B[:, 3] = P_avg * d_alpha_g
        return B

    def step(self, dt):
        self.apply_bcs() # Vital: Enforce inlet before flux calc

        # 1. Reconstruction
        # U_L, U_R are arrays of states at all interfaces j
        U_L, U_R = self.reconstruct()

        # 2. Fluxes
        F_L = self.get_flux(U_L)
        F_R = self.get_flux(U_R)

        # 3. Wave Speeds
        _, _, _, _, ul_L, ug_L, _ = self.get_prim(U_L)
        _, _, _, _, ul_R, ug_R, _ = self.get_prim(U_R)

        S_L = np.maximum(np.abs(ul_L) + C_L, np.abs(ug_L) + C_G)
        S_R = np.maximum(np.abs(ul_R) + C_L, np.abs(ug_R) + C_G)
        a = np.maximum(S_L, S_R)

        # 4. Numerical Flux Calculation
        # Rusanov base: 0.5(FL+FR) - 0.5a(UR-UL)
        NumFlux = 0.5 * (F_L + F_R) - 0.5 * a[:,None] * (U_R - U_L)

        # 5. Non-Conservative / Path Terms
        Path_Contrib = np.zeros_like(NumFlux)
        if self.scheme == 'PCCU':
            B_Psi = self.get_non_cons_term(U_L, U_R)
            Path_Contrib = B_Psi # Store B at interface

        # 6. Update Internal Cells
        # We iterate over internal cells.
        # Internal cell index i (in self.U) corresponds to:
        #   Left Interface index: i-1
        #   Right Interface index: i
        # because reconstruct() returns arrays starting at interface 0 (between cell 0 and 1)

        # Indices for internal update
        start = self.nghost
        end   = self.nghost + self.N

        # Flux Divergence
        # Flux entering from left: NumFlux[start-1 : end-1]
        # Flux leaving to right:   NumFlux[start : end]

        Flux_In  = NumFlux[start-1 : end-1]
        Flux_Out = NumFlux[start : end]

        flux_diff = (Flux_Out - Flux_In) / self.dx

        # Source Terms (Volume)
        U_internal = self.U[start:end]
        al, ag, rl, rg, _, _, P = self.get_prim(U_internal)

        # Gravity
        S_vol = np.zeros_like(U_internal)
        S_vol[:, 1] += al * rl * G
        S_vol[:, 3] += ag * rg * G

        # Pressure Volume Work (P * grad alpha)
        # Central difference using neighbors
        al_right = self.get_prim(self.U[start+1:end+1])[0]
        al_left  = self.get_prim(self.U[start-1:end-1])[0]
        d_al_dx = (al_right - al_left) / (2*self.dx)

        S_vol[:, 1] += P * d_al_dx
        S_vol[:, 3] += P * (-d_al_dx)

        # PCCU Interface Correction
        if self.scheme == 'PCCU':
            # Add interface path terms to the divergence
            # Cell i gets +0.5 * B_{i+1/2} (Right Interface)
            # Cell i gets +0.5 * B_{i-1/2} (Left Interface)

            B_Right = Path_Contrib[start:end]
            B_Left  = Path_Contrib[start-1:end-1]

            PC_Source = (0.5 * B_Right + 0.5 * B_Left) / self.dx
            S_vol += PC_Source

        dU_dt = -flux_diff + S_vol

        # Update
        self.U[start:end] += dt * dU_dt

    def run(self):
        t = 0
        while t < T_END:
            self.step(DT_FIXED)
            t += DT_FIXED

        # Return only internal domain
        idx_s = self.nghost
        idx_e = self.nghost + self.N
        return self.x, self.get_prim(self.U[idx_s:idx_e])[1]

# --- 3. Analytical Solution ---
def get_analytical(t):
    x_ana = np.linspace(0, L_PIPE, 500)
    v0 = 10.0
    alpha_l_0 = 0.8

    # Wave front location
    x_front = v0*t + 0.5*G*t**2

    # Velocity profile behind front
    v_x = np.sqrt(v0**2 + 2*G*x_ana)

    # Alpha_l profile
    alpha_l = np.where(x_ana < x_front, alpha_l_0 * v0 / v_x, alpha_l_0)
    alpha_g = 1.0 - alpha_l
    return x_ana, alpha_g

# --- 4. Run Comparisons ---
print("Running schemes...")
s1 = WaterFaucetSolver('Upwind')
x1, ag1 = s1.run()

s2 = WaterFaucetSolver('CU')
x2, ag2 = s2.run()

s3 = WaterFaucetSolver('PCCU')
x3, ag3 = s3.run()

x_ref, ag_ref = get_analytical(T_END)

# --- 5. Plotting ---
plt.figure(figsize=(10, 6))

plt.plot(x_ref, ag_ref, 'k-', linewidth=2.0, label='Analytical')
plt.plot(x1, ag1, 'g--', linewidth=2.0, label='Upwind')
plt.plot(x2, ag2, 'b-.', linewidth=2.0, label='Standard CU')
plt.plot(x3, ag3, 'r.-', linewidth=1.0, markersize=4, label='PCCU')

plt.title(f"Water Faucet (t={T_END}s)")
plt.xlabel("Pipe Length (m)")
plt.ylabel("Gas Void Fraction")
plt.ylim(0.18, 0.45)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
