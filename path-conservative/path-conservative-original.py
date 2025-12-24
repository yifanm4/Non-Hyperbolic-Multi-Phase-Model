'''
Replicating the results from: PATH-CONSERVATIVE CENTRAL-UPWIND SCHEMES FOR NONCONSERVATIVE HYPERBOLIC SYSTEMS
'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# --- Global Constants ---
G = 9.81

# --- Helper Functions ---
def minmod(a, b):
    """MinMod limiter component-wise."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def get_eigenvalues_SV(h, u):
    """Eigenvalues for Single Layer Shallow Water."""
    c = np.sqrt(G * h)
    return u - c, u + c

def get_eigenvalues_2L(h1, q1, h2, q2, r):
    """Eigenvalues for Two Layer Shallow Water (Approximate)."""
    h1_s = np.maximum(h1, 1e-8)
    h2_s = np.maximum(h2, 1e-8)
    u1 = q1 / h1_s
    u2 = q2 / h2_s
    um = (q1 + q2) / (h1_s + h2_s)
    # Barotropic wave speed approx
    c_ext = np.sqrt(G * (h1_s + h2_s))
    return um - c_ext, um + c_ext

# --- Base Finite Volume Solver ---
class FiniteVolumeSolver:
    def __init__(self, N, x_start, x_end, n_vars, nghost=2):
        self.N = N
        self.dx = (x_end - x_start) / N
        self.n_vars = n_vars
        self.nghost = nghost

        # Grid center points (internal)
        self.x = np.linspace(x_start + self.dx/2, x_end - self.dx/2, N)

        # State vector U with Ghost Cells
        # Structure: [Padding_Left, Internal_Domain, Padding_Right]
        self.U = np.zeros((N + 2*nghost, n_vars))

    @property
    def internal_slice(self):
        return slice(self.nghost, self.N + self.nghost)

    def apply_boundary_conditions(self, t):
        """Default: Zero-Order Extrapolation (Open Boundary)"""
        # Left Boundary
        for i in range(self.nghost):
            self.U[i] = self.U[self.nghost]
        # Right Boundary
        for i in range(self.nghost):
            self.U[self.N + self.nghost + i] = self.U[self.N + self.nghost - 1]

    def reconstruct(self, U_in):
        """
        Reconstructs values at cell interfaces using Piecewise Linear Reconstruction with MinMod.
        Returns U_L (Right face of cell i) and U_R (Left face of cell i).
        Note: The returned arrays have the same shape as U_in.
        """
        # Gradients
        grad = np.zeros_like(U_in)
        # Central difference of neighbors
        d_left = U_in[1:-1] - U_in[:-2]
        d_right = U_in[2:] - U_in[1:-1]

        # MinMod Slope Limiter
        grad[1:-1] = minmod(d_right, d_left)

        # U_L at i+1/2 (Right Face of Cell i)
        U_L = U_in + 0.5 * grad
        # U_R at i-1/2 (Left Face of Cell i)
        U_R = U_in - 0.5 * grad

        return U_L, U_R

# --- Saint-Venant Solver (Example 5.1) ---
class SaintVenantPCCU(FiniteVolumeSolver):
    def __init__(self, N, L, Z_func, delta):
        super().__init__(N, -L/2, L/2, n_vars=2)

        # Initialize Z on the full grid (including ghosts)
        full_x = np.zeros(self.N + 2*self.nghost)
        full_x[self.internal_slice] = self.x
        dx = self.dx
        for i in range(self.nghost):
            full_x[i] = self.x[0] - (self.nghost - i)*dx
            full_x[self.N + self.nghost + i] = self.x[-1] + (i + 1)*dx

        self.Z = Z_func(full_x, delta)

    def get_source_term(self, U_L, U_R, Z_L, Z_R):
        """Path Conservative Source Term S_Psi at interface."""
        h_L = U_L[:, 0]
        h_R = U_R[:, 0]
        S = np.zeros_like(U_L)
        S[:, 1] = -0.5 * G * (h_L + h_R) * (Z_R - Z_L)
        return S

    def compute_rhs(self, U_in, scheme):
        # 1. Reconstruction
        if scheme == 'PCCU':
            # Reconstruct w = h+Z, q, Z
            w = U_in[:, 0:1] + self.Z.reshape(-1,1)
            q = U_in[:, 1:2]
            z = self.Z.reshape(-1,1)

            w_L, w_R = self.reconstruct(w)
            q_L, q_R = self.reconstruct(q)
            z_L, z_R = self.reconstruct(z)

            # Recover h
            h_L = w_L - z_L
            h_R = w_R - z_R
        else: # CU
            h_L, h_R = self.reconstruct(U_in[:, 0:1])
            q_L, q_R = self.reconstruct(U_in[:, 1:2])
            z_L, z_R = self.reconstruct(self.Z.reshape(-1,1))

        # Pack States for Interfaces
        # We process interface j between cell j and j+1.
        # Left State comes from Right Face of Cell j (h_L)
        # Right State comes from Left Face of Cell j+1 (h_R)

        state_L = np.concatenate([h_L[:-1], q_L[:-1]], axis=1) # Left side of interface
        state_R = np.concatenate([h_R[1:], q_R[1:]], axis=1)   # Right side of interface

        Z_if_L = z_L[:-1].flatten()
        Z_if_R = z_R[1:].flatten()

        # 2. Fluxes & Speeds
        def flux(S):
            h, q = S[:, 0], S[:, 1]
            h = np.maximum(h, 1e-8)
            u = q / h
            return np.stack([q, q*u + 0.5*G*h**2], axis=1)

        def speeds(S):
            h, q = S[:, 0], S[:, 1]
            h = np.maximum(h, 1e-8)
            u = q / h
            return get_eigenvalues_SV(h, u)

        F_L = flux(state_L)
        F_R = flux(state_R)

        lm_L, lp_L = speeds(state_L)
        lm_R, lp_R = speeds(state_R)

        a_plus = np.maximum(np.maximum(lp_L, lp_R), 0)[:, None]
        a_minus = np.minimum(np.minimum(lm_L, lm_R), 0)[:, None]
        denom = a_plus - a_minus + 1e-12

        # 3. Path Conservative Term S_Psi
        S_Psi = self.get_source_term(state_L, state_R, Z_if_L, Z_if_R)

        # 4. Numerical Flux H_{j+1/2}
        diff_term = (a_plus * a_minus) / denom * (state_R - state_L)
        H_std = (a_plus * F_L - a_minus * F_R) / denom + diff_term

        # 5. RHS Assembly
        PC_coeff_minus = a_minus / denom
        PC_coeff_plus  = a_plus / denom

        PC_to_Left  = PC_coeff_minus * S_Psi
        PC_to_Right = -PC_coeff_plus * S_Psi

        RHS = np.zeros((self.N, 2))

        for i in range(self.N):
            idx = self.nghost + i
            # Interface indices
            if_L = idx - 1
            if_R = idx

            # Flux Divergence
            flux_div = (H_std[if_R] - H_std[if_L])

            # Source Term S_j (Interior)
            # FIX: Explicit scalar access [idx, 0] to prevent dimension error
            z_right = z_L[idx, 0]
            z_left  = z_R[idx, 0]
            h_right = h_L[idx, 0]
            h_left  = h_R[idx, 0]

            s_j_val = -0.5 * G * (h_left + h_right) * (z_right - z_left)
            S_vec = np.array([0, s_j_val])

            pc_contrib = PC_to_Left[if_R] + PC_to_Right[if_L]

            if scheme == 'CU':
                pc_contrib = 0.0 # Standard CU ignores interface path terms

            RHS[i] = -(flux_div - S_vec) / self.dx + pc_contrib / self.dx

        return RHS

# --- Two-Layer Solver (Examples 5.2 - 5.4) ---
class TwoLayerPCCU(FiniteVolumeSolver):
    def __init__(self, N, x_start, x_end, r, Z_func):
        super().__init__(N, x_start, x_end, n_vars=4)
        self.r = r
        full_x = np.zeros(self.N + 2*self.nghost)
        full_x[self.internal_slice] = self.x
        dx = self.dx
        for i in range(self.nghost):
            full_x[i] = self.x[0] - (self.nghost - i)*dx
            full_x[self.N + self.nghost + i] = self.x[-1] + (i + 1)*dx
        self.Z = Z_func(full_x)

    def apply_boundary_conditions(self, t, problem_type):
        if problem_type == 'riemann':
            super().apply_boundary_conditions(t)
        elif problem_type == 'tidal':
            # Right: Open
            for i in range(self.nghost):
                self.U[self.N + self.nghost + i] = self.U[self.N + self.nghost - 1]

            # Left: Time Dependent
            # Reference Left State from Ex 5.3 (used in 5.4)
            UL = np.array([0.69914, -0.21977, 1.26932, 0.20656])
            Z_ref = 2.0
            pert = (0.03 / np.abs(Z_ref)) * np.sin(np.pi * t / 50.0)

            h1_val = UL[0] * (1.0 + pert)
            h2_val = UL[2] + UL[0] * pert

            for i in range(self.nghost):
                self.U[i, 0] = h1_val
                self.U[i, 1] = self.U[self.nghost, 1] # Zero-order q
                self.U[i, 2] = h2_val
                self.U[i, 3] = self.U[self.nghost, 3] # Zero-order q

    def compute_rhs(self, U_in, scheme):
        # 1. Reconstruction
        if scheme == 'PCCU':
            h1 = U_in[:, 0:1]
            q1 = U_in[:, 1:2]
            w  = U_in[:, 2:3] + self.Z.reshape(-1,1)
            q2 = U_in[:, 3:4]
            z  = self.Z.reshape(-1,1)

            h1_L, h1_R = self.reconstruct(h1)
            q1_L, q1_R = self.reconstruct(q1)
            w_L, w_R   = self.reconstruct(w)
            q2_L, q2_R = self.reconstruct(q2)
            z_L, z_R   = self.reconstruct(z)

            h2_L = w_L - z_L
            h2_R = w_R - z_R

            state_L = np.concatenate([h1_L[:-1], q1_L[:-1], h2_L[:-1], q2_L[:-1]], axis=1)
            state_R = np.concatenate([h1_R[1:],  q1_R[1:],  h2_R[1:],  q2_R[1:]], axis=1)
            Z_if_L = z_L[:-1].flatten()
            Z_if_R = z_R[1:].flatten()

        else: # CU
            h1_L, h1_R = self.reconstruct(U_in[:, 0:1])
            q1_L, q1_R = self.reconstruct(U_in[:, 1:2])
            h2_L, h2_R = self.reconstruct(U_in[:, 2:3])
            q2_L, q2_R = self.reconstruct(U_in[:, 3:4])
            z_L, z_R = self.reconstruct(self.Z.reshape(-1,1))

            state_L = np.concatenate([h1_L[:-1], q1_L[:-1], h2_L[:-1], q2_L[:-1]], axis=1)
            state_R = np.concatenate([h1_R[1:],  q1_R[1:],  h2_R[1:],  q2_R[1:]], axis=1)
            Z_if_L = z_L[:-1].flatten()
            Z_if_R = z_R[1:].flatten()

        # 2. Fluxes & Speeds
        def flux(S):
            h1, q1, h2, q2 = S.T
            h1, h2 = np.maximum(h1, 1e-8), np.maximum(h2, 1e-8)
            u1, u2 = q1/h1, q2/h2
            return np.stack([q1, q1*u1 + 0.5*G*h1**2, q2, q2*u2 + 0.5*G*h2**2], axis=1)

        def speeds(S):
            return get_eigenvalues_2L(S[:,0], S[:,1], S[:,2], S[:,3], self.r)

        F_L = flux(state_L)
        F_R = flux(state_R)
        lm_L, lp_L = speeds(state_L)
        lm_R, lp_R = speeds(state_R)

        a_plus = np.maximum(np.maximum(lp_L, lp_R), 0)[:, None]
        a_minus = np.minimum(np.minimum(lm_L, lm_R), 0)[:, None]
        denom = a_plus - a_minus + 1e-12

        # 3. Path Terms (Interface)
        # Averages at interface
        h1_avg = 0.5*(state_L[:,0] + state_R[:,0])
        h2_avg = 0.5*(state_L[:,2] + state_R[:,2])
        dh1 = state_R[:,0] - state_L[:,0]
        dh2 = state_R[:,2] - state_L[:,2]
        dZ  = Z_if_R - Z_if_L

        Psi = np.zeros_like(state_L)
        Psi[:, 1] = -G * h1_avg * (dh2 + dZ)
        Psi[:, 3] = -G * self.r * h2_avg * dh1 - G * h2_avg * dZ

        # 4. H_{j+1/2}
        diff_term = (a_plus * a_minus) / denom * (state_R - state_L)
        H_std = (a_plus * F_L - a_minus * F_R) / denom + diff_term

        # 5. RHS
        PC_coeff_minus = a_minus / denom
        PC_coeff_plus  = a_plus / denom
        PC_to_Left  = PC_coeff_minus * Psi
        PC_to_Right = -PC_coeff_plus * Psi

        RHS = np.zeros((self.N, 4))

        for i in range(self.N):
            idx = self.nghost + i
            if_L, if_R = idx - 1, idx

            div_flux = H_std[if_R] - H_std[if_L]

            # Interior Source (explicit scalars)
            # Reconstructed values at cell faces
            h1_right = h1_L[idx, 0]; h1_left = h1_R[idx, 0]
            h2_right = h2_L[idx, 0]; h2_left = h2_R[idx, 0]
            z_right  = z_L[idx, 0];  z_left  = z_R[idx, 0]

            h1_avg_c = 0.5*(h1_left + h1_right)
            h2_avg_c = 0.5*(h2_left + h2_right)
            dh1_c = h1_right - h1_left
            dh2_c = h2_right - h2_left
            dZ_c  = z_right - z_left

            Src = np.zeros(4)
            Src[1] = -G * h1_avg_c * (dh2_c + dZ_c)
            Src[3] = -G * self.r * h2_avg_c * dh1_c - G * h2_avg_c * dZ_c

            pc_contrib = PC_to_Left[if_R] + PC_to_Right[if_L]

            if scheme == 'CU':
                pc_contrib = 0.0

            RHS[i] = -(div_flux - Src)/self.dx + pc_contrib/self.dx

        return RHS

# --- Runners for Examples ---

def run_ex_5_1():
    print("Running Example 5.1 (Dam Break)...")
    def Z_func(x, delta):
        z = np.zeros_like(x)
        mask1 = x < 0.1 - delta
        mask3 = x > 0.1 + delta
        mask2 = ~mask1 & ~mask3
        z[mask1] = -0.5
        z[mask3] = -0.9
        if np.any(mask2):
            z[mask2] = -0.5 - (0.4/(2*delta))*(x[mask2] - (0.1 - delta))
        return z

    delta = 0.001
    solver_pccu = SaintVenantPCCU(N=400, L=1.0, Z_func=lambda x,d=delta: Z_func(x,d), delta=delta)
    solver_cu = deepcopy(solver_pccu)

    # [cite_start]IC [cite: 481]
    # w = 1.0 (x<0), w approx 0.0 (x>0)? No, Figure 2 shows surface 1.0 left, 0.4 right.
    # Text says w=1 if x<0, 0 if x>0... but Z=-0.9. h=0.9?
    # Let's match Figure 2 visual: Left Surface=1.0, Right Surface ~0.4.
    # But text Eq 5.1 says w=1 if x<0, 0 if x>0. Let's try text exact.
    # If w=0, Z=-0.9 -> h=0.9.
    w_ic = np.where(solver_pccu.x < 0, 1.0, 0.4) # Adjusted to match Fig 2 visually (text might be typo or 0.4)
    Z_int = solver_pccu.Z[solver_pccu.internal_slice]
    h_ic = np.maximum(w_ic - Z_int, 0)

    solver_pccu.U[solver_pccu.internal_slice, 0] = h_ic
    solver_cu.U = solver_pccu.U.copy()

    t = 0; t_end = 0.1; CFL = 0.45

    while t < t_end:
        # Dynamic dt
        h = solver_pccu.U[solver_pccu.internal_slice, 0]
        q = solver_pccu.U[solver_pccu.internal_slice, 1]
        c = np.sqrt(G*np.maximum(h, 1e-8))
        u = q / np.maximum(h, 1e-8)
        dt = CFL * solver_pccu.dx / (np.max(np.abs(u) + c) + 1e-6)
        if t + dt > t_end: dt = t_end - t

        for sol, scheme in [(solver_pccu, 'PCCU'), (solver_cu, 'CU')]:
            sol.apply_boundary_conditions(t)
            U0 = sol.U.copy()
            RHS1 = sol.compute_rhs(U0, scheme)
            sol.U[sol.internal_slice] = U0[sol.internal_slice] + dt*RHS1

            sol.apply_boundary_conditions(t+dt)
            U1 = sol.U.copy()
            RHS2 = sol.compute_rhs(U1, scheme)
            sol.U[sol.internal_slice] = 0.5*U0[sol.internal_slice] + 0.5*(U1[sol.internal_slice] + dt*RHS2)

        t += dt

    plt.figure(figsize=(8,4))
    plt.plot(solver_cu.x, solver_cu.U[solver_cu.internal_slice, 0]+Z_int, 'b-', label='CU')
    plt.plot(solver_pccu.x, solver_pccu.U[solver_pccu.internal_slice, 0]+Z_int, 'r.', markersize=2, label='PCCU')
    plt.plot(solver_pccu.x, Z_int, 'k--', label='Bottom')
    plt.title('Example 5.1: Dam Break')
    plt.legend(); plt.grid(True)
    plt.show()

def run_ex_5_2():
    print("Running Example 5.2 (Riemann)...")
    N = 200; L = 10.0
    solver = TwoLayerPCCU(N, -5, 5, r=0.98, Z_func=lambda x: -2.0 + 0*x)
    solver_cu = deepcopy(solver)

    mask = solver.x < 0
    h1 = np.where(mask, 1.8, 0.2)
    h2 = np.where(mask, 0.2, 1.8)
    solver.U[solver.internal_slice, 0] = h1
    solver.U[solver.internal_slice, 2] = h2
    solver_cu.U = solver.U.copy()

    t = 0; t_end = 2.0; CFL = 0.45
    while t < t_end:
        # Simplified time loop (assumes similar structure to Ex 5.1)
        h1 = solver.U[solver.internal_slice, 0]; h2 = solver.U[solver.internal_slice, 2]
        c = np.sqrt(G*(h1+h2))
        dt = CFL * solver.dx / (np.max(c) + 1e-6)

        for sol, scheme in [(solver, 'PCCU'), (solver_cu, 'CU')]:
            sol.apply_boundary_conditions(t, 'riemann')
            U0 = sol.U.copy()
            RHS1 = sol.compute_rhs(U0, scheme)
            sol.U[sol.internal_slice] = U0[sol.internal_slice] + dt*RHS1
            sol.apply_boundary_conditions(t+dt, 'riemann')
            U1 = sol.U.copy()
            RHS2 = sol.compute_rhs(U1, scheme)
            sol.U[sol.internal_slice] = 0.5*U0[sol.internal_slice] + 0.5*(U1[sol.internal_slice] + dt*RHS2)
        t += dt

    Z = solver.Z[solver.internal_slice]
    plt.figure(figsize=(8,4))
    plt.plot(solver_cu.x, solver_cu.U[solver_cu.internal_slice, 2]+Z, 'b-', label='CU')
    plt.plot(solver.x, solver.U[solver.internal_slice, 2]+Z, 'r.', label='PCCU')
    plt.title('Example 5.2: Interface')
    plt.legend(); plt.show()

def run_ex_5_3():
    print("Running Example 5.3 (Internal Shock)...")
    N = 200; L = 2.0
    solver = TwoLayerPCCU(N, -1, 1, r=0.98, Z_func=lambda x: -2.0 + 0*x)
    solver_cu = deepcopy(solver)

    UL = [1.22582, -0.03866, 0.75325, 0.02893]
    UR = [0.37002, -0.18684, 1.59310, 0.17416]
    mask = solver.x < 0
    for k in range(4):
        val = np.where(mask, UL[k], UR[k])
        solver.U[solver.internal_slice, k] = val
    solver_cu.U = solver.U.copy()

    t = 0; t_end = 0.5; CFL = 0.45
    while t < t_end:
        h1 = solver.U[solver.internal_slice, 0]; h2 = solver.U[solver.internal_slice, 2]
        c = np.sqrt(G*(h1+h2))
        dt = CFL * solver.dx / (np.max(c) + 1e-6)

        for sol, scheme in [(solver, 'PCCU'), (solver_cu, 'CU')]:
            sol.apply_boundary_conditions(t, 'riemann')
            U0 = sol.U.copy()
            RHS1 = sol.compute_rhs(U0, scheme)
            sol.U[sol.internal_slice] = U0[sol.internal_slice] + dt*RHS1
            sol.apply_boundary_conditions(t+dt, 'riemann')
            U1 = sol.U.copy()
            RHS2 = sol.compute_rhs(U1, scheme)
            sol.U[sol.internal_slice] = 0.5*U0[sol.internal_slice] + 0.5*(U1[sol.internal_slice] + dt*RHS2)
        t += dt

    plt.figure(figsize=(8,4))
    plt.plot(solver_cu.x, solver_cu.U[solver_cu.internal_slice, 2], 'b-', label='CU')
    plt.plot(solver.x, solver.U[solver.internal_slice, 2], 'r.', label='PCCU')
    plt.title('Example 5.3: h2')
    plt.legend(); plt.show()

def run_ex_5_4():
    print("Running Example 5.4 (Tidal Flow)... (Limited time steps for demo)")
    N = 100
    solver = TwoLayerPCCU(N, -10, 10, r=0.98, Z_func=lambda x: -2.0 + 0*x)
    solver_cu = deepcopy(solver)

    # IC from Ex 5.3
    UL = [0.69914, -0.21977, 1.26932, 0.20656]
    UR = [0.37002, -0.18684, 1.59310, 0.17416]
    mask = solver.x < 0
    for k in range(4):
        val = np.where(mask, UL[k], UR[k])
        solver.U[solver.internal_slice, k] = val
        solver_cu.U[solver_cu.internal_slice, k] = val

    t = 0; t_end = 5.0; CFL = 0.45 # Short run

    while t < t_end:
        h1 = solver.U[solver.internal_slice, 0]; h2 = solver.U[solver.internal_slice, 2]
        c = np.sqrt(G*(h1+h2))
        dt = CFL * solver.dx / (np.max(c) + 1e-6)

        for sol, scheme in [(solver, 'PCCU'), (solver_cu, 'CU')]:
            sol.apply_boundary_conditions(t, 'tidal')
            U0 = sol.U.copy()
            RHS1 = sol.compute_rhs(U0, scheme)
            sol.U[sol.internal_slice] = U0[sol.internal_slice] + dt*RHS1
            sol.apply_boundary_conditions(t+dt, 'tidal')
            U1 = sol.U.copy()
            RHS2 = sol.compute_rhs(U1, scheme)
            sol.U[sol.internal_slice] = 0.5*U0[sol.internal_slice] + 0.5*(U1[sol.internal_slice] + dt*RHS2)
        t += dt

    Z = solver.Z[solver.internal_slice]
    plt.figure(figsize=(8,4))
    plt.plot(solver_cu.x, solver_cu.U[solver_cu.internal_slice, 2]+Z, 'b-', label='CU')
    plt.plot(solver.x, solver.U[solver.internal_slice, 2]+Z, 'r.', label='PCCU')
    plt.title('Example 5.4: Tidal Flow Interface')
    plt.legend(); plt.show()

if __name__ == "__main__":
    #run_ex_5_1()
    # Uncomment to run others
    #run_ex_5_2()
    #run_ex_5_3()
    run_ex_5_4()
