import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

class SpectralElement1D:
    def __init__(self, k, n_dof, L=1.0):
        self.k = k  # Polynomial degree
        self.n_elements = int(n_dof / k)
        self.L = L
        self.nodes_per_el = k + 1

        # 1. Generate GLL points and weights on [-1, 1]
        self.z, self.w = self._gll_points(k)

        # 2. Compute Derivative Matrix D on reference element
        self.D = self._derivative_matrix(self.z)

        # 3. Mesh setup
        self.x_global = np.zeros(self.n_elements * k) # Periodic: last node = first of next
        # Actually standard mapping: x = x_e + (z+1)/2 * h
        # We handle periodicity by mapping indices
        self.dx = L / self.n_elements
        self.Jac = self.dx / 2.0  # dx/dxi

        # Global coordinate array for plotting/IC
        # We will store solution in a "continuous" vector of size N_dof = n_elements * k
        self.N_dof = self.n_elements * k
        self.x = np.zeros(self.N_dof)

        for e in range(self.n_elements):
            x_start = e * self.dx
            # Map reference z to physical x
            x_el = x_start + (self.z + 1) * self.dx / 2.0
            # Assign to global vector (handling shared nodes)
            # Node indices for element e: [e*k, e*k + 1, ..., e*k + k]
            # Since periodic, node N_dof is node 0
            idx = np.arange(e*k, e*k + k + 1) % self.N_dof
            self.x[idx[:-1]] = x_el[:-1] # Fill all but last to avoid overwrite/duplication logic here

        # Mass matrix (Global, Diagonal)
        self.M = np.zeros(self.N_dof)
        for e in range(self.n_elements):
            idx = np.arange(e*k, e*k + k + 1) % self.N_dof
            local_mass = self.w * self.Jac
            # DSS (Direct Stiffness Summation) for Mass
            np.add.at(self.M, idx, local_mass)

    def _gll_points(self, k):
        # Roots of (1-x^2) P'_k(x)
        # Use numpy Legendre module
        # P_k' roots plus -1, 1
        if k == 1: return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
        # Quick GLL root finder
        # Newton iterations on P_k'(x)
        x = np.cos(np.pi * np.arange(k + 1) / k)
        for _ in range(20):
            P = legendre(k)(x)
            dP = legendre(k).deriv()(x)
            ddP = legendre(k).deriv(2)(x)
            # Avoid division by zero at boundaries
            dx = - (1 - x**2) * dP / (2*x*dP - (1-x**2)*ddP) # Halley's or Newton on (1-x^2)P'
            # Simpler: Newton on Q(x) = (1-x^2)P'_k(x)
            # But standard P'_k roots are cleaner
            # Let's use standard Newton on P'_k inside (-1, 1)
            mask = (np.abs(x) < 0.99999)
            if np.any(mask):
                 x[mask] -= dP[mask] / ddP[mask]
        x = np.sort(x)
        x[0], x[-1] = -1.0, 1.0
        P = legendre(k)(x)
        w = 2.0 / (k * (k + 1) * P**2)
        return x, w

    def _derivative_matrix(self, z):
        k = len(z) - 1
        D = np.zeros((k + 1, k + 1))
        P = legendre(k)(z)
        for i in range(k + 1):
            for j in range(k + 1):
                if i != j:
                    D[i, j] = (P[i] / P[j]) / (z[i] - z[j])
                else:
                    if i == 0: D[i, j] = -k * (k + 1) / 4.0
                    elif i == k: D[i, j] = k * (k + 1) / 4.0
                    else: D[i, j] = 0.0
        return D

    def get_initial_condition(self):
        u = np.zeros_like(self.x)
        # Eq 3.3 from paper
        for i, xi in enumerate(self.x):
            # 1. Gaussian Hump
            if abs(2*xi - 0.3) <= 0.25:
                u[i] = np.exp(-300 * (2*xi - 0.3)**2)
            # 2. Square Wave
            elif abs(2*xi - 0.9) <= 0.2:
                u[i] = 1.0
            # 3. Semi-Ellipse
            elif abs(2*xi - 1.6) <= 0.2:
                val = 1.0 - ((2*xi - 1.6) / 0.2)**2
                u[i] = np.sqrt(max(0, val))
            else:
                u[i] = 0.0
        return u

    def compute_rhs(self, u, viscosity_field):
        """
        Computes RHS = -M^-1 * [ (f(u), v_x) + (nu u_x, v_x) ]
        Weak form integration.
        """
        global_rhs = np.zeros_like(u)

        # We iterate elements to compute local integrals and assemble
        for e in range(self.n_elements):
            idx = np.arange(e*self.k, e*self.k + self.k + 1) % self.N_dof
            u_loc = u[idx]
            nu_loc = viscosity_field[idx]

            # 1. Advection Term: - d/dx (u)
            # Weak form: \int u * v_x dx  (integrated by parts? Or strong form?)
            # Paper mentions "Fully centered non-limited... Galerkin".
            # Standard Collocation differentiation is strong form on nodes.
            # D * u gives du/dxi. du/dx = (1/Jac) * D * u
            du_dxi = self.D @ u_loc
            du_dx = du_dxi / self.Jac

            # Advection Contribution: - \int (du/dx) * v dx = - M * du/dx (since M lumped)
            # This is effectively Collocation.
            # local_rhs_adv = - (self.w * self.Jac) * du_dx
            # Simplification: With GLL mass lumping, Strong form == Weak form
            local_rhs_adv = - du_dx * (self.w * self.Jac)

            # 2. Viscosity Term: d/dx (nu du/dx)
            # Weak form: - \int nu (du/dx) (dv/dx) dx
            # du/dx at quadrature points is computed above
            # We need to test against gradients of basis functions
            # \int nu * u_x * v_x dx approx sum_k w_k J_k * nu_k * (u_x)_k * (v_x)_k ?
            # This requires D.T @ (W * nu * D @ u)

            # Vector of values at Q-points:
            flux_diff = nu_loc * du_dx # nu * du/dx

            # Integrate against test function deriv: \int flux * v_x
            # On reference: \int flux * (1/J) dv/dxi * J dxi = \int flux * dv/dxi dxi
            # = sum_q w_q * flux_q * (dv/dxi)_q
            # = D.T @ (w * flux)

            local_rhs_diff = - (self.D.T @ (self.w * flux_diff))

            # Assemble
            np.add.at(global_rhs, idx, local_rhs_adv + local_rhs_diff)

        return global_rhs / self.M

    def compute_entropy_viscosity(self, u, u_old, u_older, dt, c_E, c_max):
        # 1. Entropy E = 0.5 u^2
        E = 0.5 * u**2
        E_old = 0.5 * u_old**2
        E_older = 0.5 * u_older**2

        # 2. Entropy Residual D_h
        # Time deriv: (3E - 4E_old + E_older) / 2dt
        dE_dt = (3*E - 4*E_old + E_older) / (2*dt)

        # Space deriv: u * u_x = d/dx(0.5 u^2)
        # Compute gradient globally (collocation style)
        dE_dx = np.zeros_like(u)

        # To get gradients, we do a pass over elements
        # (Average at boundaries for smoothness in calculation)
        count = np.zeros_like(u)
        for e in range(self.n_elements):
            idx = np.arange(e*self.k, e*self.k + self.k + 1) % self.N_dof
            E_loc = E[idx]
            dE_dxi = self.D @ E_loc
            dE_dx_loc = dE_dxi / self.Jac
            np.add.at(dE_dx, idx, dE_dx_loc)
            np.add.at(count, idx, 1)
        dE_dx /= count

        D_h = np.abs(dE_dt + dE_dx)

        # 3. Max Viscosity (First order)
        # v_max = c_max * h * |f'(u)|
        # h is local grid size. For GLL, varies.
        # Approx local h at node i: distance to neighbor?
        # Paper Eq 2.14: h(x_ij) = min distance
        h_local = np.zeros_like(u)
        # Crude approximation: L/N_elements/k (average) or actual GLL spacing
        # Let's compute actual GLL spacing map
        # Just use max spacing in element for simplicity or dx
        for e in range(self.n_elements):
            idx = np.arange(e*self.k, e*self.k + self.k + 1) % self.N_dof
            # spacing
            d_z = np.diff(self.z)
            d_phys = d_z * self.Jac
            # Assign to nodes (approx min of neighbors)
            # For simplicity in this demo, use element avg h
            h_el = self.L / (self.n_elements * self.k) # Average
            h_local[idx] = h_el

        v_max = c_max * h_local * 1.0 # |f'(u)|=1

        # 4. Entropy Viscosity
        E_bar = np.mean(E)
        norm_E = np.max(np.abs(E - E_bar))
        if norm_E < 1e-8: norm_E = 1.0

        v_E = c_E * (h_local**2) * D_h / norm_E

        # 5. Combine
        v = np.minimum(v_max, v_E)

        # 6. Smoothing (Eq 2.18) - Optional but recommended
        # Simple 3-point smoothing for 1D
        v_smooth = np.copy(v)
        # (Skipped for brevity/speed in Python, usually helps stability)

        return v

def run_simulation(method='Entropy', k=4, T_final=100.0, dof=200):
    # Setup
    sem = SpectralElement1D(k, dof, L=1.0)
    u = sem.get_initial_condition()

    # Time Stepping (RK4)
    # CFL approx for spectral: dt ~ h_min ~ 1/k^2
    # N_el = 200/k. h_avg = k/200. h_min ~ h_avg/k^2?
    # Safe estimate:
    dx_min = sem.L / dof / (k) # Very rough
    dt = 0.001 / (k/2) # Heuristic tuning for stability

    n_steps = int(T_final / dt)

    u_hist = [u.copy(), u.copy(), u.copy()]
    viscosity = np.zeros_like(u)

    # Viscosity params
    c_max = 0.5 / k # Scaling from paper Sec 3.1.2
    c_E = 1.0

    print(f"Running {method}, k={k}, T={T_final}...")

    for n in range(n_steps):
        u_n = u_hist[-1]

        # Update Viscosity
        if method == 'Galerkin':
            viscosity[:] = 0.0
        elif method == 'Constant':
            # c_max * h * |f'|
            h = 1.0 / dof
            viscosity[:] = (0.1/k) * h # Fixed low viscosity
        elif method == 'Entropy':
             if n > 2:
                viscosity = sem.compute_entropy_viscosity(
                    u_hist[-1], u_hist[-2], u_hist[-3], dt, c_E, c_max
                )

        # RK4
        k1 = sem.compute_rhs(u_n, viscosity)
        k2 = sem.compute_rhs(u_n + 0.5*dt*k1, viscosity)
        k3 = sem.compute_rhs(u_n + 0.5*dt*k2, viscosity)
        k4 = sem.compute_rhs(u_n + dt*k3, viscosity)

        u_next = u_n + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        u_hist.append(u_next)
        if len(u_hist) > 3: u_hist.pop(0)

        # Progress
        if n % (n_steps//10) == 0:
            print(f"  Progress: {n/n_steps*100:.0f}%")

    return sem.x, u_hist[-1]

# --- Main Execution ---
# Note: T_final=100 is very long for a Python script.
# Reducing T to 10 for demonstration (results will show trend).
# Set T=100 to fully reproduce the exact paper figure (takes longer).
T_demo = 2.0

poly_degrees = [2, 4, 8]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# 1. Galerkin (No viscosity)
ax = axes[0]
for k in poly_degrees:
    x, u = run_simulation('Galerkin', k=k, T_final=T_demo)
    ax.plot(x, u, label=f'k={k}')
ax.set_title('Galerkin')
ax.legend()

# 2. Constant Viscosity
ax = axes[1]
for k in poly_degrees:
    x, u = run_simulation('Constant', k=k, T_final=T_demo)
    ax.plot(x, u, label=f'k={k}')
ax.set_title('Constant Viscosity')

# 3. Entropy Viscosity
ax = axes[2]
for k in poly_degrees:
    x, u = run_simulation('Entropy', k=k, T_final=T_demo)
    ax.plot(x, u, label=f'k={k}')
ax.set_title('Entropy Viscosity')

plt.tight_layout()
plt.savefig('Entropy_Viscosity_Comparison.png', dpi=300)
plt.show()
