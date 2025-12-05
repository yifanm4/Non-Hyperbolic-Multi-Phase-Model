import numpy as np
import matplotlib.pyplot as plt

class GeneralFaucetSolver:
    def __init__(self, N=100, L=12.0, T=0.5, dt=1e-4):
        # Discretization parameters
        self.N = N
        self.L = L
        self.dx = L / N
        self.x = np.linspace(self.dx / 2, L - self.dx / 2, N)
        self.dt = dt
        self.T_final = T

        # Physics constants
        self.rho1 = 1000.0
        self.rho2 = 1.0
        self.d_rho = 999.0
        self.gx = 9.81

        # Regularization (Turbulence model parameters)
        self.lm = 0.05
        self.nu_k = 1e-5

        # State Initialization
        self.alpha = np.ones(N) * 0.2
        self.u1 = np.ones(N) * 10.0
        #self.u2 = np.zeros(N)
        self.u2 = np.ones(N) * 1e-10    # Small non-zero to avoid division by zero
        self.j = 8.0 # Constant total volumetric flux

        # Initialize Relative Momentum (W)
        self.W = self.get_W_from_primitives(self.alpha, self.u1, self.u2)

        # Storage for time derivatives (needed for pressure solver)
        self.rho_m_old = self.get_rho_m(self.alpha)
        self.J_old = self.get_J(self.alpha, self.W)

    # --- Helper Functions (Constitutive Relations) ---
    def get_rho_m(self, a):
        return (1 - a) * self.rho1 + a * self.rho2

    def get_Gamma(self, a):
        return self.rho1 / (1 - a) + self.rho2 / a

    def get_Q(self, a):
        return 1.0 / self.get_Gamma(a)

    def get_J(self, a, W):
        return self.get_Q(a) * W

    def get_W_from_primitives(self, a, u1, u2):
        return self.get_Gamma(a) * a * (1 - a) * (u2 - u1)

    def get_Q_prime(self, alpha):
        # Derivative dQ/da = -Q^2 * dGamma/da
        gamma_prime = self.rho1 / ((1 - alpha)**2) - self.rho2 / (alpha**2)
        return -1.0 * (self.get_Q(alpha)**2) * gamma_prime

    def recover_velocities(self, alpha, W):
        J = self.get_J(alpha, W)
        u_r = J / (alpha * (1 - alpha) + 1e-8)
        # u1 is liquid, u2 is gas
        u1 = self.j - alpha * u_r
        u2 = self.j + (1 - alpha) * u_r
        return u1, u2

    # --- Generalized Upwind Scheme ---
    def compute_flux_divergence(self, F_cell, v_eigen, F_inlet):
        """
        Computes dF/dx using First-Order Upwind based on v_eigen direction.
        """
        N = self.N

        # 1. Reconstruct Fluxes at Faces (0 to N)
        F_faces = np.zeros(N + 1)

        # --- Inlet Face (0) ---
        v_inlet = v_eigen[0]
        if v_inlet > 0:
            # Flow entering domain -> Use Boundary Condition
            F_faces[0] = F_inlet
        else:
            # Flow leaving domain -> Use internal node
            F_faces[0] = F_cell[0]

        # --- Internal Faces (1 to N-1) ---
        # Face i is between Node i-1 and Node i
        v_faces = 0.5 * (v_eigen[:-1] + v_eigen[1:])

        # Upwind Logic:
        # If v > 0, flow from Left (i-1)
        # If v < 0, flow from Right (i)
        F_faces[1:N] = np.where(v_faces > 0, F_cell[:-1], F_cell[1:])

        # --- Outlet Face (N) ---
        v_outlet = v_eigen[-1]
        if v_outlet > 0:
            # Flow leaving -> Transmissive BC
            F_faces[N] = F_cell[-1]
        else:
            # Flow entering from outlet -> Fallback (should apply BC here if needed)
            F_faces[N] = F_cell[-1]

        # 2. Compute Divergence
        # dF/dx = (F_right - F_left) / dx
        flux_divergence = (F_faces[1:] - F_faces[:-1]) / self.dx

        return flux_divergence

    # --- Main Time Step ---
    def step(self):
        # 1. Evolve Alpha and W
        self.update_conservation_eqs()

        # 2. Recover primitive velocities for analysis
        self.u1, self.u2 = self.recover_velocities(self.alpha, self.W)

        # 3. Solve for Pressure
        P = self.solve_pressure()

        # 4. Save old state for next step's time derivatives
        self.rho_m_old = self.get_rho_m(self.alpha)
        self.J_old = self.get_J(self.alpha, self.W)

        return P

    def update_conservation_eqs(self):
        # Calculate Eigenvelocity v_eig = Q'W + j
        # This determines the direction of information flow
        Q_prime = self.get_Q_prime(self.alpha)
        v_eig = Q_prime * self.W + self.j

        # --- 1. Mass Equation ---
        Q = self.get_Q(self.alpha)
        F_mass = self.alpha * self.j + Q * self.W

        # Inlet Flux BC
        Q_in = self.get_Q(0.2)
        W_in = self.get_W_from_primitives(0.2, 10.0, 0.0)
        F_mass_in = 0.2 * self.j + Q_in * W_in

        # Compute Divergence
        dF_mass_dx = self.compute_flux_divergence(
            F_mass,
            v_eigen=v_eig,
            F_inlet=F_mass_in
        )

        self.alpha -= self.dt * dF_mass_dx

        # --- 2. Relative Momentum Equation ---
        # Flux A_W calculation
        term1 = 0.5 * Q_prime * self.W**2
        term2 = self.W * self.j
        term3 = 0.5 * self.d_rho * self.j**2
        A_W = term1 + term2 - term3

        # Inlet Flux BC (Need Q' at inlet conditions)
        dG_da_in = self.rho1 / (0.8**2) - self.rho2 / (0.2**2)
        Qp_in = -1.0 * (Q_in**2) * dG_da_in

        term1_in = 0.5 * Qp_in * W_in**2
        term2_in = W_in * self.j
        A_W_in = term1_in + term2_in - term3

        # Compute Divergence
        dA_W_dx = self.compute_flux_divergence(
            A_W,
            v_eigen=v_eig,
            F_inlet=A_W_in
        )

        # Forces Calculation
        # Turbulent Viscosity
        u_r = (Q * self.W) / (self.alpha * (1 - self.alpha) + 1e-6)
        nu = self.nu_k + self.lm * np.abs(u_r)

        # Viscous force gradient
        dW_dx = np.gradient(self.W, self.dx)
        F_visc = np.gradient(nu * dW_dx, self.dx)

        # Gravity
        F_grav = -self.gx * self.d_rho

        # Update W
        self.W += self.dt * (-dA_W_dx + F_grav + F_visc)

    def solve_pressure(self):
        # Calculate gradients for the Mixture Momentum Equation
        A_m = (1 - self.alpha) * self.rho1 * self.u1**2 + \
              self.alpha * self.rho2 * self.u2**2

        dA_m_dx = np.gradient(A_m, self.dx)

        rho_m = self.get_rho_m(self.alpha)
        J = self.get_J(self.alpha, self.W)

        # Time derivative of momentum
        mom_new = rho_m * self.j - self.d_rho * J
        mom_old = self.rho_m_old * self.j - self.d_rho * self.J_old
        d_mom_dt = (mom_new - mom_old) / self.dt

        # Solve for Pressure Gradient
        dP_dx = self.gx * rho_m - dA_m_dx - d_mom_dt

        # Integrate Spatial Gradient (BC: P_outlet = 0)
        P = np.zeros(self.N)
        for i in range(self.N - 2, -1, -1):
            P[i] = P[i+1] - dP_dx[i] * self.dx

        return P

# --- Run & Plot ---
# Initialize Solver (Coarse Grid for Speed)
solver = GeneralFaucetSolver(N=100, T=0.5)
times = np.arange(0, 0.5, solver.dt)

for t in times:
    P = solver.step()

# Initialize Solver (Fine Grid for Speed)
solver_fine = GeneralFaucetSolver(N=1000, T=0.5)
times_fine = np.arange(0, 0.5, solver_fine.dt)

for t in times_fine:
    P_fine = solver_fine.step()


# # Initialize Solver (Finest Grid for Speed) # Not working
# solver_finest = GeneralFaucetSolver(N=2000, T=0.5)
# times_finest = np.arange(0, 0.5, solver_finest.dt/10)

# for t in times_fine:
#     P_finest = solver_finest.step()



# Create plots
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

# Plot 1: Void Fraction
ax[0].plot(solver.x, solver.alpha, 'b',label='Coarse Grid')
ax[0].plot(solver_fine.x, solver_fine.alpha, 'r--', label='Fine Grid')
#ax[0].plot(solver_finest.x, solver_finest.alpha, 'g--', label='Finest Grid')
ax[0].legend()
ax[0].set_title(r'Void Fraction $\alpha$')
ax[0].set_ylim(0.15, 0.45)
ax[0].set_xlabel('Position (m)')
ax[0].grid(True)

# Plot 2: Ug
ax[1].plot(solver.x, solver.u1, 'b', label='Coarse Grid')
ax[1].plot(solver_fine.x, solver_fine.u1, 'r--', label='Fine Grid')
#ax[1].plot(solver_finest.x, solver_finest.u1, 'g--', label='Finest Grid')
ax[1].legend()
ax[1].set_title('Phase Velocities')
ax[1].set_xlabel('Position (m)')
ax[1].set_ylabel('Velocity (m/s)')
ax[1].grid(True)

# Plot 3: Ul
ax[2].plot(solver.x, solver.u2, 'b', label='Coarse Grid')
ax[2].plot(solver_fine.x, solver_fine.u2, 'r--', label='Fine Grid')
#ax[2].plot(solver_finest.x, solver_finest.u2, 'g--', label='Finest Grid')
ax[2].legend()
ax[2].set_title('Phase Velocities')
ax[2].set_xlabel('Position (m)')
ax[2].set_ylabel('Velocity (m/s)')
ax[2].grid(True)

# Plot 4: Pressure
ax[3].plot(solver.x, P / 1000, 'k')
ax[3].plot(solver_fine.x, P_fine / 1000, 'r--')
#ax[3].plot(solver_finest.x, P_finest / 1000, 'g--')
ax[3].legend()
ax[3].set_title('Pressure Profile [kPa]')
ax[3].set_xlabel('Position (m)')
ax[3].grid(True)

plt.tight_layout()
plt.savefig('BC_water_faucet_results.png', dpi=300)
plt.show()
