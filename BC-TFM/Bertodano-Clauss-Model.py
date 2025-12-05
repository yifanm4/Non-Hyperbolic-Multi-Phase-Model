import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters & Geometry ---
L = 12.0          # Tube length [m]
N = 100           # Number of nodes
dx = L / N        # Mesh size [cite: 279]
T_final = 0.5     # Simulation time [s]
dt =1e-4       # Time step (small for stability)

# Fluid Properties
rho_1 = 1000.0    # Water (Heavy)
rho_2 = 1.0       # Air (Light)
d_rho = rho_1 - rho_2
g_x = 9.81        # Gravity (Driving force)

# Turbulent Viscosity Parameters (From Paper)
l_m = 0.1         # Mixing length [cite: 119]
nu_k = 1e-5       # Kinematic viscosity

# --- 2. Initialization ---
x = np.linspace(dx/2, L - dx/2, N)

# Initial State: alpha = 0.2, u_l = 10, u_g = 0
alpha = np.ones(N) * 0.2
u1 = np.ones(N) * 10.0
u2 = np.zeros(N)

# Derived Quantities used in Paper
j_inlet = (1 - 0.2) * 10.0 + 0.2 * 0.0  # j is constant in space
j = np.ones(N) * j_inlet

# Calculate W (Relative Momentum)
def get_Q(al):
    # Reciprocal inertial function Q [cite: 74, 81]
    # Gamma_0 = rho1/(1-a) + rho2/a
    gamma = rho_1 / (1 - al) + rho_2 / al
    return 1.0 / gamma

def get_W(al, u_1, u_2):
    u_r = u_2 - u_1
    J = al * (1 - al) * u_r
    gamma = rho_1 / (1 - al) + rho_2 / al
    return gamma * J

W = get_W(alpha, u1, u2)

# --- 3. Analytic Solution (Ransom Faucet) ---
def analytic_solution(x_grid, t):
    # Distance a particle travels in time t: x = v0*t + 0.5*g*t^2
    # But here we look for steady state profile or transient
    # Standard analytic for void fraction at time t:
    v0 = 10.0
    # Velocity at x: v(x) = sqrt(v0^2 + 2*g*x)
    v_x = np.sqrt(v0**2 + 2 * g_x * x_grid)
    # Conservation: alpha_l * v_l = const -> (1-alpha)*v = (1-0.2)*10
    al_l_analytic = (1 - 0.2) * 10.0 / v_x
    return 1.0 - al_l_analytic

# --- 4. Solver Loop ---
# We solve for alpha (Mass) and W (Relative Momentum)
# Using Upwind Differencing for Advection

for t in np.arange(0, T_final, dt):
    # --- A. Update Alpha (Mass Eq)  ---
    # d_t(alpha) + d_x(alpha*j + Q*W) = 0
    Q = get_Q(alpha)
    flux_mass = alpha * j + Q * W

    # First-order upwind flux
    # Velocity of waves is roughly u2. In faucet, everything moves downstream (+x)
    d_flux_mass = np.zeros(N)
    d_flux_mass[1:] = (flux_mass[1:] - flux_mass[:-1]) / dx
    d_flux_mass[0] = (flux_mass[0] - (0.2 * j_inlet + get_Q(0.2)*get_W(0.2, 10, 0))) / dx # Inlet BC

    alpha_new = alpha - dt * d_flux_mass

    # --- B. Update W (Momentum Eq)  ---
    # d_t(W) + d_x(A_W) = Forces

    # Calculate Flux Term A_W [cite: 97]
    # Need Q_prime (derivative of Q w.r.t alpha)
    # d(1/Gamma)/d_alpha = -1/Gamma^2 * dGamma/dalpha
    # dGamma/da = rho1/(1-a)^2 - rho2/a^2
    dGamma_da = rho_1 / ((1 - alpha)**2) - rho_2 / (alpha**2)
    Q_prime = -1.0 * (Q**2) * dGamma_da

    A_W = 0.5 * Q_prime * W**2 + W * j - 0.5 * d_rho * j**2

    # Discretize Advection d_x(A_W) (Upwind)
    d_A_W = np.zeros(N)
    d_A_W[1:] = (A_W[1:] - A_W[:-1]) / dx
    # Inlet BC for W
    W_in = get_W(0.2, 10, 0)
    Q_in = get_Q(0.2)
    Gamma_in = 1/Q_in
    dGamma_da_in = rho_1 / ((1 - 0.2)**2) - rho_2 / (0.2**2)
    Q_prime_in = -1.0 * (Q_in**2) * dGamma_da_in
    A_W_in = 0.5 * Q_prime_in * W_in**2 + W_in * j_inlet - 0.5 * d_rho * j_inlet**2
    d_A_W[0] = (A_W[0] - A_W_in) / dx

    # Calculate Forces
    # 1. Gravity (Driving force)
    F_grav = -g_x * d_rho

    # 2. Viscosity (Paper's stabilizer) [cite: 107, 117]
    # F_visc = d_x(nu * d_x(W))
    # nu = nu_k + l_m * |u_r|
    # u_r = W / (alpha * (1-alpha) * Gamma) = Q*W / (alpha*(1-alpha))
    u_r = (Q * W) / (alpha * (1 - alpha) + 1e-6) # Avoid div/0
    nu = nu_k + l_m * np.abs(u_r)

    # Central difference for diffusion
    F_visc = np.zeros(N)
    # Interior nodes
    dW_dx = np.gradient(W, dx)
    shear = nu * dW_dx
    F_visc = np.gradient(shear, dx)

    # Explicit Update W (Since j is constant, d_t(W - drho*j) = d_t(W))
    W_new = W + dt * (-d_A_W + F_grav + F_visc)

    # Update State
    alpha = alpha_new
    W = W_new

# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(x, alpha, 'b-', label='Simulated (Paper TFM Model)')
plt.plot(x, analytic_solution(x, T_final), 'r--', label='Analytical Solution')
plt.title(f'Water Faucet Problem (T={T_final}s)\nUsing TFM with Turbulent Viscosity [cite: 115]')
plt.xlabel('Position x [m]')
plt.ylabel('Void Fraction alpha')
plt.ylim(0.15, 0.4)
plt.grid(True)
plt.legend()
plt.show()
