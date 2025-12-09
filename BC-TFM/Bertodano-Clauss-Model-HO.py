import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters & Geometry ---
L = 12.0          # Tube length [m]
N = 1000           # Number of nodes
dx = L / N        # Mesh size
T_final = 0.5     # Simulation time [s]
#dt = 1e-4         # Time step
dt = 1e-4*100/(2*N)         # Time step

# Fluid Properties
rho_1 = 1000.0    # Water (Heavy)
rho_2 = 1.0       # Air (Light)
d_rho = rho_1 - rho_2
g_x = 9.81        # Gravity

# Turbulent Viscosity Parameters
l_m = 0.1
nu_k = 1e-5

# --- Helper Functions for Physics ---
def get_Q(al):
    # Reciprocal inertial function
    # Clamp alpha to avoid division by zero stability issues
    al = np.clip(al, 1e-6, 1.0 - 1e-6)
    gamma = rho_1 / (1 - al) + rho_2 / al
    return 1.0 / gamma

def get_W_from_vars(al, u_1, u_2):
    u_r = u_2 - u_1
    J = al * (1 - al) * u_r
    gamma = rho_1 / (1 - al) + rho_2 / al
    return gamma * J

def get_fluxes(al, W, j_const):
    """
    Returns Mass Flux and Momentum Flux (A_W) for a given state.
    Used for both cell centers (Upwind) and reconstructed states (MUSCL).
    """
    Q = get_Q(al)

    # 1. Mass Flux
    flux_mass = al * j_const + Q * W

    # 2. Momentum Flux Term A_W
    # dGamma/da = rho1/(1-a)^2 - rho2/a^2
    al_safe = np.clip(al, 1e-6, 1.0 - 1e-6)
    dGamma_da = rho_1 / ((1 - al_safe)**2) - rho_2 / (al_safe**2)
    Q_prime = -1.0 * (Q**2) * dGamma_da

    A_W = 0.5 * Q_prime * W**2 + W * j_const - 0.5 * d_rho * j_const**2

    return flux_mass, A_W

# --- Limiter Function ---
def van_leer(r):
    # Van Leer Limiter: (r + |r|) / (1 + |r|)
    return (r + np.abs(r)) / (1.0 + np.abs(r))

# --- Analytic Solution ---
# def analytic_solution(x_grid):
#     v0 = 10.0
#     v_x = np.sqrt(v0**2 + 2 * g_x * x_grid)
#     al_l_analytic = (1 - 0.2) * 10.0 / v_x
#     return 1.0 - al_l_analytic

def analytic_solution(x_grid, t_curr):
    """
    Ransom Faucet Analytic Solution at time t.
    Handles the transient contact discontinuity.
    """
    v0 = 10.0
    alpha0 = 0.2

    # 1. Calculate the location of the "front" (distance traveled by particle at inlet)
    # x = v0*t + 0.5*g*t^2
    x_front = v0 * t_curr + 0.5 * g_x * t_curr**2

    alpha_analytic = np.zeros_like(x_grid)

    for i, x_val in enumerate(x_grid):
        if x_val < x_front:
            # --- Region 1: New Fluid (Steady State Profile) ---
            # Velocity has increased due to gravity: v = sqrt(v0^2 + 2gx)
            v_x = np.sqrt(v0**2 + 2 * g_x * x_val)

            # Conservation of liquid mass: (1-alpha)*v = (1-alpha0)*v0
            # (1 - alpha) = (1 - 0.2) * 10 / v_x
            alpha_l = 1.0 - ( (1.0 - alpha0) * v0 / v_x )
            alpha_analytic[i] = alpha_l
        else:
            # --- Region 2: Old Fluid (Initial Condition) ---
            # The fluid here started with uniform velocity v0.
            # While it accelerates in time (v = v0 + gt), it remains spatially uniform.
            # dv/dx = 0 implies d(alpha)/dt = 0.
            alpha_analytic[i] = alpha0

    return alpha_analytic

# --- Update the Plotting Section ---
# When you plot, make sure to pass T_final to the function:
# plt.plot(x_up, analytic_solution(x_up, T_final), 'k--', ...)

# --- Main Solver Function ---
def run_simulation(scheme='upwind'):
    # Initialization
    x = np.linspace(dx/2, L - dx/2, N)
    alpha = np.ones(N) * 0.2
    u1 = np.ones(N) * 10.0
    u2 = np.zeros(N)

    # Invariants
    j_inlet = (1 - 0.2) * 10.0 + 0.2 * 0.0
    j = np.ones(N) * j_inlet

    # Initial W
    W = get_W_from_vars(alpha, u1, u2)

    # Pre-calculate Inlet Fluxes (Fixed BC)
    W_in = get_W_from_vars(0.2, 10.0, 0.0)
    flux_mass_in, A_W_in = get_fluxes(0.2, W_in, j_inlet)

    for t in np.arange(0, T_final, dt):

        # --- Flux Calculation ---
        if scheme == 'upwind':
            # 1st Order Upwind: Flux at face i uses state at i-1 (since velocity > 0)
            # We calculate fluxes at cell centers, then shift for faces
            f_mass_center, f_W_center = get_fluxes(alpha, W, j)

            # Flux at Left Face of cell i
            flux_mass_face = np.zeros(N)
            flux_W_face    = np.zeros(N)

            # Interior faces take neighbor value
            flux_mass_face[1:] = f_mass_center[:-1]
            flux_W_face[1:]    = f_W_center[:-1]

            # Inlet face
            flux_mass_face[0] = flux_mass_in
            flux_W_face[0]    = A_W_in

            # Flux diffs
            # Outgoing flux for cell i is flux_face[i+1] which corresponds to center[i]
            d_flux_mass = (f_mass_center - flux_mass_face) / dx
            d_A_W       = (f_W_center - flux_W_face) / dx

        elif scheme == 'muscl':
            # MUSCL reconstruction
            # We need to reconstruct alpha and W at the Left and Right of interfaces.
            # Since flow is strictly positive (downstream), we only need U_Left at face i+1/2

            # 1. Calculate Slopes
            # r_i = (U_i - U_{i-1}) / (U_{i+1} - U_i)
            # Note: This requires care at boundaries.

            def get_reconstructed_L(arr):
                # Returns the Reconstructed LEFT state at the OUTGOING face (i+1/2)
                # U_L_{i+1/2} = U_i + 0.5 * phi(r) * (U_i - U_{i-1})

                U_recon = np.copy(arr)

                # We can only do high order from i=1 to N-2
                # Calculate diffs
                diff = np.diff(arr) # length N-1

                # r[i] involves diff[i-1] / diff[i]
                # We handle i=1..N-2
                denom = diff[1:]      # U_{i+1} - U_i
                numer = diff[:-1]     # U_i - U_{i-1}

                # Avoid div by zero
                r = np.zeros_like(denom)
                mask = np.abs(denom) > 1e-10
                r[mask] = numer[mask] / denom[mask]

                phi = van_leer(r)

                # Add correction to U_i (indices 1 to N-2)
                # U_recon[1:-1] are the cells we are correcting
                # The slope uses the diff from the left (numer)
                U_recon[1:-1] = arr[1:-1] + 0.5 * phi * numer

                return U_recon

            alpha_L = get_reconstructed_L(alpha)
            W_L     = get_reconstructed_L(W)

            # Calculate fluxes based on Reconstructed Left States
            f_mass_recon, f_W_recon = get_fluxes(alpha_L, W_L, j)

            # Flux at Left Face of cell i
            flux_mass_face = np.zeros(N)
            flux_W_face    = np.zeros(N)

            flux_mass_face[1:] = f_mass_recon[:-1]
            flux_W_face[1:]    = f_W_recon[:-1]

            flux_mass_face[0] = flux_mass_in
            flux_W_face[0]    = A_W_in

            # Outgoing flux is just the calculated recon flux at current cell
            d_flux_mass = (f_mass_recon - flux_mass_face) / dx
            d_A_W       = (f_W_recon - flux_W_face) / dx

        # --- Forces (Gravity + Viscosity) ---
        Q_curr = get_Q(alpha)

        # Gravity
        F_grav = -g_x * d_rho

        # Viscosity (Central Difference Stabilizer)
        u_r = (Q_curr * W) / (alpha * (1 - alpha) + 1e-6)
        nu = nu_k + l_m * np.abs(u_r)

        dW_dx = np.gradient(W, dx)
        shear = nu * dW_dx
        F_visc = np.gradient(shear, dx)

        # --- Updates ---
        alpha_new = alpha - dt * d_flux_mass
        W_new = W + dt * (-d_A_W + F_grav + F_visc)

        alpha = alpha_new
        W = W_new

    return x, alpha

# --- 5. Run & Visualization ---
print("Running First-Order Upwind...")
x_up, alpha_up = run_simulation('upwind')

print("Running MUSCL + Van Leer...")
x_muscl, alpha_muscl = run_simulation('muscl')

print("Plotting...")
plt.figure(figsize=(10, 6))

# Plot Analytic
#plt.plot(x_up, analytic_solution(x_up), 'k--', linewidth=2, label='Analytical Solution')
plt.plot(x_up, analytic_solution(x_up, T_final), 'k--', linewidth=2, label='Analytical Solution')

# Plot Upwind
plt.plot(x_up, alpha_up, 'b-', alpha=0.6, label='1st Order Upwind')

# Plot MUSCL
plt.plot(x_muscl, alpha_muscl, 'r-', linewidth=2, label='MUSCL + Van Leer')

plt.title(f'Water Faucet Problem (T={T_final}s)\nComparison of Numerical Schemes')
plt.xlabel('Position x [m]')
plt.ylabel('Void Fraction alpha')
plt.ylim(0.15, 0.50)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
