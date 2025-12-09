import numpy as np
import matplotlib.pyplot as plt

def get_diffs(u, dx):
    """Calculates forward and backward differences for the limiter."""
    # u_i - u_{i-1}
    du_minus = u - np.roll(u, 1)
    # u_{i+1} - u_i
    du_plus = np.roll(u, -1) - u
    return du_minus, du_plus

def van_leer_limiter(r):
    """Van Leer slope limiter."""
    # r is the ratio of slopes (du_minus / du_plus)
    # Formula: (r + |r|) / (1 + |r|)
    # We use a vectorized version that handles division by zero safely
    numerator = r + np.abs(r)
    denominator = 1.0 + np.abs(r)
    return np.where(denominator != 0, numerator / denominator, 0.0)

def solve_comparison():
    # --- 1. Parameters ---
    L = 2.0 * np.pi
    N = 1000
    c = 1.0
    T_max = 2000.0        # Longer time to really show the decay
    CFL = 0.5           # Kept lower for high-order stability

    dx = L / N
    dt = CFL * dx / c
    x = np.linspace(0, L, N, endpoint=False)

    # --- 2. Initialization ---
    u_initial = np.sin(x)

    # Initialize solvers
    u_upwind = u_initial.copy()
    u_muscl = u_initial.copy()

    # History tracking
    time_hist = []
    energy_upwind = []
    energy_muscl = []
    energy_exact = []

    # Initial Energy
    E0 = np.sum(u_initial**2) * dx

    # --- 3. Time Loop ---
    t = 0.0
    while t < T_max:
        # Track Energy (Action proxy)
        time_hist.append(t)
        energy_upwind.append(np.sum(u_upwind**2) * dx)
        energy_muscl.append(np.sum(u_muscl**2) * dx)
        energy_exact.append(E0) # Analytic energy is constant

        # --- A. Update 1st Order Upwind ---
        # u_i = u_i - nu * (u_i - u_{i-1})
        u_upwind = u_upwind - CFL * (u_upwind - np.roll(u_upwind, 1))

        # --- B. Update 2nd Order MUSCL ---
        # 1. Calculate Slopes (r)
        du_minus, du_plus = get_diffs(u_muscl, dx)

        # Avoid divide by zero for r calculation
        # We add a tiny epsilon or handle mask
        r = np.zeros_like(u_muscl)
        mask = du_plus != 0
        r[mask] = du_minus[mask] / du_plus[mask]

        # 2. Apply Limiter (Phi)
        phi = van_leer_limiter(r)

        # 3. Reconstruct Left States at interfaces (i+1/2)
        # Higher order correction: 0.5 * phi * (1 - CFL) * du_plus
        # This (1-CFL) term makes it 2nd order in TIME as well (Lax-Wendroff type flux)
        flux_correction = 0.5 * phi * (1 - CFL) * du_plus

        # u_L at interface i+1/2
        u_L_face = u_muscl + flux_correction

        # 4. Update Solution
        # Flux F_{i+1/2} = c * u_L_{i+1/2}
        # Update: u_i = u_i - CFL * (u_L_{i+1/2} - u_L_{i-1/2})
        flux = c * u_L_face
        u_muscl = u_muscl - (dt/dx) * (flux - np.roll(flux, 1))

        t += dt

    # --- 4. Plotting ---
    plt.figure(figsize=(14, 6))

    # Plot 1: Wave Shapes at Final Time
    plt.subplot(1, 2, 1)
    # Exact solution shifts by c*t
    u_exact_final = np.sin(x - c * T_max)

    plt.plot(x, u_exact_final, 'k--', alpha=0.6, label='Exact (No Loss)')
    plt.plot(x, u_upwind, 'r-', label='1st Order Upwind')
    plt.plot(x, u_muscl, 'g-', linewidth=2, label='2nd Order MUSCL')

    plt.title(f'Wave Shape at t={T_max:.1f}')
    plt.xlabel('Position x')
    plt.ylabel('u(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Action/Energy Decay
    plt.subplot(1, 2, 2)
    plt.plot(time_hist, np.array(energy_exact)/E0, 'k--', alpha=0.6, label='Exact')
    plt.plot(time_hist, np.array(energy_muscl)/E0, 'g-', linewidth=2, label='MUSCL')
    plt.plot(time_hist, np.array(energy_upwind)/E0, 'r-', label='Upwind')

    plt.title('Normalized Action/Energy over Time')
    plt.xlabel('Time')
    plt.ylabel('Energy Ratio $E(t)/E_0$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("advection_action_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    solve_comparison()
