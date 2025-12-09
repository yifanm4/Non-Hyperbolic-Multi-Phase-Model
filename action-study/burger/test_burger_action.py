import numpy as np
import matplotlib.pyplot as plt

def van_leer(r):
    """Van Leer slope limiter for MUSCL scheme."""
    numerator = r + np.abs(r)
    denominator = 1.0 + np.abs(r)
    return np.where(denominator != 0, numerator / denominator, 0.0)

def get_rhs_advection_muscl(u, dx, dt):
    """Calculates the Advective Flux update using 2nd Order MUSCL."""
    # 1. Slope Calculation
    du_minus = u - np.roll(u, 1)
    du_plus = np.roll(u, -1) - u

    r = np.zeros_like(u)
    mask = du_plus != 0
    r[mask] = du_minus[mask] / du_plus[mask]

    phi = van_leer(r)

    # 2. Reconstruct Left State (u_{i+1/2}^L)
    # Using local velocity for the characteristic correction
    local_cfl = np.abs(u) * (dt/dx)
    u_L = u + 0.5 * phi * (1 - local_cfl) * du_plus

    # 3. Flux (Conservative Form: F = 0.5 * u^2)
    # Since u > 0, the flux at the face is determined by u_L
    F_face = 0.5 * u_L**2

    # Return net flux: -(F_{i+1/2} - F_{i-1/2}) / dx
    return - (F_face - np.roll(F_face, 1)) / dx

def get_rhs_diffusion(u, dx, nu):
    """Calculates the Diffusive term using Central Differences."""
    if nu == 0:
        return np.zeros_like(u)

    # u_xx approx (u_{i+1} - 2u_i + u_{i-1}) / dx^2
    u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
    return nu * u_xx

def solve_viscous_comparison():
    # --- 1. Parameters ---
    L = 2.0 * np.pi
    N = 200
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    T_max = 2.0

    # Compare 3 viscosities
    viscosities = [0.0, 0.01, 0.1]
    labels = ['Inviscid (nu=0)', 'Low Visc (nu=0.01)', 'High Visc (nu=0.1)']
    colors = ['k', 'r', 'b']

    # Initial Condition: Shifted Sine
    u0 = 1.0 + 0.5 * np.sin(x)

    # Storage for results
    results = []

    # --- 2. Simulation Loop for each Viscosity ---
    for nu in viscosities:
        u = u0.copy()
        t = 0.0

        # History tracking
        time_hist = []
        mass_hist = []
        energy_hist = []

        M0 = np.sum(u) * dx
        E0 = np.sum(0.5 * u**2) * dx

        while t < T_max:
            # --- Dynamic Time Step ---
            max_u = np.max(np.abs(u))

            # Advection constraint (CFL < 1.0)
            dt_adv = 0.5 * dx / (max_u + 1e-6)

            # Diffusion constraint (Von Neumann: dt < dx^2 / 2nu)
            if nu > 0:
                dt_diff = 0.4 * (dx**2) / nu
            else:
                dt_diff = 1.0 # Infinite if no viscosity

            dt = min(dt_adv, dt_diff)

            # Sync to end time
            if t + dt > T_max: dt = T_max - t

            # --- Record Data ---
            time_hist.append(t)
            mass_hist.append(np.sum(u)*dx)
            energy_hist.append(np.sum(0.5 * u**2)*dx)

            # --- Update Step (Operator Splitting-ish) ---
            # 1. Advection Contribution (MUSCL)
            rhs_adv = get_rhs_advection_muscl(u, dx, dt)

            # 2. Diffusion Contribution (Central)
            rhs_diff = get_rhs_diffusion(u, dx, nu)

            # 3. Time Integration (Euler Explicit)
            # u_new = u + dt * (Advection + Diffusion)
            u = u + dt * (rhs_adv + rhs_diff)

            t += dt

        results.append({
            'u': u,
            'time': time_hist,
            'mass': mass_hist,
            'energy': energy_hist,
            'M0': M0,
            'E0': E0
        })

    # --- 3. Visualization ---
    plt.figure(figsize=(14, 10))

    # Plot A: Final Wave Shapes
    plt.subplot(2, 2, 1)
    plt.plot(x, u0, 'g--', alpha=0.3, label='Initial')
    for i, res in enumerate(results):
        plt.plot(x, res['u'], color=colors[i], linewidth=2, label=labels[i])
    plt.title(f'Wave Shape at t={T_max}')
    plt.xlabel('Position x')
    plt.ylabel('Velocity u')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot B: Zoom in on the "Shock"
    plt.subplot(2, 2, 2)
    # Find the shock location (approx index 150 for this setup)
    zoom_slice = slice(120, 180)
    for i, res in enumerate(results):
        plt.plot(x[zoom_slice], res['u'][zoom_slice], color=colors[i], marker='o', markersize=3, label=labels[i])
    plt.title('Zoom on Shock Front')
    plt.xlabel('Position x')
    plt.grid(True, alpha=0.3)

    # Plot C: Mass Conservation
    plt.subplot(2, 2, 3)
    for i, res in enumerate(results):
        # Normalize mass
        norm_mass = np.array(res['mass']) / res['M0']
        plt.plot(res['time'], norm_mass, color=colors[i], label=labels[i])
    plt.title('Mass Conservation')
    plt.ylabel('M(t) / M(0)')
    plt.xlabel('Time')
    plt.ylim(0.999, 1.001)
    plt.grid(True, alpha=0.3)

    # Plot D: Energy Decay
    plt.subplot(2, 2, 4)
    for i, res in enumerate(results):
        norm_energy = np.array(res['energy']) / res['E0']
        plt.plot(res['time'], norm_energy, color=colors[i], label=labels[i])
    plt.title('Energy Decay (Dissipation)')
    plt.ylabel('E(t) / E(0)')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_viscous_comparison()
