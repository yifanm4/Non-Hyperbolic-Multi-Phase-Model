import numpy as np
import matplotlib.pyplot as plt

def compute_pressure(rho, rho_u, E, gamma=1.4):
    """Calculates Pressure from Conservative variables."""
    rho = np.maximum(rho, 1e-8)
    u_sq = (rho_u**2) / rho
    return (gamma - 1.0) * (E - 0.5 * u_sq)

def get_flux(rho, rho_u, E, P):
    """Calculates Flux Vector F(Q)."""
    rho = np.maximum(rho, 1e-8)
    u = rho_u / rho

    f0 = rho_u
    f1 = rho_u * u + P
    f2 = u * (E + P)

    return np.array([f0, f1, f2])

def solve_euler_8_plots():
    # --- 1. Parameters ---
    gamma = 1.4
    N = 400
    L = 10.0
    t_end = 0.20*L
    CFL = 0.9
    dx = L / N
    x = np.linspace(0, L, N)

    # --- 2. Initial Conditions (Sod Shock Tube) ---
    rho_L, u_L, P_L = 1.0, 0.0, 1.0
    rho_R, u_R, P_R = 0.125, 0.0, 0.1

    rho = np.zeros(N)
    rho_u = np.zeros(N)
    E = np.zeros(N)

    mask_L = x < L/2
    rho[mask_L] = rho_L
    rho[~mask_L] = rho_R

    E[mask_L] = P_L / (gamma - 1.0)
    E[~mask_L] = P_R / (gamma - 1.0)

    # History Trackers
    history = {'time': [], 'mass': [], 'momentum': [], 'energy': [], 'action': []}

    # Initial Totals for Normalization
    M0 = np.sum(rho) * dx
    E0 = np.sum(E) * dx

    # --- 3. Time Stepping Loop ---
    t = 0.0
    while t < t_end:
        # --- A. Primitives & Integrals ---
        P = compute_pressure(rho, rho_u, E, gamma)
        rho_safe = np.maximum(rho, 1e-8)
        u = rho_u / rho_safe
        c = np.sqrt(gamma * P / rho_safe)

        # Calculate Integrals
        total_mass = np.sum(rho) * dx
        total_mom  = np.sum(rho_u) * dx
        total_eng  = np.sum(E) * dx

        # Action Density = Kinetic - Internal
        kinetic = 0.5 * rho * u**2
        internal = P / (gamma - 1.0)
        total_action = np.sum(kinetic - internal) * dx

        history['time'].append(t)
        history['mass'].append(total_mass)
        history['momentum'].append(total_mom)
        history['energy'].append(total_eng)
        history['action'].append(total_action)

        # --- B. Time Step ---
        max_wave = np.max(np.abs(u) + c)
        dt = CFL * dx / max_wave
        if t + dt > t_end: dt = t_end - t

        # --- C. HLL Flux Calculation ---
        # Reconstruct
        rho_L_i, rho_R_i = rho[:-1], rho[1:]
        rhou_L_i, rhou_R_i = rho_u[:-1], rho_u[1:]
        E_L_i, E_R_i = E[:-1], E[1:]
        P_L_i, P_R_i = P[:-1], P[1:]
        u_L_i, u_R_i = u[:-1], u[1:]
        c_L_i, c_R_i = c[:-1], c[1:]

        # Wave speeds
        S_L = np.minimum(u_L_i - c_L_i, u_R_i - c_R_i)
        S_R = np.maximum(u_L_i + c_L_i, u_R_i + c_R_i)

        # Fluxes
        F_L = get_flux(rho_L_i, rhou_L_i, E_L_i, P_L_i)
        F_R = get_flux(rho_R_i, rhou_R_i, E_R_i, P_R_i)

        # HLL Logic
        dQ_0 = rho_R_i - rho_L_i
        dQ_1 = rhou_R_i - rhou_L_i
        dQ_2 = E_R_i - E_L_i

        denom = S_R - S_L + 1e-8

        hll_0 = (S_R * F_L[0] - S_L * F_R[0] + S_L * S_R * dQ_0) / denom
        hll_1 = (S_R * F_L[1] - S_L * F_R[1] + S_L * S_R * dQ_1) / denom
        hll_2 = (S_R * F_L[2] - S_L * F_R[2] + S_L * S_R * dQ_2) / denom

        Flux_0 = np.where(S_L >= 0, F_L[0], np.where(S_R <= 0, F_R[0], hll_0))
        Flux_1 = np.where(S_L >= 0, F_L[1], np.where(S_R <= 0, F_R[1], hll_1))
        Flux_2 = np.where(S_L >= 0, F_L[2], np.where(S_R <= 0, F_R[2], hll_2))

        # Update
        rho[1:-1]   -= (dt/dx) * (Flux_0[1:] - Flux_0[:-1])
        rho_u[1:-1] -= (dt/dx) * (Flux_1[1:] - Flux_1[:-1])
        E[1:-1]     -= (dt/dx) * (Flux_2[1:] - Flux_2[:-1])

        # Transmissive Boundaries
        rho[0], rho[-1] = rho[1], rho[-2]
        rho_u[0], rho_u[-1] = rho_u[1], rho_u[-2]
        E[0], E[-1] = E[1], E[-2]

        t += dt

    # --- 4. Plotting (8 Graphs) ---
    # Final Primitive Variables
    P = compute_pressure(rho, rho_u, E, gamma)
    u = rho_u / (rho + 1e-8)
    e_internal = P / ((gamma - 1.0) * rho + 1e-8)

    t_hist = np.array(history['time'])

    plt.figure(figsize=(18, 10))

    # --- Row 1: Primitive Variables ---
    plt.subplot(2, 4, 1)
    plt.plot(x, rho, 'k-', linewidth=2)
    plt.title('Density (rho)')
    plt.grid(True)

    plt.subplot(2, 4, 2)
    plt.plot(x, u, 'b-', linewidth=2)
    plt.title('Velocity (u)')
    plt.grid(True)

    plt.subplot(2, 4, 3)
    plt.plot(x, P, 'r-', linewidth=2)
    plt.title('Pressure (P)')
    plt.grid(True)

    plt.subplot(2, 4, 4)
    plt.plot(x, e_internal, 'g-', linewidth=2)
    plt.title('Internal Energy (e)')
    plt.grid(True)

    # --- Row 2: Conservation Laws ---

    # 5. Mass (Normalized)
    plt.subplot(2, 4, 5)
    plt.plot(t_hist, np.array(history['mass'])/M0, 'k-')
    plt.title('Total Mass (Conserved)')
    plt.ylabel('M(t) / M0')
    plt.ylim(0.999, 1.001)
    plt.grid(True)

    # 6. Momentum (Linear Growth)
    plt.subplot(2, 4, 6)
    plt.plot(t_hist, history['momentum'], 'b-', label='Sim')
    plt.plot(t_hist, (P_L - P_R)*t_hist, 'k--', alpha=0.6, label='Force')
    plt.title('Total Momentum')
    plt.ylabel('Momentum')
    plt.legend()
    plt.grid(True)

    # 7. Total Energy (Normalized)
    plt.subplot(2, 4, 7)
    plt.plot(t_hist, np.array(history['energy'])/E0, 'r-')
    plt.title('Total Energy (Conserved)')
    plt.ylabel('E(t) / E0')
    plt.ylim(0.999, 1.001)
    plt.grid(True)

    # 8. Action (Lagrangian)
    plt.subplot(2, 4, 8)
    plt.plot(t_hist, history['action'], 'g-')
    plt.title('Total Action (T - V)')
    plt.ylabel('Integral')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("euler_hll_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    solve_euler_8_plots()
