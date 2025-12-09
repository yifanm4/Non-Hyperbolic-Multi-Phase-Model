import numpy as np
import matplotlib.pyplot as plt

def compute_pressure(rho, rho_u, E, gamma=1.4):
    """Calculates Pressure from Conservative variables."""
    rho = np.maximum(rho, 1e-8)
    u_sq = (rho_u**2) / rho
    return (gamma - 1.0) * (E - 0.5 * u_sq)

def get_flux_vectors(rho, rho_u, E, P):
    """Helper to compute physical flux vector F(Q)."""
    rho = np.maximum(rho, 1e-8)
    u = rho_u / rho
    f0 = rho_u
    f1 = rho_u * u + P
    f2 = u * (E + P)
    # Stack into (3, N) array for easier math later
    return np.array([f0, f1, f2])

def solve_euler_roe():
    # --- 1. Parameters ---
    gamma = 1.4
    N = 400
    L = 1.0
    t_end = 0.20
    CFL = 0.9
    dx = L / N
    x = np.linspace(0, L, N)

    # --- 2. Initial Conditions (Sod Shock Tube) ---
    rho_L0, u_L0, P_L0 = 1.0, 0.0, 1.0
    rho_R0, u_R0, P_R0 = 0.125, 0.0, 0.1

    rho = np.zeros(N)
    rho_u = np.zeros(N)
    E = np.zeros(N)

    mask_L = x < 0.5
    rho[mask_L] = rho_L0
    rho[~mask_L] = rho_R0
    E[mask_L] = P_L0 / (gamma - 1.0)
    E[~mask_L] = P_R0 / (gamma - 1.0)

    # History Trackers
    history = {'time': [], 'mass': [], 'momentum': [], 'energy': [], 'action': []}
    M0 = np.sum(rho) * dx
    E0 = np.sum(E) * dx

    # --- 3. Time Stepping Loop ---
    t = 0.0
    while t < t_end:
        # A. Primitives
        P = compute_pressure(rho, rho_u, E, gamma)
        rho_safe = np.maximum(rho, 1e-8)
        u = rho_u / rho_safe
        c = np.sqrt(gamma * P / rho_safe)
        H = (E + P) / rho_safe  # Enthalpy

        # B. Conservation Checks
        total_mass = np.sum(rho) * dx
        total_mom  = np.sum(rho_u) * dx
        total_eng  = np.sum(E) * dx
        kinetic = 0.5 * rho * u**2
        internal = P / (gamma - 1.0)
        total_action = np.sum(kinetic - internal) * dx

        history['time'].append(t)
        history['mass'].append(total_mass)
        history['momentum'].append(total_mom)
        history['energy'].append(total_eng)
        history['action'].append(total_action)

        # C. Time Step
        max_wave = np.max(np.abs(u) + c)
        dt = CFL * dx / max_wave
        if t + dt > t_end: dt = t_end - t

        # D. Roe Flux Calculation
        # -----------------------
        # We need Left (i) and Right (i+1) states at interfaces
        rho_L_iface, rho_R_iface = rho[:-1], rho[1:]
        u_L_iface, u_R_iface     = u[:-1], u[1:]
        P_L_iface, P_R_iface     = P[:-1], P[1:]
        H_L_iface, H_R_iface     = H[:-1], H[1:]
        E_L_iface, E_R_iface     = E[:-1], E[1:]
        rhou_L_iface, rhou_R_iface = rho_u[:-1], rho_u[1:]

        # 1. Roe Averages
        sqrt_rho_L = np.sqrt(rho_L_iface)
        sqrt_rho_R = np.sqrt(rho_R_iface)
        denom = sqrt_rho_L + sqrt_rho_R

        u_hat = (sqrt_rho_L * u_L_iface + sqrt_rho_R * u_R_iface) / denom
        H_hat = (sqrt_rho_L * H_L_iface + sqrt_rho_R * H_R_iface) / denom
        c_hat = np.sqrt((gamma - 1.0) * (H_hat - 0.5 * u_hat**2))

        # 2. Eigenvalues (Wave Speeds)
        # lambda_1 (Left Sound), lambda_2 (Contact), lambda_3 (Right Sound)
        l1 = u_hat - c_hat
        l2 = u_hat
        l3 = u_hat + c_hat

        # Entropy Fix (Harten): Prevent zero wave speeds (Rarefaction Shocks)
        epsilon = 0.2 * (np.abs(u_hat) + c_hat)
        def entropy_fix(lam):
            return np.where(np.abs(lam) < epsilon,
                           (lam**2 + epsilon**2) / (2 * epsilon),
                           np.abs(lam))

        abs_l1 = entropy_fix(l1)
        abs_l2 = entropy_fix(l2)
        abs_l3 = entropy_fix(l3)

        # 3. Wave Strengths (Alpha)
        # Differences across interface
        dp = P_R_iface - P_L_iface
        du = u_R_iface - u_L_iface
        drho = rho_R_iface - rho_L_iface

        # Projected diffs
        alpha_2 = drho - dp / (c_hat**2)
        alpha_3 = (dp + rho_L_iface * c_hat * du) / (2 * c_hat**2) # using approximate rho
        # Better approximation for alphas using Roe density:
        rho_hat = np.sqrt(rho_L_iface * rho_R_iface)
        alpha_3 = (dp + rho_hat * c_hat * du) / (2 * c_hat**2)
        alpha_1 = (dp - rho_hat * c_hat * du) / (2 * c_hat**2)

        # 4. Right Eigenvectors (K)
        # K1 = [1, u-c, H-uc]
        K1_0, K1_1, K1_2 = 1.0, u_hat - c_hat, H_hat - u_hat * c_hat
        # K2 = [1, u, 0.5*u^2]
        K2_0, K2_1, K2_2 = 1.0, u_hat, 0.5 * u_hat**2
        # K3 = [1, u+c, H+uc]
        K3_0, K3_1, K3_2 = 1.0, u_hat + c_hat, H_hat + u_hat * c_hat

        # 5. Compute Flux
        # F_{i+1/2} = 0.5 * (F_L + F_R) - 0.5 * sum(|lambda| * alpha * K)

        F_L_vec = get_flux_vectors(rho_L_iface, rhou_L_iface, E_L_iface, P_L_iface)
        F_R_vec = get_flux_vectors(rho_R_iface, rhou_R_iface, E_R_iface, P_R_iface)

        # Dissipation Terms
        # Term 1
        d1_0 = abs_l1 * alpha_1 * K1_0
        d1_1 = abs_l1 * alpha_1 * K1_1
        d1_2 = abs_l1 * alpha_1 * K1_2
        # Term 2
        d2_0 = abs_l2 * alpha_2 * K2_0
        d2_1 = abs_l2 * alpha_2 * K2_1
        d2_2 = abs_l2 * alpha_2 * K2_2
        # Term 3
        d3_0 = abs_l3 * alpha_3 * K3_0
        d3_1 = abs_l3 * alpha_3 * K3_1
        d3_2 = abs_l3 * alpha_3 * K3_2

        Flux_0 = 0.5 * (F_L_vec[0] + F_R_vec[0]) - 0.5 * (d1_0 + d2_0 + d3_0)
        Flux_1 = 0.5 * (F_L_vec[1] + F_R_vec[1]) - 0.5 * (d1_1 + d2_1 + d3_1)
        Flux_2 = 0.5 * (F_L_vec[2] + F_R_vec[2]) - 0.5 * (d1_2 + d2_2 + d3_2)

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
    P = compute_pressure(rho, rho_u, E, gamma)
    u = rho_u / (rho + 1e-8)
    e_internal = P / ((gamma - 1.0) * rho + 1e-8)
    t_hist = np.array(history['time'])

    plt.figure(figsize=(18, 10))

    # Primitives
    plt.subplot(2, 4, 1)
    plt.plot(x, rho, 'k-', linewidth=2)
    plt.title('Density (rho) - Roe Solver')
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

    # Conservation
    plt.subplot(2, 4, 5)
    plt.plot(t_hist, np.array(history['mass'])/M0, 'k-')
    plt.title('Total Mass')
    plt.ylabel('M(t) / M0')
    plt.ylim(0.999, 1.001)
    plt.grid(True)

    plt.subplot(2, 4, 6)
    plt.plot(t_hist, history['momentum'], 'b-')
    plt.plot(t_hist, (P_L0 - P_R0) * t_hist, 'k--', alpha=0.6)
    plt.title('Total Momentum')
    plt.grid(True)

    plt.subplot(2, 4, 7)
    plt.plot(t_hist, np.array(history['energy'])/E0, 'r-')
    plt.title('Total Energy')
    plt.ylabel('E(t) / E0')
    plt.ylim(0.999, 1.001)
    plt.grid(True)

    plt.subplot(2, 4, 8)
    plt.plot(t_hist, history['action'], 'g-')
    plt.title('Total Action (T - V)')
    plt.ylabel('Integral')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("euler_roe_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    solve_euler_roe()
