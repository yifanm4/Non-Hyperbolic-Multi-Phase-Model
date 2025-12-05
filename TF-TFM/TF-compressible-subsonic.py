import numpy as np
import matplotlib.pyplot as plt

def solve_simulation_complete():
    # ==========================================
    # 1. SETUP & PARAMETERS
    # ==========================================
    L = 50.0            # Length [m]
    N = 100             # Grid points
    dx = L / N

    T_final = 0.100     # Final time [s]
    dt = 5e-6           # Time step [s]
    n_steps = int(round(T_final / dt))

    # Save steps for t = 0, 25, 50, 75, 100 ms
    save_times = [0.0, 0.025, 0.050, 0.075, 0.100]
    save_steps = [int(round(t/dt)) for t in save_times]

    # Constants
    Cv1 = 1355.0; R1 = 452.0
    Cv2 = 2710.0; R2 = 903.0

    # Grid (Staggered)
    x_c = np.linspace(dx/2, L - dx/2, N) # Cell centers

    # ==========================================
    # 2. INITIAL CONDITIONS
    # ==========================================
    def get_ICs():
        term_sin = np.sin(np.pi * x_c / L)

        # Multi-gas ICs
        p1_mg = 12e6 * (1.0 + term_sin) * (0.5 + 0.1 * term_sin)
        p2_mg = 12e6 * (1.0 + term_sin) * (0.5 - 0.1 * term_sin)
        T_mg = 413.0

        # Multiphase ICs
        p_mp = 12e6 * (1.0 + term_sin)
        alpha1_mp = 0.5 + 0.1 * term_sin
        alpha2_mp = 1.0 - alpha1_mp
        T_mp = 413.0

        return p1_mg, p2_mg, T_mg, p_mp, alpha1_mp, alpha2_mp, T_mp

    def get_flux(u, phi):
        flx = np.zeros(len(u))
        for i in range(1, len(u)-1):
            if u[i] >= 0: flx[i] = u[i] * phi[i-1]
            else: flx[i] = u[i] * phi[i]
        return flx

    # ==========================================
    # 3. SOLVER: MULTI-GAS
    # ==========================================
    print("Running Multi-Gas Simulation...")
    p1_0, p2_0, T_0, _, _, _, _ = get_ICs()

    # Initialize Conservative Variables
    rho1 = p1_0 / (R1 * T_0)
    rho2 = p2_0 / (R2 * T_0)
    E1 = rho1 * Cv1 * T_0
    E2 = rho2 * Cv2 * T_0

    u1 = np.zeros(N+1)
    u2 = np.zeros(N+1)

    mg_results = {}

    # Save initial state
    curr_T1 = E1/(rho1*Cv1); P1 = rho1*R1*curr_T1
    curr_T2 = E2/(rho2*Cv2); P2 = rho2*R2*curr_T2
    mg_results[0] = {
        'x': x_c,
        'p': P1+P2,
        'u1': 0.5*(u1[:-1]+u1[1:]),
        'u2': 0.5*(u2[:-1]+u2[1:])
    }

    # Working variables
    r1 = rho1.copy(); r2 = rho2.copy()
    e1 = E1.copy(); e2 = E2.copy()
    v1 = u1.copy(); v2 = u2.copy()

    for n in range(1, n_steps+1):
        # --- Update Mass ---
        f_r1 = get_flux(v1, r1)
        f_r2 = get_flux(v2, r2)
        r1 = r1 - (dt/dx)*(f_r1[1:] - f_r1[:-1])
        r2 = r2 - (dt/dx)*(f_r2[1:] - f_r2[:-1])

        # --- Update Energy ---
        curr_T1 = e1 / (r1 * Cv1)
        curr_P1 = r1 * R1 * curr_T1
        curr_H1 = e1 + curr_P1
        f_e1 = get_flux(v1, curr_H1)

        curr_T2 = e2 / (r2 * Cv2)
        curr_P2 = r2 * R2 * curr_T2
        curr_H2 = e2 + curr_P2
        f_e2 = get_flux(v2, curr_H2)

        e1 = e1 - (dt/dx)*(f_e1[1:] - f_e1[:-1])
        e2 = e2 - (dt/dx)*(f_e2[1:] - f_e2[:-1])

        # --- Update Momentum ---
        dp1 = np.zeros(N+1); dp2 = np.zeros(N+1)
        dp1[1:-1] = (curr_P1[1:] - curr_P1[:-1])/dx
        dp2[1:-1] = (curr_P2[1:] - curr_P2[:-1])/dx

        rf1 = np.zeros(N+1); rf2 = np.zeros(N+1)
        rf1[1:-1] = 0.5*(r1[1:] + r1[:-1])
        rf2[1:-1] = 0.5*(r2[1:] + r2[:-1])

        v1_new = np.zeros_like(v1)
        v2_new = np.zeros_like(v2)

        for i in range(1, N):
            if v1[i]>0: du1 = (v1[i]-v1[i-1])/dx
            else: du1 = (v1[i+1]-v1[i])/dx

            if v2[i]>0: du2 = (v2[i]-v2[i-1])/dx
            else: du2 = (v2[i+1]-v2[i])/dx

            v1_new[i] = v1[i] + dt * (-v1[i]*du1 - (1.0/rf1[i])*dp1[i])
            v2_new[i] = v2[i] + dt * (-v2[i]*du2 - (1.0/rf2[i])*dp2[i])

        v1 = v1_new; v2 = v2_new

        if n in save_steps:
            T1 = e1/(r1*Cv1); P1 = r1*R1*T1
            T2 = e2/(r2*Cv2); P2 = r2*R2*T2
            mg_results[n] = {
                'x': x_c,
                'p': P1+P2,
                'u1': 0.5*(v1[:-1]+v1[1:]),
                'u2': 0.5*(v2[:-1]+v2[1:])
            }

    # ==========================================
    # 4. SOLVER: MULTIPHASE
    # ==========================================
    print("Running Multiphase Simulation...")
    _, _, _, p_0, a1_0, a2_0, T_mp_0 = get_ICs()

    rhop1 = p_0 / (R1 * T_mp_0)
    rhop2 = p_0 / (R2 * T_mp_0)
    m1 = a1_0 * rhop1
    m2 = a2_0 * rhop2
    E1_mp = m1 * Cv1 * T_mp_0
    E2_mp = m2 * Cv2 * T_mp_0

    u1_mp = np.zeros(N+1)
    u2_mp = np.zeros(N+1)

    mp_results = {}
    mp_results[0] = {
        'x': x_c,
        'p': p_0,
        'u1': 0.5*(u1_mp[:-1]+u1_mp[1:]),
        'u2': 0.5*(u2_mp[:-1]+u2_mp[1:])
    }

    curr_a1 = a1_0.copy(); curr_p = p_0.copy()
    K1 = R1/Cv1; K2 = R2/Cv2

    for n in range(1, n_steps+1):
        # --- Update Mass ---
        f_m1 = get_flux(u1_mp, m1)
        f_m2 = get_flux(u2_mp, m2)
        m1 = m1 - (dt/dx)*(f_m1[1:] - f_m1[:-1])
        m2 = m2 - (dt/dx)*(f_m2[1:] - f_m2[:-1])

        # --- Update Energy ---
        H1_vol = E1_mp + curr_a1 * curr_p
        H2_vol = E2_mp + (1.0-curr_a1) * curr_p

        f_E1 = get_flux(u1_mp, H1_vol)
        f_E2 = get_flux(u2_mp, H2_vol)

        E1_new = E1_mp - (dt/dx)*(f_E1[1:] - f_E1[:-1])
        E2_new = E2_mp - (dt/dx)*(f_E2[1:] - f_E2[:-1])

        # --- Update Momentum ---
        dp = np.zeros(N+1)
        dp[1:-1] = (curr_p[1:] - curr_p[:-1])/dx

        m1f = np.zeros(N+1); a1f = np.zeros(N+1); m2f = np.zeros(N+1); a2f = np.zeros(N+1)
        m1f[1:-1] = 0.5*(m1[1:]+m1[:-1])
        a1f[1:-1] = 0.5*(curr_a1[1:]+curr_a1[:-1])
        m2f[1:-1] = 0.5*(m2[1:]+m2[:-1])
        a2f[1:-1] = 1.0 - a1f[1:-1]

        v1_new = np.zeros_like(u1_mp)
        v2_new = np.zeros_like(u2_mp)

        for i in range(1, N):
            if u1_mp[i]>0: du1 = (u1_mp[i]-u1_mp[i-1])/dx
            else: du1 = (u1_mp[i+1]-u1_mp[i])/dx

            if u2_mp[i]>0: du2 = (u2_mp[i]-u2_mp[i-1])/dx
            else: du2 = (u2_mp[i+1]-u2_mp[i])/dx

            f1 = (a1f[i] / (m1f[i]+1e-9)) * dp[i]
            f2 = (a2f[i] / (m2f[i]+1e-9)) * dp[i]

            v1_new[i] = u1_mp[i] + dt * (-u1_mp[i]*du1 - f1)
            v2_new[i] = u2_mp[i] + dt * (-u2_mp[i]*du2 - f2)

        u1_mp = v1_new; u2_mp = v2_new

        # --- Energy Source ---
        a1_star = (K1 * E1_new) / (K1 * E1_new + K2 * E2_new)
        d_alpha = a1_star - curr_a1
        work = curr_p * d_alpha

        E1_mp = E1_new - work
        E2_mp = E2_new + work

        curr_a1 = (K1 * E1_mp) / (K1 * E1_mp + K2 * E2_mp)
        curr_p = (K1 * E1_mp) / (curr_a1 + 1e-9)

        if n in save_steps:
            mp_results[n] = {
                'x': x_c,
                'p': curr_p,
                'u1': 0.5*(u1_mp[:-1]+u1_mp[1:]),
                'u2': 0.5*(u2_mp[:-1]+u2_mp[1:])
            }

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    # --- PRESSURE PLOT ---
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    times = [0, 0.025, 0.050, 0.075, 0.100]

    for i, t in enumerate(times):
        step = save_steps[i]
        axes[i].plot(mg_results[step]['x'], mg_results[step]['p']/1e6, 'k--', label='Multi-gas')
        axes[i].plot(mp_results[step]['x'], mp_results[step]['p']/1e6, 'k-', label='Multiphase')
        axes[i].set_ylabel(f'P (MPa)\nt={t*1000:.0f}ms')

        # [MODIFIED] Commented out fixed limits as requested
        # axes[i].set_ylim(12, 16)

        if i==0: axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Distance (m)')
    plt.tight_layout()
    plt.savefig('comparison_pressure_updated.png')

    # --- VELOCITY PLOT ---
    fig2, axes2 = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    for i, t in enumerate(times):
        step = save_steps[i]
        ax = axes2[i]

        # Multi-gas velocities
        ax.plot(mg_results[step]['x'], mg_results[step]['u1'], 'b--', label='MG u1')
        ax.plot(mg_results[step]['x'], mg_results[step]['u2'], 'r--', label='MG u2')

        # Multiphase velocities
        ax.plot(mp_results[step]['x'], mp_results[step]['u1'], 'b-', label='MP u1')
        ax.plot(mp_results[step]['x'], mp_results[step]['u2'], 'r-', label='MP u2')

        ax.set_ylabel(f'Vel (m/s)\nt={t*1000:.0f}ms')
        if i==0: ax.legend(loc='upper right', fontsize='small', ncol=2)

    axes2[-1].set_xlabel('Distance (m)')
    plt.tight_layout()
    plt.savefig('comparison_velocity_updated.png')
    plt.show()

if __name__ == "__main__":
    solve_simulation_complete()
