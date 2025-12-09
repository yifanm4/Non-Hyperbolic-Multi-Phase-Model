import numpy as np
import matplotlib.pyplot as plt

def solve_updated_multiphase_shock_tube():
    # ==========================================
    # 1. PARAMETERS (Multiphase Case)
    # ==========================================
    L = 44.0
    N = 960
    dx = L / N

    T_final = 0.008
    dt = 1e-6
    n_steps = int(T_final / dt)

    # # Relaxation times
    # tau_D = 5e-6
    # tau_T = 5e-6

    tau_D = 1e-3
    tau_T = 1e-3

    # Phase 1 (Gas/High Pressure)
    Cv1 = 13550.0
    R1 = 4517.0

    # Phase 2 (Liquid/Low Pressure)
    Cv2 = 1355.0
    R2 = 451.7

    x_diaphragm = 14.6

    # Grid
    x_c = np.linspace(dx/2, L - dx/2, N)

    # ==========================================
    # 2. INITIALIZATION
    # ==========================================
    # Primitive Variables
    p = np.zeros(N)
    alpha1 = np.zeros(N)
    alpha2 = np.zeros(N)
    T1 = np.zeros(N)
    T2 = np.zeros(N)

    # Velocities (Faces)
    u1 = np.zeros(N+1)
    u2 = np.zeros(N+1)

    EPS = 1e-6

    for i in range(N):
        if x_c[i] < x_diaphragm:
            # LEFT: High P
            p[i] = 20e6
            T1[i] = 414.0
            T2[i] = 414.0
            alpha1[i] = 1.0 - EPS
            alpha2[i] = EPS
        else:
            # RIGHT: Low P
            p[i] = 0.1e6
            T1[i] = 300.0
            T2[i] = 300.0
            alpha1[i] = EPS
            alpha2[i] = 1.0 - EPS

    # Calculate Conservative Scalars
    # Initial rho calculation: rho = p / (R*T)
    rho1 = p / (R1 * T1)
    rho2 = p / (R2 * T2)

    # Conservative variables: m (macroscopic density), E (total energy density)
    m1 = rho1 * alpha1
    m2 = rho2 * alpha2

    E1_vol = m1 * (Cv1 * T1)
    E2_vol = m2 * (Cv2 * T2)

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================
    def upwind_flux(u_face, scalar_cell):
        flux = np.zeros_like(u_face)
        for i in range(1, N):
            if u_face[i] >= 0:
                flux[i] = u_face[i] * scalar_cell[i-1]
            else:
                flux[i] = u_face[i] * scalar_cell[i]
        return flux

    # ==========================================
    # 3. TIME LOOP
    # ==========================================
    print(f"Starting Simulation: {n_steps} steps")

    for n in range(n_steps):
        # --- A. Update Scalars (Mass & Energy) ---

        # Fluxes using OLD velocities
        f_m1 = upwind_flux(u1, m1)
        f_m2 = upwind_flux(u2, m2)
        f_E1 = upwind_flux(u1, E1_vol)
        f_E2 = upwind_flux(u2, E2_vol)

        # Determine Dv, De at Cell Centers (Updated Definition)
        # Using macroscopic densities m1, m2
        # denom_D = (m1 + m2) * tau_D
        # Dv_c = (m1 * m2) / (denom_D + 1e-20)

        # denom_T = (m1 * Cv1 + m2 * Cv2) * tau_T
        # De_c = (m1 * Cv1 * m2 * Cv2) / (denom_T + 1e-20)

        denom_D = (alpha1*rho1 + alpha2*rho2) * tau_D
        Dv_c = (alpha1*rho1 * alpha2*rho2) / (denom_D + EPS)

        denom_T = (alpha1*rho1 * Cv1 + alpha2*rho2 * Cv2) * tau_T
        De_c = (alpha1*rho1 * Cv1 * alpha2*rho2 * Cv2) / (denom_T + EPS)


        # Interpolate velocities to centers for source terms
        u1_c = 0.5 * (u1[:-1] + u1[1:])
        u2_c = 0.5 * (u2[:-1] + u2[1:])

        # Recover Temps for source calc
        e1_int = E1_vol - 0.5 * m1 * u1_c**2
        e2_int = E2_vol - 0.5 * m2 * u2_c**2

        e1_int = np.maximum(e1_int, 1e-5)
        e2_int = np.maximum(e2_int, 1e-5)

        T1_loc = e1_int / (m1 * Cv1)
        T2_loc = e2_int / (m2 * Cv2)

        # Heat exchange
        Q = De_c * (T2_loc - T1_loc)

        # Drag dissipation
        rel_u = u2_c - u1_c
        drag_work_1 = 0.5 * Dv_c * rel_u**2 + u1_c * Dv_c * rel_u
        drag_work_2 = 0.5 * Dv_c * rel_u**2 + u2_c * Dv_c * (-rel_u)

        # Update Mass
        m1 -= (dt/dx) * (f_m1[1:] - f_m1[:-1])
        m2 -= (dt/dx) * (f_m2[1:] - f_m2[:-1])

        # Update Energy (Convection + Source)
        E1_vol -= (dt/dx) * (f_E1[1:] - f_E1[:-1])
        E1_vol += dt * (Q + drag_work_1)

        E2_vol -= (dt/dx) * (f_E2[1:] - f_E2[:-1])
        E2_vol += dt * (-Q + drag_work_2)

        # --- Recover Primitive Variables ---
        # Temperature
        e1_int_new = E1_vol - 0.5 * m1 * u1_c**2
        e2_int_new = E2_vol - 0.5 * m2 * u2_c**2
        e1_int_new = np.maximum(e1_int_new, 1e-5)
        e2_int_new = np.maximum(e2_int_new, 1e-5)

        T1 = e1_int_new / (m1 * Cv1)
        T2 = e2_int_new / (m2 * Cv2)

        # Pressure & Alpha (Algebraic Closure)
        # alpha1 = (m1 R1 T1) / (m1 R1 T1 + m2 R2 T2)
        A = m1 * R1 * T1
        B = m2 * R2 * T2
        alpha1 = A / (A + B + 1e-20)
        alpha2 = 1.0 - alpha1

        p = A / (alpha1 + 1e-20)

        # --- B. Update Momentum (Staggered) ---
        grad_p = np.zeros(N+1)
        for i in range(1, N):
            grad_p[i] = (p[i] - p[i-1]) / dx

        # Interpolate m, alpha to faces
        m1_f = np.zeros(N+1)
        m2_f = np.zeros(N+1)
        alpha1_f = np.zeros(N+1)

        m1_f[1:N] = 0.5 * (m1[:-1] + m1[1:])
        m2_f[1:N] = 0.5 * (m2[:-1] + m2[1:])
        alpha1_f[1:N] = 0.5 * (alpha1[:-1] + alpha1[1:])
        alpha2_f = 1.0 - alpha1_f

        # Dv at faces (Updated Definition)
        denom_D_f = (m1_f + m2_f) * tau_D
        Dv_f = (m1_f * m2_f) / (denom_D_f + 1e-20)

        # Update Inner Faces
        for i in range(1, N):
            # Advection (Upwind)
            if u1[i] > 0: du1 = (u1[i] - u1[i-1])/dx
            else: du1 = (u1[i+1] - u1[i])/dx

            if u2[i] > 0: du2 = (u2[i] - u2[i-1])/dx
            else: du2 = (u2[i+1] - u2[i])/dx

            # Forces
            force1 = (alpha1_f[i] / (m1_f[i] + 1e-10)) * grad_p[i]
            force2 = (alpha2_f[i] / (m2_f[i] + 1e-10)) * grad_p[i]

            # Drag Force
            drag1 = (Dv_f[i] / (m1_f[i] + 1e-10)) * (u2[i] - u1[i])
            drag2 = (Dv_f[i] / (m2_f[i] + 1e-10)) * (u1[i] - u2[i])

            u1[i] += dt * (-u1[i]*du1 - force1 + drag1)
            u2[i] += dt * (-u2[i]*du2 - force2 + drag2)

        # BCs
        u1[0] = 0; u1[N] = 0
        u2[0] = 0; u2[N] = 0

        if n % 1000 == 0:
            print(f"Step {n}/{n_steps}")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 8))

    # Pressure
    plt.subplot(2, 2, 1)
    plt.plot(x_c, p/1e6, 'k-')
    plt.title('Pressure (MPa)')
    plt.xlabel('x (m)')
    plt.grid(True)

    # Alpha
    plt.subplot(2, 2, 2)
    plt.plot(x_c, alpha1, 'b-', label='Alpha 1 (Gas)')
    plt.plot(x_c, alpha2, 'r--', label='Alpha 2')
    plt.title('Volume Fraction')
    plt.legend()
    plt.grid(True)

    # Velocity
    plt.subplot(2, 2, 3)
    u1_p = 0.5*(u1[:-1]+u1[1:])
    u2_p = 0.5*(u2[:-1]+u2[1:])
    plt.plot(x_c, u1_p, 'b-', label='u1')
    plt.plot(x_c, u2_p, 'r--', label='u2')
    plt.title('Velocity (m/s)')
    plt.legend()
    plt.grid(True)

    # Temperature
    plt.subplot(2, 2, 4)
    plt.plot(x_c, T1, 'b-', label='T1')
    plt.plot(x_c, T2, 'r--', label='T2')
    plt.title('Temperature (K)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'TF_Multiphase_Shock_Tube_Results_tau_D_{tau_D}_tau_T_{tau_T}.png')
    plt.show()

if __name__ == "__main__":
    solve_updated_multiphase_shock_tube()
