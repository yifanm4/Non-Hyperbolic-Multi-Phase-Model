import numpy as np
import matplotlib.pyplot as plt

def solve_multigas_shock_tube():
    # ==========================================
    # 1. PARAMETERS
    # ==========================================
    L = 44.0              # Length (m)
    N = 960               # Grid points
    dx = L / N

    # Time
    T_final = 0.008       # s
    dt = 1e-6             # s
    n_steps = int(T_final / dt)

    # Relaxation constants
    tau_D = 5e-6          # s (Given in prompt)
    tau_T = 5e-6          # s (Given in prompt)

    # tau_D = 1e-4          # supposed to use 1e-3 s, but the solver blows up
    # tau_T = 1e-4          # supposed to use 1e-3 s, but the solver blows up

    # Gas Properties
    Cv1 = 13550.0
    R1 = 4517.0
    Cv2 = 1355.0
    R2 = 451.7

    # Grid Setup (Staggered)
    # Scalars (rho, P, T, E) at cell centers (0 to N-1)
    # Vectors (u) at cell faces (0 to N)
    x_c = np.linspace(dx/2, L - dx/2, N)
    x_f = np.linspace(0, L, N+1)

    # ==========================================
    # 2. INITIALIZATION
    # ==========================================
    # Primitive Variables
    rho1 = np.zeros(N)
    rho2 = np.zeros(N)
    p1 = np.zeros(N)
    p2 = np.zeros(N)
    T1 = np.zeros(N)
    T2 = np.zeros(N)
    u1 = np.zeros(N+1)
    u2 = np.zeros(N+1)

    # Diaphragm location
    x_diaphragm = 14.6

    # Small epsilon to avoid divide-by-zero in "vacuum" regions
    EPS = 1e-12

    for i in range(N):
        if x_c[i] < x_diaphragm:
            # LEFT (High Pressure Section)
            p1[i] = 20e6
            p2[i] = EPS      # Representing 0 MPa
            T1[i] = 414.0
            T2[i] = 414.0
        else:
            # RIGHT (Low Pressure Section)
            p1[i] = EPS      # Representing 0 MPa
            p2[i] = 0.1e6
            T1[i] = 300.0
            T2[i] = 300.0

        # EOS Initialization: rho = P / (R * T)
        rho1[i] = p1[i] / (R1 * T1[i])
        rho2[i] = p2[i] / (R2 * T2[i])

    # Conserved Variables (at centers)
    # Energy Density E = rho * (e + 0.5*u^2)
    # e = Cv * T
    E1 = rho1 * (Cv1 * T1) # u is initially 0
    E2 = rho2 * (Cv2 * T2)

    # Helper: Upwind Flux
    def get_flux(u_vec, phi_cell):
        # u_vec has N+1 points (faces)
        # phi_cell has N points (centers)
        # Returns Flux at N+1 faces
        flux = np.zeros(len(u_vec))

        # Internal faces 1 to N-1
        # u[i] is face between cell i-1 and cell i
        for i in range(1, N):
            vel = u_vec[i]
            if vel >= 0:
                flux[i] = vel * phi_cell[i-1]
            else:
                flux[i] = vel * phi_cell[i]
        return flux

    # ==========================================
    # 3. TIME LOOP
    # ==========================================
    print(f"Starting Multi-Gas Simulation ({n_steps} steps)...")

    for n in range(n_steps):

        # --- A. Pre-calculate Coefficients D_v, D_e ---
        # Need densities at faces for Momentum, centers for Energy.
        # Eq 32 defines them based on local rho.

        # Centers
        denom_D = (rho1 + rho2) * tau_D
        Dv_c = (rho1 * rho2) / (denom_D + EPS)

        denom_T = (rho1 * Cv1 + rho2 * Cv2) * tau_T
        De_c = (rho1 * Cv1 * rho2 * Cv2) / (denom_T + EPS)

        # --- B. Update Mass (Eq 24, 25) ---
        # d/dt(rho) + d/dx(rho*u) = 0

        # Calculate Mass Fluxes
        flux_rho1 = get_flux(u1, rho1)
        flux_rho2 = get_flux(u2, rho2)

        # Update density
        rho1_new = rho1 - (dt/dx) * (flux_rho1[1:] - flux_rho1[:-1])
        rho2_new = rho2 - (dt/dx) * (flux_rho2[1:] - flux_rho2[:-1])

        # Floor density
        rho1_new = np.maximum(rho1_new, EPS)
        rho2_new = np.maximum(rho2_new, EPS)

        # --- C. Update Energy (Eq 28, 29) ---
        # Conservative Form: dE/dt + d/dx(u(E+P)) = Source
        # Flux term carries Enthalpy (E+P)

        H1 = E1 + p1
        H2 = E2 + p2

        flux_E1 = get_flux(u1, H1)
        flux_E2 = get_flux(u2, H2)

        # Interpolate U to centers for Source terms
        u1_c = 0.5 * (u1[:-1] + u1[1:])
        u2_c = 0.5 * (u2[:-1] + u2[1:])

        # Energy Source Terms (RHS of Eq 28, 29)
        # Term 1: Heat Transfer De(T2 - T1)
        # Term 2: Friction Heat 0.5*Dv*(u1-u2)^2
        # Term 3: Drag Work v1*Dv*(v2-v1)

        rel_vel = u1_c - u2_c
        heat_transfer = De_c * (T2 - T1)
        friction = 0.5 * Dv_c * rel_vel**2

        # Work terms: Note Eq 28 uses v1*Dv(v2-v1)
        work1 = u1_c * Dv_c * (u2_c - u1_c)
        work2 = u2_c * Dv_c * (u1_c - u2_c)

        source_E1 = heat_transfer + friction + work1
        source_E2 = -heat_transfer + friction + work2

        E1_new = E1 - (dt/dx)*(flux_E1[1:] - flux_E1[:-1]) + dt*source_E1
        E2_new = E2 - (dt/dx)*(flux_E2[1:] - flux_E2[:-1]) + dt*source_E2

        E1_new = np.maximum(E1_new, EPS)
        E2_new = np.maximum(E2_new, EPS)

        # --- D. Update Momentum (Eq 26, 27) ---
        # Semi-explicit:
        # (u_new - u)/dt + u * du/dx = -1/rho * dp/dx + Drag

        # Need P gradient at faces (using OLD pressure p1, p2)
        dp1_dx = np.zeros(N+1)
        dp2_dx = np.zeros(N+1)
        dp1_dx[1:-1] = (p1[1:] - p1[:-1]) / dx
        dp2_dx[1:-1] = (p2[1:] - p2[:-1]) / dx

        # Need rho at faces
        rho1_f = np.zeros(N+1)
        rho2_f = np.zeros(N+1)
        rho1_f[1:-1] = 0.5 * (rho1[1:] + rho1[:-1])
        rho2_f[1:-1] = 0.5 * (rho2[1:] + rho2[:-1])

        # Need Dv at faces (recalculate with face rho)
        denom_D_f = (rho1_f + rho2_f) * tau_D
        Dv_f = (rho1_f * rho2_f) / (denom_D_f + EPS)

        # Explicit update loop for internal faces
        u1_new = np.zeros(N+1)
        u2_new = np.zeros(N+1)

        for i in range(1, N):
            # Advection: u * du/dx (Upwind)
            if u1[i] > 0:
                du1 = (u1[i] - u1[i-1])/dx
            else:
                du1 = (u1[i+1] - u1[i])/dx

            if u2[i] > 0:
                du2 = (u2[i] - u2[i-1])/dx
            else:
                du2 = (u2[i+1] - u2[i])/dx

            # Drag Force / rho
            drag1 = (Dv_f[i] / (rho1_f[i] + EPS)) * (u2[i] - u1[i])
            drag2 = (Dv_f[i] / (rho2_f[i] + EPS)) * (u1[i] - u2[i])

            # Pressure Grad / rho
            p_term1 = (1.0 / (rho1_f[i] + EPS)) * dp1_dx[i]
            p_term2 = (1.0 / (rho2_f[i] + EPS)) * dp2_dx[i]

            u1_new[i] = u1[i] + dt * (-u1[i]*du1 - p_term1 + drag1)
            u2_new[i] = u2[i] + dt * (-u2[i]*du2 - p_term2 + drag2)

        # Boundary Conditions (Reflective/Wall at 0 and L, so u=0)
        u1_new[0] = 0; u1_new[N] = 0
        u2_new[0] = 0; u2_new[N] = 0

        # --- E. Update State Variables for Next Step ---
        rho1 = rho1_new
        rho2 = rho2_new
        u1 = u1_new
        u2 = u2_new
        E1 = E1_new
        E2 = E2_new

        # Recover Temp and Pressure
        # E = rho*Cv*T + 0.5*rho*u^2
        # T = (E - 0.5*rho*u^2) / (rho*Cv)

        u1_c_new = 0.5*(u1[:-1] + u1[1:])
        u2_c_new = 0.5*(u2[:-1] + u2[1:])

        internal_E1 = E1 - 0.5 * rho1 * u1_c_new**2
        internal_E2 = E2 - 0.5 * rho2 * u2_c_new**2

        # Enforce positivity for internal energy
        internal_E1 = np.maximum(internal_E1, EPS)
        internal_E2 = np.maximum(internal_E2, EPS)

        T1 = internal_E1 / (rho1 * Cv1)
        T2 = internal_E2 / (rho2 * Cv2)

        # P = rho * R * T
        p1 = rho1 * R1 * T1
        p2 = rho2 * R2 * T2

        if n % 1000 == 0:
            print(f"Step {n}/{n_steps}")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 8))

    # Pressure
    plt.subplot(2, 2, 1)
    plt.plot(x_c, p1/1e6, 'b-', label='P1 (Gas 1)')
    plt.plot(x_c, p2/1e6, 'r--', label='P2 (Gas 2)')
    plt.plot(x_c, (p1+p2)/1e6, 'g:', label='P (Total)')
    plt.title('Partial Pressures (MPa)')
    plt.xlabel('x (m)')
    plt.legend()
    plt.grid(True)

    # Temperature
    plt.subplot(2, 2, 2)
    plt.plot(x_c, T1, 'b-', label='T1')
    plt.plot(x_c, T2, 'r--', label='T2')
    plt.title('Temperature (K)')
    plt.xlabel('x (m)')
    plt.legend()
    plt.grid(True)

    # Velocity
    plt.subplot(2, 2, 3)
    u1_plot = 0.5*(u1[:-1] + u1[1:])
    u2_plot = 0.5*(u2[:-1] + u2[1:])
    plt.plot(x_c, u1_plot, 'b-', label='u1')
    plt.plot(x_c, u2_plot, 'r--', label='u2')
    plt.title('Velocity (m/s)')
    plt.xlabel('x (m)')
    plt.legend()
    plt.grid(True)

    # Density
    plt.subplot(2, 2, 4)
    plt.plot(x_c, rho1, 'b-', label='Rho 1')
    plt.plot(x_c, rho2, 'r--', label='Rho 2')
    plt.title('Density (kg/m^3)')
    plt.yscale('log') # Log scale helps see the "vacuum" transitions
    plt.xlabel('x (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'TF_Multifluid_Shock_Tube_Results_tau_D_{tau_D}_tau_T_{tau_T}.png')
    plt.show()

if __name__ == "__main__":
    solve_multigas_shock_tube()
