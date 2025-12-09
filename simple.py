import numpy as np
import matplotlib.pyplot as plt

def solve_full_two_fluid_model(
    n_cells=50,
    dt=1e-5,          # Small dt for acoustic stability
    t_final=0.5,
    C_vm=10.0,        # Virtual Mass Coefficient (Stability/Coupling)
    #C_vm=0.0,         # Virtual Mass Coefficient

    C_drag=0.0,       # Interfacial Drag (Usually 0 for this benchmark)
    C_ip=0.0,         # Interfacial pressure strength
    C_E=1.0,          # Entropy viscosity coefficient
    C_max=0.5,        # Maximum viscosity limiter coefficient
    use_muscl=False,  # Toggle MUSCL (second-order upwind) for mass fluxes
    use_quick=False   # Toggle QUICK (third-order upwind) for mass fluxes
):
    # --- 1. CONFIGURATION ---
    L = 12.0
    dx = L / n_cells
    g = 9.81

    # Grid (Staggered)
    # Centers (0..N-1) for P, alpha, rho
    x_c = np.linspace(dx/2, L-dx/2, n_cells)

    # Constants for EOS
    R_gas = 287.0      # Air
    T_iso = 300.0      # Isothermal
    rho_l_ref = 1000.0
    P_ref = 100000.0
    c_l = 1000.0       # Stiff Liquid Sound Speed

    # Boundary / Initial Conditions
    P_out = 100000.0
    alpha_in = 0.2
    ul_in = 10.0
    ug_in = 0.0

    # --- 2. INITIALIZATION ---
    P = np.ones(n_cells) * P_out
    alpha_g = np.ones(n_cells) * alpha_in
    alpha_l = 1.0 - alpha_g

    # Velocities at Faces (0..N)
    ul = np.ones(n_cells + 1) * 10.0
    ug = np.zeros(n_cells + 1)

    # EOS Functions
    def get_rho_g(p): return p / (R_gas * T_iso) # Ideal Gas
    def get_rho_l(p): return rho_l_ref + (p - P_ref)/(c_l**2) # Stiff Gas

    # Entropy viscosity helper (scalar Burgers-like form on velocities)
    def entropy_viscosity(u, u_prev):
        u_f = np.clip(u, -1e6, 1e6)  # guard overflow
        u_prev_f = np.clip(u_prev, -1e6, 1e6)
        E = 0.5 * u_f**2
        E_prev = 0.5 * u_prev_f**2
        dE_dt = (E - E_prev) / dt

        F = (1.0/3.0) * u_f**3
        dF_dx = np.zeros_like(u)
        dF_dx[1:-1] = (F[2:] - F[:-2]) / (2*dx)
        dF_dx[0] = dF_dx[1]
        dF_dx[-1] = dF_dx[-2]

        R = np.abs(dE_dt + dF_dx)
        E_scale = np.max(np.abs(E)) + 1e-8  # avoid zero divide

        nu_e = C_E * dx**2 * R / E_scale
        nu_max = C_max * dx * np.abs(u_f)
        return np.minimum(nu_e, nu_max)

    # MUSCL helper with van Leer limiter
    def van_leer(a, b):
        return (2 * a * b) / (a + b + 1e-16) if a * b > 0 else 0.0

    def limited_slopes(arr):
        slopes = np.zeros_like(arr)
        for j in range(len(arr)):
            dl = arr[j] - arr[j-1] if j > 0 else 0.0
            dr = arr[j+1] - arr[j] if j < len(arr) - 1 else 0.0
            slopes[j] = van_leer(dl, dr)
        return slopes

    def quick_face(phi_ext, face_idx, vel, n_cells_local):
        # face_idx in [0, n_cells]; phi_ext has ghosted size n_cells+2
        if vel > 0 and face_idx >= 2:
            phi_U = phi_ext[face_idx]       # upstream cell i-1
            phi_UU = phi_ext[face_idx - 1]  # upstream of upstream
            phi_UD = phi_ext[face_idx + 1]  # downstream cell
            return (3 * phi_U + 6 * phi_UU - phi_UD) / 8.0
        if vel < 0 and face_idx <= n_cells_local - 2:
            phi_U = phi_ext[face_idx + 1]   # upstream (right) cell
            phi_UU = phi_ext[face_idx + 2]  # upstream of upstream
            phi_UD = phi_ext[face_idx]      # downstream cell
            return (3 * phi_U + 6 * phi_UU - phi_UD) / 8.0
        # fallback to first-order upwind near boundaries
        return phi_ext[face_idx] if vel > 0 else phi_ext[face_idx + 1]

    # Initialize Conserved Mass Densities
    rho_g = get_rho_g(P)
    rho_l = get_rho_l(P)
    Rg = alpha_g * rho_g
    Rl = alpha_l * rho_l

    # --- 3. TIME LOOP ---
    t = 0.0
    step = 0

    print(f"Simulating... (C_vm={C_vm}, C_ip={C_ip}, C_E={C_E}, C_max={C_max})")

    ug_prev = ug.copy()
    ul_prev = ul.copy()

    while t < t_final:
        step += 1
        ug_old = ug.copy()
        ul_old = ul.copy()

        # --- A. MASS CONSERVATION (Update Rg, Rl) ---
        flux_Rg = np.zeros(n_cells + 1)
        flux_Rl = np.zeros(n_cells + 1)

        # Inlet Density (approx based on P[0])
        rho_g_in = get_rho_g(P[0])
        rho_l_in = get_rho_l(P[0])

        slopes_Rg = limited_slopes(Rg) if use_muscl else None
        slopes_Rl = limited_slopes(Rl) if use_muscl else None
        if use_quick:
            Rg_ext = np.zeros(n_cells + 2)
            Rl_ext = np.zeros(n_cells + 2)
            Rg_ext[1:-1] = Rg
            Rl_ext[1:-1] = Rl
            Rg_ext[0] = alpha_in * rho_g_in
            Rl_ext[0] = (1.0 - alpha_in) * rho_l_in
            Rg_ext[-1] = Rg[-1]
            Rl_ext[-1] = Rl[-1]

        # Flux Calculation (Upwind)
        for i in range(n_cells + 1):
            # Gas Flux
            if use_quick:
                val = quick_face(Rg_ext, i, ug[i], n_cells)
            else:
                if i == 0:
                    left = (alpha_in * rho_g_in)
                    right = (Rg[0] - 0.5 * (slopes_Rg[0] if use_muscl else 0.0))
                elif i == n_cells:
                    left = Rg[-1] + 0.5 * (slopes_Rg[-1] if use_muscl else 0.0)
                    right = Rg[-1]
                else:
                    left = Rg[i-1] + 0.5 * (slopes_Rg[i-1] if use_muscl else 0.0)
                    right = Rg[i] - 0.5 * (slopes_Rg[i] if use_muscl else 0.0)
                val = left if ug[i] > 0 else right
            flux_Rg[i] = np.clip(val * ug[i], -1e8, 1e8)

            # Liquid Flux
            if use_quick:
                val_l = quick_face(Rl_ext, i, ul[i], n_cells)
            else:
                if i == 0:
                    left_l = ((1.0-alpha_in) * rho_l_in)
                    right_l = (Rl[0] - 0.5 * (slopes_Rl[0] if use_muscl else 0.0))
                elif i == n_cells:
                    left_l = Rl[-1] + 0.5 * (slopes_Rl[-1] if use_muscl else 0.0)
                    right_l = Rl[-1]
                else:
                    left_l = Rl[i-1] + 0.5 * (slopes_Rl[i-1] if use_muscl else 0.0)
                    right_l = Rl[i] - 0.5 * (slopes_Rl[i] if use_muscl else 0.0)
                val_l = left_l if ul[i] > 0 else right_l
            flux_Rl[i] = np.clip(val_l * ul[i], -1e8, 1e8)

        # Update Partial Densities
        Rg += dt * -(flux_Rg[1:] - flux_Rg[:-1]) / dx
        Rl += dt * -(flux_Rl[1:] - flux_Rl[:-1]) / dx
        Rg = np.clip(Rg, 1e-12, None)
        Rl = np.clip(Rl, 1e-12, None)

        # --- B. PRESSURE SOLVER (Newton-Raphson on EOS) ---
        # Solve P such that Vol_g + Vol_l = 1
        P_iter = P.copy()
        for _ in range(5):
            r_g = get_rho_g(P_iter)
            r_l = get_rho_l(P_iter)

            vol_err = (Rg / r_g) + (Rl / r_l) - 1.0

            # Derivatives
            drg_dP = 1.0/(R_gas*T_iso)
            drl_dP = 1.0/(c_l**2)

            J = -(Rg/r_g**2)*drg_dP - (Rl/r_l**2)*drl_dP

            P_iter = P_iter - vol_err / (J + 1e-16)

        P = P_iter

        # Update Primitives
        rho_g = get_rho_g(P)
        rho_l = get_rho_l(P)
        alpha_g = Rg / rho_g
        alpha_l = Rl / rho_l

        # --- C. MOMENTUM CONSERVATION ---
        # Gradients
        dPdx = np.zeros(n_cells + 1)
        dPdx[1:-1] = (P[1:] - P[:-1]) / dx
        dPdx[-1] = (P_out - P[-1]) / (dx/2) # Outlet BC
        dPdx[0] = dPdx[1] # Inlet Zero-Grad approx

        dadx = np.zeros(n_cells + 1)
        dadx[1:-1] = (alpha_g[1:] - alpha_g[:-1]) / dx

        # Entropy viscosities (computed on current state vs previous step)
        nu_g = entropy_viscosity(ug, ug_prev)
        nu_l = entropy_viscosity(ul, ul_prev)

        # Update Internal Faces
        for i in range(1, n_cells):
            # Bound velocities to tame explosive growth
            ug[i] = np.clip(ug[i], -1e4, 1e4)
            ul[i] = np.clip(ul[i], -1e4, 1e4)

            # 1. Advection (Upwind)
            adv_g = ug[i] * (ug[i]-ug[i-1])/dx if ug[i]>0 else ug[i]*(ug[i+1]-ug[i])/dx
            adv_l = ul[i] * (ul[i]-ul[i-1])/dx if ul[i]>0 else ul[i]*(ul[i+1]-ul[i])/dx

            # 2. Interfacial / Virtual Mass Force
            # F_vm ~ C_vm * rho_mix * d(alpha)/dx
            rho_mix = 0.5 * ((alpha_g[i]*rho_g[i] + alpha_l[i]*rho_l[i]) +
                             (alpha_g[i-1]*rho_g[i-1] + alpha_l[i-1]*rho_l[i-1]))
            F_vm = C_vm * rho_mix * dadx[i]

            # 3. Interfacial pressure term: p_int * grad(alpha_g)
            # p_int scales with slip velocity to capture interface forces
            slip = np.clip(ul[i] - ug[i], -1e4, 1e4)
            p_int = C_ip * rho_mix * np.minimum(slip*slip, 1e8)
            F_ip = p_int * dadx[i]

            # 4. Drag (Simple linear drag: F ~ C * (ul - ug))
            F_drag = C_drag * (ul[i] - ug[i])

            # 5. Artificial viscosity (entropy viscosity) Laplacian
            lap_g = nu_g[i] * (ug[i+1] - 2*ug[i] + ug[i-1]) / (dx**2)
            lap_l = nu_l[i] * (ul[i+1] - 2*ul[i] + ul[i-1]) / (dx**2)

            # 6. Updates
            # Gas Momentum (Light phase!)
            rho_g_f = 0.5*(rho_g[i] + rho_g[i-1])
            acc_g = -adv_g - (1/rho_g_f)*dPdx[i] + (F_ip/rho_g_f) - (F_vm/rho_g_f) + (F_drag/rho_g_f) + lap_g
            ug[i] += dt * acc_g

            # Liquid Momentum
            rho_l_f = 0.5*(rho_l[i] + rho_l[i-1])
            acc_l = g - adv_l - (1/rho_l_f)*dPdx[i] - (F_ip/rho_l_f) + (F_vm/rho_l_f) - (F_drag/rho_l_f) + lap_l
            ul[i] += dt * acc_l

        # BCs
        ul[0], ug[0] = ul_in, ug_in
        ul[-1], ug[-1] = ul[-2], ug[-2]

        # Store previous velocity for next entropy residual
        ug_prev = ug_old
        ul_prev = ul_old

        t += dt

    return x_c, alpha_g, ug, ul, P

if __name__ == "__main__":
    # --- RUN ---
    # First-order vs MUSCL comparison
    nCells = 400
    C_vm = 0.0
    C_ip=0.0
    C_E =2.0
    C_max =1.0
    
    # Reference 1st-order (no entropy viscosity)
    x1, ag1, ug1, ul1, P1 = solve_full_two_fluid_model(
        t_final=0.5, n_cells=nCells, C_vm=C_vm, C_ip=C_ip, C_E=0.0, C_max=0.0, use_muscl=False
    )
    # MUSCL results (no entropy viscosity)
    x2, ag2, ug2, ul2, P2 = solve_full_two_fluid_model(
        t_final=0.5, n_cells=nCells, C_vm=C_vm, C_ip=C_ip, C_E=0.0, C_max=0.0, use_muscl=True
    )
    # MUSCL with entropy viscosity
    x3, ag3, ug3, ul3, P3 = solve_full_two_fluid_model(
        t_final=0.5, n_cells=nCells, C_vm=C_vm, C_ip=C_ip, C_E=C_E, C_max=C_max, use_muscl=True
    )
    # QUICK (third-order upwind) without entropy viscosity
    x4, ag4, ug4, ul4, P4 = solve_full_two_fluid_model(
        t_final=0.5, n_cells=nCells, C_vm=C_vm, C_ip=C_ip, C_E=0.0, C_max=0.0, use_quick=True
    )
    
    
    # --- PLOTTING ---
    # Map face velocities to centers
    ug_c1 = 0.5 * (ug1[1:] + ug1[:-1])
    ul_c1 = 0.5 * (ul1[1:] + ul1[:-1])
    ug_c2 = 0.5 * (ug2[1:] + ug2[:-1])
    ul_c2 = 0.5 * (ul2[1:] + ul2[:-1])
    ug_c3 = 0.5 * (ug3[1:] + ug3[:-1])
    ul_c3 = 0.5 * (ul3[1:] + ul3[:-1])
    ug_c4 = 0.5 * (ug4[1:] + ug4[:-1])
    ul_c4 = 0.5 * (ul4[1:] + ul4[:-1])
    
    # Analytical Liquid V & Alpha
    v_exact = np.sqrt(10**2 + 2*9.81*x1)
    a_exact = 1.0 - (0.8 * 10 / v_exact)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"1D Two-Fluid Model Results (t=0.5s)", fontsize=16)
    
    # 1. Void Fraction
    axs[0,0].plot(x1, ag1, 'b-o', markersize=4, label='1st-order')
    axs[0,0].plot(x2, ag2, 'g-s', markersize=4, label='MUSCL')
    axs[0,0].plot(x3, ag3, 'm-^', markersize=4, label='MUSCL + EV')
    axs[0,0].plot(x4, ag4, 'k-d', markersize=4, label='QUICK')
    axs[0,0].plot(x1, a_exact, 'r--', label='Analytical (Steady)')
    axs[0,0].set_title(r"Void Fraction $\alpha_g$")
    axs[0,0].set_ylabel("Void Fraction")
    axs[0,0].grid(True)
    axs[0,0].legend()
    
    # 2. Pressure
    axs[0,1].plot(x1, P1, 'b-o', markersize=4, label='1st-order')
    axs[0,1].plot(x2, P2, 'g-s', markersize=4, label='MUSCL')
    axs[0,1].plot(x3, P3, 'm-^', markersize=4, label='MUSCL + EV')
    axs[0,1].plot(x4, P4, 'k-d', markersize=4, label='QUICK')
    axs[0,1].set_title("Pressure $P$ (Pa)")
    axs[0,1].set_ylabel("Pressure (Pa)")
    axs[0,1].grid(True)
    axs[0,1].ticklabel_format(useOffset=False) # Show actual values
    
    # 3. Liquid Velocity
    axs[1,0].plot(x1, ul_c1, 'b-o', markersize=4, label='1st-order')
    axs[1,0].plot(x2, ul_c2, 'g-s', markersize=4, label='MUSCL')
    axs[1,0].plot(x3, ul_c3, 'm-^', markersize=4, label='MUSCL + EV')
    axs[1,0].plot(x4, ul_c4, 'k-d', markersize=4, label='QUICK')
    axs[1,0].plot(x1, v_exact, 'r--', label='Analytical')
    axs[1,0].set_title(r"Liquid Velocity $u_l$")
    axs[1,0].set_ylabel("Speed (m/s)")
    axs[1,0].set_xlabel("Position (m)")
    axs[1,0].grid(True)
    axs[1,0].legend()
    
    # 4. Gas Velocity (THE NEW OUTPUT)
    axs[1,1].plot(x1, ug_c1, 'm-o', markersize=4, label=r'Gas $u_g$ 1st')
    axs[1,1].plot(x2, ug_c2, 'c-s', markersize=4, label=r'Gas $u_g$ MUSCL')
    axs[1,1].plot(x3, ug_c3, 'k-^', markersize=4, label=r'Gas $u_g$ MUSCL + EV')
    axs[1,1].plot(x4, ug_c4, 'y-d', markersize=4, label=r'Gas $u_g$ QUICK')
    axs[1,1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[1,1].set_title(r"Gas Velocity $u_g$")
    axs[1,1].set_ylabel("Speed (m/s)")
    axs[1,1].set_xlabel("Position (m)")
    axs[1,1].grid(True)
    axs[1,1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"two_fluid_model_results_nCells_{nCells}_C_vm_{C_vm}_C_ip_{C_ip}_C_E_{C_E}_C_max_{C_max}.png", dpi=300)
    plt.show()
    