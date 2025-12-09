import numpy as np
import matplotlib.pyplot as plt

def run_flux_comparison():
    # --- Configuration ---
    N = 100               # Number of cells
    L = 1.0               # Domain length
    dx = L / N
    c = 1.0               # Advection velocity
    CFL = 0.4             # Courant number
    dt = CFL * dx / c
    t_final = 1.0         # Run for exactly one full loop (periodic)

    # Grid
    x = np.linspace(dx/2, L-dx/2, N)

    # Initial Condition: Square Wave
    # Value is 1.0 between x=0.2 and x=0.4, else 0.0
    u_init = np.zeros(N)
    u_init[(x >= 0.2) & (x <= 0.4)] = 1.0

    # --- Solvers ---

    def solve_advection(scheme_name):
        u = u_init.copy()
        t = 0.0

        while t < t_final:
            if t + dt > t_final: current_dt = t_final - t
            else: current_dt = dt

            u_new = np.zeros_like(u)
            flux = np.zeros(N + 1) # Faces 0 to N

            # 1. Compute Fluxes at Faces
            for i in range(N + 1):
                # Indices for Periodic BCs
                idx_i   = i % N          # Upstream (Donor)
                idx_im1 = (i - 1) % N    # Upstream - 1
                idx_ip1 = (i + 1) % N    # Downstream

                # --- A. 1st Order Upwind ---
                if scheme_name == 'Upwind':
                    # f = u_i
                    phi_face = u[idx_im1] # Face i is between i-1 and i?
                    # Let's align: Flux[i] is at left face of cell i.
                    # Velocity > 0. Upstream is i-1.
                    phi_face = u[idx_im1]

                # --- B. 3rd Order QUICK ---
                elif scheme_name == 'QUICK':
                    # Needs 2 Upstream (i-1, i-2) and 1 Downstream (i) relative to face
                    # Face i is between Cell i-1 and Cell i.
                    # Upstream 1: Cell i-1
                    # Upstream 2: Cell i-2
                    # Downstream: Cell i

                    val_D   = u[idx_i]          # Downstream
                    val_U   = u[idx_im1]        # Upstream 1
                    val_U2  = u[(i - 2) % N]    # Upstream 2

                    # Standard QUICK Formula
                    phi_face = (1/8) * (3*val_D + 6*val_U - 1*val_U2)

                # --- C. 2nd Order MUSCL (with Superbee Limiter) ---
                elif scheme_name == 'MUSCL':
                    # Reconstruct values at face from Upstream (i-1)
                    # u_face = u_{i-1} + 0.5 * phi(r) * (u_{i-1} - u_{i-2})

                    idx_U  = idx_im1
                    idx_U2 = (i - 2) % N
                    idx_D  = idx_i

                    dq_up   = u[idx_U] - u[idx_U2]
                    dq_down = u[idx_D] - u[idx_U]

                    # Ratio of slopes (r)
                    if abs(dq_up) < 1e-10: r = 0.0
                    else: r = dq_down / dq_up

                    # Superbee Limiter (Aggressive compressive limiter)
                    # phi = max(0, min(1, 2r), min(2, r))
                    phi = max(0.0, min(1.0, 2.0*r), min(2.0, r))

                    # Minmod (softer, diffuses more)
                    # phi = max(0.0, min(1.0, r))

                    phi_face = u[idx_U] + 0.5 * phi * dq_up

                flux[i] = c * phi_face

            # 2. Update Cells (Finite Volume)
            # u_new = u - dt/dx * (Flux_right - Flux_left)
            # Right face of i is i+1. Left face is i.
            for i in range(N):
                u_new[i] = u[i] - (current_dt / dx) * (flux[i+1] - flux[i])

            u = u_new
            t += current_dt

        return u

    # --- Run Simulations ---
    res_upwind = solve_advection('Upwind')
    res_quick  = solve_advection('QUICK')
    res_muscl  = solve_advection('MUSCL')

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Exact Solution (Same as Initial for 1 full loop)
    plt.plot(x, u_init, 'k-', linewidth=1.5, label='Exact (Square Wave)')

    # 1. Upwind
    plt.plot(x, res_upwind, 'b-o', markersize=4, alpha=0.6, label='1st Order Upwind')

    # 2. QUICK
    plt.plot(x, res_quick, 'g-s', markersize=4, alpha=0.6, label='3rd Order QUICK')

    # 3. MUSCL
    plt.plot(x, res_muscl, 'r-^', markersize=4, alpha=0.8, label='2nd Order MUSCL (Superbee)')

    plt.title(f"Comparison of Numerical Flux Schemes (N={N}, 1 Period)", fontsize=14)
    plt.xlabel("Position")
    plt.ylabel("Scalar Value")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.2, 1.4)

    # Add Text Annotations
    plt.text(0.6, 0.8, "Upwind: Smeared (Diffusive)", color='blue', fontsize=10)
    plt.text(0.6, 0.7, "QUICK: Wiggles (Dispersive)", color='green', fontsize=10)
    plt.text(0.6, 0.6, "MUSCL: Sharp & Clean", color='red', fontsize=10)

    plt.show()

if __name__ == "__main__":
    run_flux_comparison()
