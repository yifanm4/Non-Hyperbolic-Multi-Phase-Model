import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(x, t, nu):
    """
    The exact solution from the image:
    u(x,t) = 2 / (1 + exp((x - c*t) / nu))
    where c = 1 (for u_left=2, u_right=0)
    """
    # We use (x - t) because c = 1
    exponent = (x - t) / nu

    # Numerical stability: exp(700) overflows
    exponent = np.clip(exponent, -50, 50)

    return 2.0 / (1.0 + np.exp(exponent))

def get_rhs_advection_nonperiodic(u, dx, dt):
    """
    Advection update using First-Order Upwind for stability at boundaries.
    (Simpler than MUSCL for handling non-periodic boundaries in a short script,
    but still captures the shock location correctly).
    """
    # Pad to handle boundaries (Edge replication)
    # Left boundary u[-1] is effectively u[0] (Dirichlet fixed later)
    u_padded = np.pad(u, (1, 1), mode='edge')

    # Flux F = 0.5 * u^2
    F = 0.5 * u_padded**2

    # Flux at interface i-1/2 (Upwind: flow is positive, so take left value)
    # F_{i-1/2} comes from index i-1 in padded array
    # F_{i+1/2} comes from index i   in padded array

    F_left  = F[:-2]  # F_{i-1/2}
    F_right = F[1:-1] # F_{i+1/2}

    return - (F_right - F_left) / dx

def get_rhs_diffusion_nonperiodic(u, dx, nu):
    """Central difference diffusion with fixed boundaries."""
    u_padded = np.pad(u, (1, 1), mode='edge')

    # u_{i+1} - 2u_i + u_{i-1}
    u_xx = (u_padded[2:] - 2*u + u_padded[:-2]) / (dx**2)

    return nu * u_xx

def solve_traveling_wave():
    # --- 1. Parameters ---
    nu = 0.1             # Viscosity (Controls shock thickness)
    L_min, L_max = -5.0, 15.0 # Domain
    N = 200
    T_max = 8.0          # Time to let the wave propagate

    dx = (L_max - L_min) / N
    x = np.linspace(L_min, L_max, N)

    # --- 2. Initial Condition (t=0) ---
    # We start with the exact profile at t=0
    u = analytical_solution(x, 0.0, nu)
    u_initial = u.copy()

    t = 0.0

    # --- 3. Time Loop ---
    while t < T_max:
        # Dynamic Time Step
        max_u = np.max(np.abs(u))
        dt_adv = 0.5 * dx / (max_u + 1e-6)
        dt_diff = 0.4 * (dx**2) / nu
        dt = min(dt_adv, dt_diff)

        if t + dt > T_max: dt = T_max - t

        # --- Update Steps ---
        rhs_adv = get_rhs_advection_nonperiodic(u, dx, dt)
        rhs_diff = get_rhs_diffusion_nonperiodic(u, dx, nu)

        u = u + dt * (rhs_adv + rhs_diff)

        # --- Enforce Boundary Conditions (Dirichlet) ---
        # Left side is 2.0, Right side is 0.0
        u[0] = 2.0
        u[-1] = 0.0

        t += dt

    # --- 4. Comparison ---
    u_exact_final = analytical_solution(x, T_max, nu)

    plt.figure(figsize=(10, 6))

    # Plot Initial State
    plt.plot(x, u_initial, 'k--', alpha=0.3, label='Initial Condition (t=0)')

    # Plot Simulation
    plt.plot(x, u, 'r-', linewidth=4, alpha=0.6, label=f'Numerical (t={T_max})')

    # Plot Analytical
    plt.plot(x, u_exact_final, 'k:', linewidth=2, label='Analytical Eq. (Image)')

    plt.title(f'Viscous Burgers: Traveling Wave Comparison (nu={nu})')
    plt.xlabel('Position x')
    plt.ylabel('Velocity u')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    solve_traveling_wave()
