import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def analytical_delta_solution(x, t, nu, Re):
    """
    Exact solution for the Delta Function IC from your image.
    Formula:
    u = sqrt(nu / (pi*t)) * [ (exp(Re)-1) * exp(-x^2/4vt) ] / [ 1 + (exp(Re)-1)/2 * erfc(x/sqrt(4vt)) ]
    """
    if t <= 0:
        return np.zeros_like(x) # Avoid singularity

    prefactor = np.sqrt(nu / (np.pi * t))

    # Calculate terms carefully to avoid overflow if Re is huge
    # However, for Re=20, standard floats handle exp(20) fine.
    exp_Re_minus_1 = np.exp(Re) - 1.0

    numerator = exp_Re_minus_1 * np.exp(-x**2 / (4 * nu * t))

    denominator = 1.0 + (exp_Re_minus_1 / 2.0) * erfc(x / np.sqrt(4 * nu * t))

    return prefactor * (numerator / denominator)

def get_rhs_advection_upwind(u, dx):
    """Simple 1st order upwind flux for stability."""
    # F = u^2 / 2
    F = 0.5 * u**2
    # Upwind difference (assuming u > 0 mostly, but handling direction helps)
    # Using simple upwind: (F_i - F_{i-1})/dx
    return - (F - np.roll(F, 1)) / dx

def get_rhs_diffusion(u, dx, nu):
    """Central difference diffusion."""
    return nu * (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)

def solve_delta_release():
    # --- 1. Parameters ---
    nu = 0.1
    Re = 20.0          # High Re to see the "Triangle Shock"
    L = 20.0
    N = 1000            # High resolution
    dx = L / N
    x = np.linspace(-5, 15, N) # Domain shifted to see propagation

    t_start = 0.5      # Start slightly after t=0 to avoid infinity
    t_final = 5.0      # End time

    # --- 2. Initial Condition (at t = t_start) ---
    u = analytical_delta_solution(x, t_start, nu, Re)
    u_initial = u.copy()

    t = t_start

    # --- 3. Time Loop ---
    while t < t_final:
        # Dynamic Time Step
        max_u = np.max(np.abs(u))
        dt_adv = 0.5 * dx / (max_u + 1e-6)
        dt_diff = 0.4 * (dx**2) / nu
        dt = min(dt_adv, dt_diff)

        if t + dt > t_final: dt = t_final - t

        # Update
        rhs_adv = get_rhs_advection_upwind(u, dx)
        rhs_diff = get_rhs_diffusion(u, dx, nu)

        u = u + dt * (rhs_adv + rhs_diff)

        # Dirichlet BCs (u=0 far away)
        u[0] = 0.0
        u[-1] = 0.0

        t += dt

    # --- 4. Plotting ---
    u_exact_final = analytical_delta_solution(x, t_final, nu, Re)

    plt.figure(figsize=(10, 6))

    # Plot Initial Start
    plt.plot(x, u_initial, 'g--', alpha=0.4, label=f'Start Condition (t={t_start})')

    # Plot Final Numerical
    plt.plot(x, u, 'r-', linewidth=3, alpha=0.7, label=f'Numerical Solver (t={t_final})')

    # Plot Final Analytical
    plt.plot(x, u_exact_final, 'k:', linewidth=2, label='Analytical Solution')

    plt.title(f'Viscous Burgers: Delta Function Initial Condition (Re={Re})')
    plt.xlabel('Position x')
    plt.ylabel('Velocity u')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Text annotation for the shape
    plt.text(4, 0.5, "Triangle / N-Wave Shape\n(Characteristic of High Re)", ha='center')

    plt.show()

if __name__ == "__main__":
    solve_delta_release()
