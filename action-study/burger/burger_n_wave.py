import numpy as np
import matplotlib.pyplot as plt

def get_tau(t0, Re0):
    """Calculate the time constant tau from the image formula."""
    # tau = t0 * sqrt(exp(Re0) - 1)
    # We clip exp to avoid overflow if Re0 is large
    val = np.exp(Re0) - 1.0
    return t0 * np.sqrt(val)

def get_Re_t(t, t0, tau):
    """Calculate the time-varying Reynolds number Re(t)."""
    # Re(t) = ln(1 + sqrt(tau/t))
    inner = 1.0 + np.sqrt(tau / t)
    return np.log(inner)

def analytical_n_wave(x, t, nu, t0, Re0):
    """
    Exact solution from the image:
    u = (x/t) / [ 1 + (sqrt(t/t0)/(exp(Re0)-1)) * exp(...) ]
    """
    if t <= 0: return np.zeros_like(x)

    # 1. Calculate constants
    tau = get_tau(t0, Re0)
    Ret = get_Re_t(t, t0, tau)
    exp_Re0_minus_1 = np.exp(Re0) - 1.0

    # 2. Calculate the exponent argument: - (Re(t) * x^2) / (4 * nu * Re0 * t)
    # Note: The image has Re(t) in the numerator of the exponent
    exponent = - (Ret * x**2) / (4.0 * nu * Re0 * t)

    # 3. Calculate the denominator term
    # term = (1 / (exp(Re0)-1)) * sqrt(t/t0) * exp(exponent)
    prefactor = (1.0 / exp_Re0_minus_1) * np.sqrt(t / t0)
    denominator_term = prefactor * np.exp(exponent)

    # 4. Final calculation
    # u = (x/t) / (1 + denominator_term)
    u = (x / t) / (1.0 + denominator_term)

    return u

def solve_n_wave():
    # --- 1. Parameters ---
    nu = 0.05
    Re0 = 5.0          # Initial Reynolds number (Moderate value)
    t0 = 1.0           # Start time
    t_final = 10.0      # End time

    L = 15.0
    N = 300
    dx = L / N
    x = np.linspace(0, L, N) # Domain [0, L]

    # --- 2. Initial Condition (at t = t0) ---
    u = analytical_n_wave(x, t0, nu, t0, Re0)
    u_initial = u.copy()

    # Store history for plotting
    t = t0

    # --- 3. Numerical Solver Loop ---
    # We use a simple loop with Upwind Advection + Central Diffusion
    while t < t_final:
        # Dynamic Time Step
        max_u = np.max(np.abs(u))
        dt_adv = 0.5 * dx / (max_u + 1e-6)
        dt_diff = 0.4 * (dx**2) / nu
        dt = min(dt_adv, dt_diff)

        if t + dt > t_final: dt = t_final - t

        # --- A. Advection (Upwind) ---
        # u > 0 always for this solution, so simple upwind works
        F = 0.5 * u**2
        rhs_adv = - (F - np.roll(F, 1)) / dx
        # Fix boundary flux at x=0 (u=0 implies F=0)
        # The roll wraps u[-1] to u[0], which is 0 anyway if BC is 0.

        # --- B. Diffusion (Central) ---
        u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
        rhs_diff = nu * u_xx

        # Update
        u = u + dt * (rhs_adv + rhs_diff)

        # --- C. Boundary Conditions (Dirichlet) ---
        u[0] = 0.0   # Fixed at 0
        u[-1] = 0.0  # Fixed at 0 (far field)

        t += dt

    # --- 4. Validation ---
    u_exact_final = analytical_n_wave(x, t_final, nu, t0, Re0)

    # --- 5. Plotting ---
    plt.figure(figsize=(10, 6))

    # Initial
    plt.plot(x, u_initial, 'k--', alpha=0.4, label=f'Initial (t={t0})')

    # Numerical
    plt.plot(x, u, 'r-', linewidth=3, alpha=0.7, label=f'Numerical (t={t_final})')

    # Analytical
    plt.plot(x, u_exact_final, 'b:', linewidth=2, label='Analytical N-Wave')

    plt.title(f'N-Wave Solution (Re0={Re0})')
    plt.xlabel('Position x')
    plt.ylabel('Velocity u')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    solve_n_wave()
