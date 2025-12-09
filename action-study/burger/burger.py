import numpy as np
import matplotlib.pyplot as plt

def burgers_equation_solver():
    # --- Parameters ---
    L = 10.0            # Domain limit [-L, L]
    dx = 0.02           # Spatial step
    dt = 0.005          # Time step
    #t_max = 2.5         # Final time to simulate
    t_max = 3.5         # Final time to simulate
    # Create grid
    x = np.linspace(-L, L, int(2*L/dx) + 1)

    # --- Initial Condition ---
    # We use the decreasing function to ensure a shock forms
    # u goes from 2 (left) to 0 (right)
    u = 1 - (2 / np.pi) * np.arctan(x)

    # Store solutions for plotting at specific intervals
    plot_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5,3.0,3.5]
    solutions = []

    # --- Solver Loop (Upwind Scheme) ---
    t = 0.0
    current_u = u.copy()

    # Save t=0 state
    solutions.append((0.0, current_u.copy()))
    next_plot_idx = 1

    while t < t_max:
        # Calculate u_next using First-Order Upwind:
        # u_new[i] = u[i] - (dt/dx) * u[i] * (u[i] - u[i-1])
        # We use numpy array slicing for speed
        # u[1:] corresponds to index i, u[:-1] corresponds to index i-1

        # Check CFL condition for stability
        cfl = np.max(np.abs(current_u)) * dt / dx
        if cfl > 1.0:
            print("Warning: Unstable CFL condition!")
            break

        # Update step
        u_next = current_u.copy()
        u_next[1:] = current_u[1:] - (dt / dx) * current_u[1:] * (current_u[1:] - current_u[:-1])

        # Enforce Boundary Conditions (Clamped)
        u_next[0] = 2.0  # Left
        u_next[-1] = 0.0 # Right

        current_u = u_next
        t += dt

        # Save output if we reached a plot time
        if next_plot_idx < len(plot_times) and t >= plot_times[next_plot_idx]:
            solutions.append((t, current_u.copy()))
            next_plot_idx += 1

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))

    for i, (time, sol) in enumerate(solutions):
        plt.plot(x, sol, label=f't = {time:.1f}s', color=colors[i], linewidth=2)

    plt.title("Viscous-like Shock Formation (Inviscid Burgers' Equation)")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.xlim(-5, 5) # Zoom in on the center
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    burgers_equation_solver()
