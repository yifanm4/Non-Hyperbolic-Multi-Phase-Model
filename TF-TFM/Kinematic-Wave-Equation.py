import numpy as np
import matplotlib.pyplot as plt

def simulate_kinematic_wave():
    # Parameters
    Nx = 200          # Grid points
    L = 2.0 * np.pi   # Domain length (0 to 2pi)
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)

    # MODIFICATION 1: Increase run time to 3.0 seconds
    t_max = 3.0
    dt = 0.005        # Small time step for stability

    # Initial Condition: A smooth Sine Wave
    # u(x,0) = 1.0 + 0.5 * sin(x)
    u = 1.0 + 0.5 * np.sin(x)

    # Store snapshots.
    # Initialize with the state at t=0
    snapshots = [np.copy(u)]
    times = [0.0]

    # MODIFICATION 2: Add 3.0 to the output list
    output_times = [1.0, 2.0, 3.0]

    t = 0.0
    u_curr = np.copy(u)
    u_next = np.copy(u)

    # Simulation Loop (Upwind Scheme)
    # Use a small epsilon for float comparison to ensure we hit t_max
    while t < t_max - dt/2:

        # Standard Upwind Discretization
        for i in range(1, Nx):
            # Periodic boundary logic: look at i-1, wrapping -1 to end
            u_prev_x = u_curr[i-1] if i > 0 else u_curr[-1]

            # The Equation: u_t = -u * u_x
            # Flux = u * du/dx
            flux = u_curr[i] * (u_curr[i] - u_prev_x) / dx
            u_next[i] = u_curr[i] - dt * flux

        # Update Boundary (Periodic)
        u_next[0] = u_next[-1]

        # Advance time
        u_curr[:] = u_next[:]
        t += dt

        # Check if we just passed an output time
        for target in output_times:
            if abs(t - target) < dt/2:
                snapshots.append(np.copy(u_curr))
                times.append(t)

    # Plotting
    plt.figure(figsize=(10, 6))

    # MODIFICATION 3: Add color and label for the new t=3.0 case
    colors = ['green', 'orange', 'red', 'purple']
    labels = [
        't=0.0 (Initial Smooth)',
        't=1.0 (Steepening)',
        't=2.0 (Shock Formation)',
        't=3.0 (Shock Propagation)'
    ]

    # Robust Loop: iterates through labels but stops if snapshots ran out
    for k in range(len(labels)):
        if k < len(snapshots):
            plt.plot(x, snapshots[k], color=colors[k], label=labels[k], linewidth=2)

    plt.title('Kinematic Wave Eq: Shock Formation & Propagation')
    plt.xlabel('Position x')
    plt.ylabel('Amplitude u')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.savefig('kinematic_wave_shock_propagation.png')
    plt.show()

if __name__ == "__main__":
    simulate_kinematic_wave()
