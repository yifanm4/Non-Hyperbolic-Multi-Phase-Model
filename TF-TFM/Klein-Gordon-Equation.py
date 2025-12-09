import numpy as np
import matplotlib.pyplot as plt

def simulate_klein_gordon():
    # Parameters
    L = 20.0          # Domain length
    Nx = 200          # Number of grid points
    dx = L / (Nx - 1) # Grid spacing
    x = np.linspace(0, L, Nx)

    c = 1.0           # Wave speed
    CFL = 0.5         # Courant number (stability for wave part)
    dt = CFL * dx / c # Time step
    T_max = 5.0       # Total simulation time

    # The Instability Parameter
    # The paper states instability occurs if lambda < 0
    lam = -3.0

    # Initialize fields (u_prev, u_curr, u_next)
    u = np.zeros(Nx)
    u_prev = np.zeros(Nx)
    u_next = np.zeros(Nx)

    # Initial Condition: A low-k sine wave
    # k = 2*pi/L approx 0.3.
    # Since k^2 (0.09) < |lambda| (3.0), this mode is UNSTABLE.
    u = 0.1 * np.sin(2 * np.pi * x / L)
    u_prev = np.copy(u) # Assume zero initial velocity for simplicity

    # Store history for plotting
    history = []
    times = [0.0]

    # Time Stepping Loop (Central Difference)
    t = 0.0
    steps = int(T_max / dt)

    for n in range(steps):
        # Finite Difference Scheme
        # u_tt = c^2 * u_xx - c^2 * lambda * u
        # Discretized:
        # (u_next - 2u + u_prev)/dt^2 = c^2*(u_i+1 - 2u + u_i-1)/dx^2 - c^2*lam*u

        for i in range(1, Nx - 1):
            d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
            source_term = - (c**2) * lam * u[i]

            u_next[i] = 2*u[i] - u_prev[i] + (dt**2) * ( (c**2) * d2u_dx2 + source_term )

        # Boundary Conditions (Fixed/Dirichlet)
        u_next[0] = 0
        u_next[-1] = 0

        # Update
        u_prev[:] = u[:]
        u[:] = u_next[:]
        t += dt

        # Save specific time snapshots for plotting
        if n % (steps // 5) == 0:
            history.append(np.copy(u))
            times.append(t)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(history):
        plt.plot(x, data, label=f't = {times[i]:.2f}')

    plt.title(f'Instability of Hyperbolic Klein-Gordon Eq (lambda={lam})')
    plt.xlabel('Position x')
    plt.ylabel('Amplitude u')
    #plt.loglog()
    plt.legend()
    plt.grid(True)
    plt.savefig('Klein_Gordon_Instability.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    simulate_klein_gordon()
