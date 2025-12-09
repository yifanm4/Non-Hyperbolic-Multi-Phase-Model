import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def inverted_pendulum(state, t):
    theta, omega = state
    g = 9.81
    L = 1.0
    # Equation of motion: theta'' = (g/L) * sin(theta)
    # Positive feedback loop (instability)
    dydt = [omega, (g/L) * np.sin(theta)]
    return dydt

def simulate_instability_comparison():
    t = np.linspace(0, 10, 1000)

    # --- Initial Conditions ---

    # Case 1: The "Original" Perfect Case
    # Mathematically exact equilibrium. Should stay flat forever.
    y0_Perfect = [0.0, 0.0]

    # Case 2: Butterfly Wing (Microscopic Perturbation)
    # 0.000001 radians difference
    y0_Tiny = [0.00000001, 0.0]

    # Case 3: A Small Nudge
    # 0.0001 radians difference
    y0_Small = [0.000001, 0.0]

    # Solve ODEs
    sol_Perfect = odeint(inverted_pendulum, y0_Perfect, t)
    sol_Tiny    = odeint(inverted_pendulum, y0_Tiny, t)
    sol_Small   = odeint(inverted_pendulum, y0_Small, t)

    # Extract angles
    theta_Perf = sol_Perfect[:, 0]
    theta_Tiny = sol_Tiny[:, 0]
    theta_Small = sol_Small[:, 0]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plotting the trajectories
    plt.plot(t, theta_Perf, label='Original Case (Perfect 0.0)', color='blue', linewidth=3, linestyle='-')
    plt.plot(t, theta_Tiny, label='Tiny Error (1e-8)', color='orange', linewidth=2)
    plt.plot(t, theta_Small, label='Small Error (1e-6)', color='green', linewidth=2, linestyle='--')

    plt.title('Lyapunov Instability: Perfect Equilibrium vs. Reality')
    plt.xlabel('Time (s)')
    plt.ylabel('Deviation Angle (radians)')

    # Draw the "Stable Zone" limit (arbitrary epsilon)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(-0.5, color='gray', linestyle=':', alpha=0.5)
    plt.text(0.5, 0.6, 'Arbitrary "Stable Neighborhood" boundary', color='gray')

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-1, 5) # Limit y-axis to see the divergence clearly
    plt.savefig('Lyapunov_Instability_Comparison_Inverted_Pendulum.png')
    plt.show()

if __name__ == "__main__":
    simulate_instability_comparison()
