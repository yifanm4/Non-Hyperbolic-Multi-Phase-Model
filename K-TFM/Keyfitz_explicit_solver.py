# A Non-hyperbolic conservative two-equation model
# # Check Understanding the Ill-posed Two-fluid Model (Page 24-28) for more details

import numpy as np
import matplotlib.pyplot as plt

# ---------- Helpers: U = [beta, omega] -> primitives ----------
def primitives_from_conservatives(beta, omega, rho_g, rho_l, U_mix, eps=1e-8):
    """
    Map conservative variables (beta, omega) to primitive variables
    alpha1, alpha2, u1, u2 using Keyfitz relations (26)-(28).
    beta, omega may be scalars or numpy arrays.
    """
    beta = np.asarray(beta)
    omega = np.asarray(omega)

    # avoid division by zero in (rho_l - rho_g)
    drho = rho_g - rho_l
    if abs(drho) < eps:
        drho = eps
        print("Warning: rho1 and rho2 are very close!")

    # (26): alpha_g, alpha_l using beta = alpha_g*rho_g + alpha_l*rho_l
    alpha_g = (beta - rho_l) / drho
    alpha_l = 1.0 - alpha_g

    # enforce 0 < alpha_k < 1 for robustness
    alpha_g = np.clip(alpha_g, eps, 1.0 - eps)
    alpha_l = np.clip(alpha_l, eps, 1.0 - eps)

    # recompute beta from clipped alphas (helps avoid weird states)
    beta_eff = alpha_g * rho_g + alpha_l * rho_l
    beta_eff = np.where(np.abs(beta_eff) < eps, eps, beta_eff)

    # (27)-(28): phase velocities
    num_g = U_mix * rho_l + (beta_eff - rho_l) * omega / (rho_g - rho_l)
    num_l = U_mix * rho_g + (beta_eff - rho_g) * omega / (rho_g - rho_l)

    u_g = num_g / beta_eff
    u_l = num_l / beta_eff

    return alpha_g, alpha_l, u_g, u_l


# ---------- Flux, source, waves ----------

def flux_from_state(beta, omega, rho_g, rho_l, U_mix, vel_clip=1e3):
    """
    Conservative flux F(U) using primitive variables and eq. (24):
        F1 = rho_g * alpha_g * u_g + rho_l * alpha_l * u_l
        F2 = 0.5 * (rho_g * u_g^2 - rho_l * u_l^2)
    """
    alpha_g, alpha_l, u_g, u_l = primitives_from_conservatives(
        beta, omega, rho_g, rho_l, U_mix
    )

    # clip velocities to avoid numerical overflow in squaring
    u_g = np.clip(u_g, -vel_clip, vel_clip)
    u_l = np.clip(u_l, -vel_clip, vel_clip)

    F1 = rho_g * alpha_g * u_g + rho_l * alpha_l * u_l
    F2 = 0.5 * (rho_g * u_g**2 - rho_l * u_l**2)
    return np.array([F1, F2])


def source_from_state(beta, omega, rho_g, rho_l, U_mix, g=9.81):
    """
    Source term S(U) from (25) with only gravity in G_k:
        S1 = 0
        S2 = rho1*g/alpha1 - rho2*g/alpha2
    """
    alpha_g, alpha_l, u_g, u_l = primitives_from_conservatives(
        beta, omega, rho_g, rho_l, U_mix
    )
    G1 = rho_g * g
    G2 = rho_l * g
    S1 = np.zeros_like(beta)
    S2 = G1 / alpha_g - G2 / alpha_l
    return np.array([S1, S2])


def U_mix_from_primitives(alpha1, alpha2, u1, u2):
    """
    Compute mixture velocity U_mix from primitive variables.
    U_mix = alpha1 * u1 + alpha2 * u2
    """
    return alpha1 * u1 + alpha2 * u2

def beta_from_primitives(alpha1, alpha2, rho1, rho2):
    """
    Compute mixture density beta from primitive variables.
    beta = alpha1 * rho1 + alpha2 * rho2
    """
    return alpha1 * rho1 + alpha2 * rho2

def omega_from_primitives(rho1,rho2,u1,u2):

    """
    Compute momentum omega from primitive variables.
    omega = rho1*u1-rho2*u2
    """
    return rho1*u1 - rho2*u2


def characteristic_speed(beta, omega, rho_g, rho_l, U_mix, eps=1e-8, lam_clip=1e6):
    """Characteristic speed lambda (real part) per Eq. (31)."""
    beta_safe = np.clip(beta, eps, None)
    drho = rho_g - rho_l
    if abs(drho) < eps:
        drho = np.sign(drho) * eps if drho != 0 else eps
    # Re(lambda) = rho_g*rho_l * [ U*(rho_l - rho_g) + beta*omega ] / (beta^2*(rho_l - rho_g))
    lam_num = rho_g * rho_l * (U_mix * (rho_l - rho_g) + beta_safe * omega)
    lam_den = (beta_safe ** 2) * (rho_l - rho_g)
    lam = lam_num / lam_den
    return float(np.clip(lam, -lam_clip, lam_clip))

def main():
    # Example usage
    rhog = 1.0  # Density of gas phase
    rhol = 1000.0   # Density of liquid phase

    tini = 0.0
    tEnd = 0.5                # Final time, s
    dt = 1e-4  # s (reduced for stability)
    nt = int((tEnd - tini) / dt)  # Number of time steps
    ncells = 1000               # Number of cells
    L = 12.0  # pipe length (m)
    dx = L / ncells   # Step size
    nx = ncells + 1               # Number of points
    x = np.linspace(0, L, nx)

    # parameters initial conditions
    ul_ini = np.zeros(nx)
    ug_ini = np.zeros(nx)
    alphag_ini = np.zeros(nx)
    p_ini = np.zeros(nx)
    gx = np.zeros(nx)

    ul_ini[:] = 10.0  # m/s
    ug_ini[:] = 1e-8  # m/s
    gx[:] = 9.81  # gravity
    alphag_ini[:] = 0.2  # gas void fraction
    p_ini[:] = 1e5  # pa

    U_mix_const = U_mix_from_primitives(alphag_ini, 1 - alphag_ini, ug_ini, ul_ini)
    # keep U spatially uniform (Eq. 19); assume inlet value holds everywhere
    if np.isscalar(U_mix_const):
        U_mix_const = np.full_like(alphag_ini, U_mix_const)
    beta_ini = beta_from_primitives(alphag_ini, 1 - alphag_ini, rhog, rhol)
    omega_ini = omega_from_primitives(rhog, rhol, ug_ini, ul_ini)

    beta_old = beta_ini.copy()
    omega_old = omega_ini.copy()
    beta_new = beta_ini.copy()
    omega_new = omega_ini.copy()

    print_every = max(nt // 20, 1)  # print ~20 times or at least once
    snapshot_times = np.arange(0.1, tEnd + 1e-12, 0.1)
    snap_idx = 0
    snaps_beta, snaps_omega = [], []
    snaps_alphag, snaps_ug, snaps_ul, snaps_umix = [], [], [], []

    # Time-stepping loop
    for step in range(nt):
        if step % print_every == 0 or step == nt - 1:
            print(f"Step {step+1}/{nt}, t = {(step+1)*dt:.6f} s")
        # fixed at inlet, extrapolate at outlet
        beta_old[0] = beta_ini[0]
        omega_old[0] = omega_ini[0]
        beta_old[-1] = 2 * beta_old[-2] - beta_old[-3]
        omega_old[-1] = 2 * omega_old[-2] - omega_old[-3]

        # Interface Rusanov fluxes
        F_hat = np.zeros((2, nx - 1))
        for i in range(nx - 1):
            F_L = flux_from_state(beta_old[i], omega_old[i], rhog, rhol, U_mix_const[i])
            F_R = flux_from_state(beta_old[i + 1], omega_old[i + 1], rhog, rhol, U_mix_const[i + 1])
            lam_L = characteristic_speed(beta_old[i], omega_old[i], rhog, rhol, U_mix_const[i])
            lam_R = characteristic_speed(beta_old[i + 1], omega_old[i + 1], rhog, rhol, U_mix_const[i + 1])
            a = max(abs(lam_L), abs(lam_R))
            a = min(a, 1e6)
            F_hat[:, i] = 0.5 * (F_L + F_R) - 0.5 * a * np.array(
                [beta_old[i + 1] - beta_old[i], omega_old[i + 1] - omega_old[i]]
            )

        for i in range(1, ncells):
            Source = source_from_state(beta_old[i], omega_old[i], rhog, rhol, U_mix_const, gx[i])
            beta_new[i] = beta_old[i] - dt / dx * (F_hat[0, i] - F_hat[0, i - 1]) + dt * Source[0]
            omega_new[i] = omega_old[i] - dt / dx * (F_hat[1, i] - F_hat[1, i - 1]) + dt * Source[1]

        # Enforce boundary conditions on updated fields
        beta_new[0] = beta_ini[0]
        omega_new[0] = omega_ini[0]
        beta_new[-1] = 2 * beta_new[-2] - beta_new[-3]
        omega_new[-1] = 2 * omega_new[-2] - omega_new[-3]

        # Save snapshots at specified times (after update)
        current_time = (step + 1) * dt
        while snap_idx < len(snapshot_times) and current_time + 1e-12 >= snapshot_times[snap_idx]:
            snaps_beta.append(beta_new.copy())
            snaps_omega.append(omega_new.copy())
            a_g, a_l, u_g, u_l = primitives_from_conservatives(beta_new, omega_new, rhog, rhol, U_mix_const)
            snaps_alphag.append(a_g.copy())
            snaps_ug.append(u_g.copy())
            snaps_ul.append(u_l.copy())
            snaps_umix.append(U_mix_from_primitives(a_g, a_l, u_g, u_l).copy())
            snap_idx += 1

        # Update old values for next time step
        beta_old = beta_new.copy()
        omega_old = omega_new.copy()

    # Final primitive variables for plotting (water faucet flow)
    alphag_final, alphal_final, ug_final, ul_final = primitives_from_conservatives(
        beta_old, omega_old, rhog, rhol, U_mix_const
    )
    U_mix_final = U_mix_from_primitives(alphag_final, alphal_final, ug_final, ul_final)

    # Plot conserved variable snapshots
    if snaps_beta:
        plt.figure(figsize=(10, 5))
        for tval, b, o, um in zip(snapshot_times, snaps_beta, snaps_omega, snaps_umix):
            plt.plot(x, b, label=f"beta t={tval:.1f}s", alpha=0.7)
        plt.xlabel("x (m)")
        plt.ylabel("Value")
        plt.title("Conserved variables snapshots (beta)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"conserved_snapshots_beta_{ncells}.png", dpi=200)

    # Plot conserved variable snapshots
    if snaps_beta:
        plt.figure(figsize=(10, 5))
        for tval, b, o, um in zip(snapshot_times, snaps_beta, snaps_omega, snaps_umix):
            plt.plot(x, o, label=f"omega t={tval:.1f}s", linestyle="--", alpha=0.7)
        plt.xlabel("x (m)")
        plt.ylabel("Value")
        plt.title("Conserved variables snapshots (omega)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"conserved_snapshots_omega_{ncells}.png", dpi=200)

    # Plot conserved variable snapshots
    if snaps_beta:
        plt.figure(figsize=(10, 5))
        for tval, b, o, um in zip(snapshot_times, snaps_beta, snaps_omega, snaps_umix):
            plt.plot(x, um, label=f"U_mix t={tval:.1f}s", linestyle=":")
        plt.xlabel("x (m)")
        plt.ylabel("Value")
        plt.title("Conserved variables snapshots (U_mix)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"conserved_snapshots_Umix_{ncells}.png", dpi=200)

    # Plot primitive variable snapshots (water faucet flow)
    if snaps_alphag:
        plt.figure(figsize=(10, 5))
        for tval, ag, ug, ul in zip(snapshot_times, snaps_alphag, snaps_ug, snaps_ul):
            plt.plot(x, ag, label=f"alpha_g t={tval:.1f}s")
        plt.xlabel("x (m)")
        plt.ylabel("Value")
        plt.title("Primitive variables snapshots (alphag)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"primitive_snapshots_alphag_{ncells}.png", dpi=200)

    # Plot primitive variable snapshots (water faucet flow)
    if snaps_alphag:
        plt.figure(figsize=(10, 5))
        for tval, ag, ug, ul in zip(snapshot_times, snaps_alphag, snaps_ug, snaps_ul):
            plt.plot(x, ug, label=f"u_g t={tval:.1f}s", linestyle="--")
        plt.xlabel("x (m)")
        plt.ylabel("Value")
        plt.title("Primitive variables snapshots (ug)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"primitive_snapshots_ug_{ncells}.png", dpi=200)

    # Plot primitive variable snapshots (water faucet flow)
    if snaps_alphag:
        plt.figure(figsize=(10, 5))
        for tval, ag, ug, ul in zip(snapshot_times, snaps_alphag, snaps_ug, snaps_ul):
            plt.plot(x, ul, label=f"u_l t={tval:.1f}s", linestyle=":")
        plt.xlabel("x (m)")
        plt.ylabel("Value")
        plt.title("Primitive variables snapshots (ul)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"primitive_snapshots_ul_{ncells}.png", dpi=200)

    # # Plot conserved variables at final time
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, beta_old, label="beta (mixture density)")
    # plt.plot(x, omega_old, label="omega (momentum)")
    # plt.plot(x, U_mix_final, label="U_mix (mixture velocity)")
    # plt.xlabel("x (m)")
    # plt.ylabel("Value")
    # plt.title("Final conserved variables")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("final_conserved_vars.png", dpi=200)

    # # Plot primitive variables at final time (water faucet flow)
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, alphag_final, label="alpha_g")
    # plt.plot(x, ug_final, label="u_g")
    # plt.plot(x, ul_final, label="u_l")
    # plt.xlabel("x (m)")
    # plt.ylabel("Value")
    # plt.title("Final primitive variables (water faucet flow)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("final_primitive_vars.png", dpi=200)

    plt.show()




if __name__ == "__main__":
  main()
