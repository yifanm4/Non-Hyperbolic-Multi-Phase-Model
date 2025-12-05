"""
Implicit Newton-Krylov solver for the non-hyperbolic conservative two-equation model.
Solves the same setup as new.py but with backward Euler in time and JFNK for the nonlinear solve.
Generates snapshot plots every 0.1 s and final-state plots for conserved and primitive variables.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import newton_krylov
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scipy is required for the implicit solver (newton_krylov). Please install scipy and retry."
    ) from exc


# ---------- Helpers ----------
def primitives_from_conservatives(beta, omega, rho_g, rho_l, U_mix, eps=1e-8):
    beta = np.asarray(beta)
    omega = np.asarray(omega)

    drho = rho_g - rho_l
    if abs(drho) < eps:
        drho = np.sign(drho) * eps if drho != 0 else eps

    alpha_g = (beta - rho_l) / drho
    alpha_l = 1.0 - alpha_g
    alpha_g = np.clip(alpha_g, eps, 1.0 - eps)
    alpha_l = np.clip(alpha_l, eps, 1.0 - eps)

    beta_eff = alpha_g * rho_g + alpha_l * rho_l
    beta_eff = np.where(np.abs(beta_eff) < eps, eps, beta_eff)

    num_g = U_mix * rho_l + (beta_eff - rho_l) * omega / (rho_g - rho_l)
    num_l = U_mix * rho_g + (beta_eff - rho_g) * omega / (rho_g - rho_l)
    u_g = num_g / beta_eff
    u_l = num_l / beta_eff
    return alpha_g, alpha_l, u_g, u_l


def flux_from_state(beta, omega, rho_g, rho_l, U_mix, vel_clip=1e3):
    alpha_g, alpha_l, u_g, u_l = primitives_from_conservatives(
        beta, omega, rho_g, rho_l, U_mix
    )
    u_g = np.clip(u_g, -vel_clip, vel_clip)
    u_l = np.clip(u_l, -vel_clip, vel_clip)
    F1 = rho_g * alpha_g * u_g + rho_l * alpha_l * u_l
    F2 = 0.5 * (rho_g * u_g**2 - rho_l * u_l**2)
    return np.array([F1, F2])


def source_from_state(beta, omega, rho_g, rho_l, U_mix, g=9.81):
    alpha_g, alpha_l, _, _ = primitives_from_conservatives(beta, omega, rho_g, rho_l, U_mix)
    G1 = rho_g * g
    G2 = rho_l * g
    S1 = np.zeros_like(beta)
    S2 = G1 / alpha_g - G2 / alpha_l
    return np.array([S1, S2])


def U_mix_from_primitives(alpha1, alpha2, u1, u2):
    return alpha1 * u1 + alpha2 * u2


def beta_from_primitives(alpha1, alpha2, rho1, rho2):
    return alpha1 * rho1 + alpha2 * rho2


def omega_from_primitives(rho1, rho2, u1, u2):
    return rho1 * u1 - rho2 * u2


def characteristic_speed(beta, omega, rho_g, rho_l, U_mix, eps=1e-8, lam_clip=1e6):
    beta_safe = np.clip(beta, eps, None)
    drho = rho_g - rho_l
    if abs(drho) < eps:
        drho = np.sign(drho) * eps if drho != 0 else eps
    lam_num = rho_g * rho_l * (U_mix * (rho_l - rho_g) + beta_safe * omega)
    lam_den = (beta_safe ** 2) * (rho_l - rho_g)
    lam = lam_num / lam_den
    return float(np.clip(lam, -lam_clip, lam_clip))


# ---------- Implicit solver ----------
def backward_euler_step(U_old, dt, dx, params):
    """
    Solve U^{n+1} implicitly via Newton-Krylov:
        U - U_old + dt * L(U) = 0
    where L(U) is the spatial operator with Rusanov flux and source.
    """
    beta_old, omega_old = U_old
    rhog, rhol, U_mix_const, gx, beta_in, omega_in = params
    nx = beta_old.size
    ncells = nx - 1

    def residual(U_vec):
        beta = U_vec[:nx].copy()
        omega = U_vec[nx:].copy()

        # enforce BCs on working copies for fluxes
        beta_bc = beta.copy()
        omega_bc = omega.copy()
        beta_bc[0] = beta_in
        omega_bc[0] = omega_in
        beta_bc[-1] = 2 * beta_bc[-2] - beta_bc[-3]
        omega_bc[-1] = 2 * omega_bc[-2] - omega_bc[-3]

        # interface fluxes
        F_hat = np.zeros((2, nx - 1))
        for i in range(nx - 1):
            F_L = flux_from_state(beta_bc[i], omega_bc[i], rhog, rhol, U_mix_const[i])
            F_R = flux_from_state(beta_bc[i + 1], omega_bc[i + 1], rhog, rhol, U_mix_const[i + 1])
            lam_L = characteristic_speed(beta_bc[i], omega_bc[i], rhog, rhol, U_mix_const[i])
            lam_R = characteristic_speed(beta_bc[i + 1], omega_bc[i + 1], rhog, rhol, U_mix_const[i + 1])
            a = min(max(abs(lam_L), abs(lam_R)), 1e6)
            F_hat[:, i] = 0.5 * (F_L + F_R) - 0.5 * a * np.array(
                [beta_bc[i + 1] - beta_bc[i], omega_bc[i + 1] - omega_bc[i]]
            )

        R_beta = np.zeros_like(beta)
        R_omega = np.zeros_like(omega)

        # interior residuals
        for i in range(1, ncells):
            Source = source_from_state(beta_bc[i], omega_bc[i], rhog, rhol, U_mix_const, gx[i])
            R_beta[i] = beta[i] - beta_old[i] + dt / dx * (F_hat[0, i] - F_hat[0, i - 1]) - dt * Source[0]
            R_omega[i] = omega[i] - omega_old[i] + dt / dx * (F_hat[1, i] - F_hat[1, i - 1]) - dt * Source[1]

        # boundary residuals (Dirichlet inlet, extrapolated outlet)
        R_beta[0] = beta[0] - beta_in
        R_omega[0] = omega[0] - omega_in
        R_beta[-1] = beta[-1] - (2 * beta[-2] - beta[-3])
        R_omega[-1] = omega[-1] - (2 * omega[-2] - omega[-3])

        return np.concatenate([R_beta, R_omega])

    U_guess = np.concatenate([beta_old, omega_old])
    U_new = newton_krylov(residual, U_guess, method="lgmres", inner_M=None, verbose=0, f_tol=1e-8)
    beta_new = U_new[:nx]
    omega_new = U_new[nx:]
    return beta_new, omega_new


def main():
    # Physical parameters
    rhog = 1.0   # gas density
    rhol = 1000.0  # liquid density

    # Discretization
    tini = 0.0
    tEnd = 0.5
    dt = 5e-3  # implicit allows larger step than explicit version
    nt = int((tEnd - tini) / dt)
    ncells = 1000
    L = 12.0
    dx = L / ncells
    nx = ncells + 1
    x = np.linspace(0, L, nx)

    # Initial conditions
    ul_ini = np.full(nx, 10.0)
    ug_ini = np.full(nx, 1e-8)
    alphag_ini = np.full(nx, 0.2)
    gx = np.full(nx, 9.81)

    U_mix_const = U_mix_from_primitives(alphag_ini, 1 - alphag_ini, ug_ini, ul_ini)
    if np.isscalar(U_mix_const):
        U_mix_const = np.full_like(alphag_ini, U_mix_const)

    beta_ini = beta_from_primitives(alphag_ini, 1 - alphag_ini, rhog, rhol)
    omega_ini = omega_from_primitives(rhog, rhol, ug_ini, ul_ini)

    beta_old = beta_ini.copy()
    omega_old = omega_ini.copy()

    params = (rhog, rhol, U_mix_const, gx, beta_ini[0], omega_ini[0])

    # Snapshot storage every 0.1s
    snapshot_times = np.arange(0.1, tEnd + 1e-12, 0.1)
    snap_idx = 0
    snaps_beta, snaps_omega = [], []
    snaps_alphag, snaps_ug, snaps_ul, snaps_umix = [], [], [], []

    print_every = max(nt // 20, 1)
    for step in range(nt):
        if step % print_every == 0 or step == nt - 1:
            print(f"[Implicit] Step {step+1}/{nt}, t = {(step+1)*dt:.6f} s")

        beta_new, omega_new = backward_euler_step((beta_old, omega_old), dt, dx, params)

        # snapshots
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

        beta_old = beta_new
        omega_old = omega_new

    # Final primitives
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
        plt.savefig(f"implicit_conserved_snapshots_beta_{ncells}.png", dpi=200)

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
        plt.savefig(f"implicit_conserved_snapshots_omega_{ncells}.png", dpi=200)

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
        plt.savefig(f"implicit_conserved_snapshots_Umix_{ncells}.png", dpi=200)

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
        plt.savefig(f"implicit_primitive_snapshots_alphag_{ncells}.png", dpi=200)

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
        plt.savefig(f"implicit_primitive_snapshots_ug_{ncells}.png", dpi=200)

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
        plt.savefig(f"implicit_primitive_snapshots_ul_{ncells}.png", dpi=200)

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
