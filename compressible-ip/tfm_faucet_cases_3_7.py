
#!/usr/bin/env python3

"""
tfm_faucet_cases_3_7.py

A robust(ish) 1-D two-fluid "water faucet" benchmark solver that compares
the Case 3–7 hyperbolicity modifications (from your quartic-case notes)
using a conservative phase-mass/phase-momentum formulation.

Key idea (why this behaves better than your last plot):
- We advance phase masses m_k = alpha_k*rho_k and phase momenta m_k*u_k
  (conservative variables).
- Pressure p is recovered *consistently* each step by enforcing
    alpha_g + alpha_l = 1
  with barotropic EOS rho_k(p). This prevents alpha spikes and keeps mass bounded.
- Interfacial pressure correction uses the "minimum" pi_min formula (Singh & Mousseau),
  optionally scaled by an "iota" (for compressible adjustment).
- Cases 3–7 are implemented as additional wave-speed bounds in the Rusanov flux
  (so they influence dissipation without breaking conservation).

This is meant as a clean numerical sandbox; you can later replace:
- EOS rho_k(p) with ideal gas + stiffened gas
- drag model
- flux (Rusanov -> Roe/HLLC/path-conservative)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Helpers: EOS + pi + (p,q)
# ---------------------------

def rho_barotropic_linear(p: np.ndarray, rho_ref: float, p_ref: float, c: float) -> np.ndarray:
    """rho(p) = rho_ref + (p - p_ref)/c^2 (barotropic linearized EOS)."""
    return rho_ref + (p - p_ref) / (c * c)

def solve_p_from_masses(
    mg: np.ndarray,
    ml: np.ndarray,
    rho_g_ref: float, rho_l_ref: float,
    p_ref: float,
    cg: float, cl: float,
) -> np.ndarray:
    """
    Solve mg/rho_g(p) + ml/rho_l(p) = 1, with rho_k(p) linear in p.

    If rho_k(p) = A_k + B_k p, equation becomes quadratic in p and can be solved analytically.
    """
    Bg = 1.0 / (cg * cg)
    Bl = 1.0 / (cl * cl)
    Ag = rho_g_ref - p_ref * Bg
    Al = rho_l_ref - p_ref * Bl

    a = Bg * Bl
    b = (Ag * Bl + Al * Bg - mg * Bl - ml * Bg)
    c = (Ag * Al - mg * Al - ml * Ag)

    # Near-linear fallback (rare unless you make one phase nearly incompressible)
    if abs(a) < 1e-30:
        return -c / (b + 1e-30)

    disc = b * b - 4.0 * a * c
    disc = np.maximum(disc, 0.0)
    sdisc = np.sqrt(disc)
    p1 = (-b + sdisc) / (2.0 * a)
    p2 = (-b - sdisc) / (2.0 * a)

    # pick the root closer to p_ref (and prefer positive)
    pick1 = np.abs(p1 - p_ref) <= np.abs(p2 - p_ref)
    p = np.where(pick1, p1, p2)
    other = np.where(pick1, p2, p1)
    p = np.where((p < 0) & (other > 0), other, p)
    return p

def pi_min(alpha_g: np.ndarray, ug: np.ndarray, ul: np.ndarray, rhog: np.ndarray, rhol: np.ndarray) -> np.ndarray:
    """Minimum incompressible interfacial pressure correction (Singh & Mousseau Eq. 33)."""
    alpha_l = 1.0 - alpha_g
    num = alpha_g * alpha_l * rhog * rhol * (ug - ul) ** 2
    den = alpha_l * rhog + alpha_g * rhol
    return np.where(den > 0, num / den, 0.0)

def compute_pq(alpha_g: np.ndarray, ug: np.ndarray, ul: np.ndarray, rhog: np.ndarray, rhol: np.ndarray):
    """
    p-bar and q-bar that appear in the incompressible complex pair p ± i q
    (same weighted-mean structure as the closed-form eigenvalues).
    """
    alpha_l = 1.0 - alpha_g
    denom = alpha_l * rhog + alpha_g * rhol
    pbar = np.where(denom > 0, (ug * alpha_l * rhog + ul * alpha_g * rhol) / denom, 0.0)
    qbar = np.where(
        denom > 0,
        np.abs(ug - ul) * np.sqrt(np.maximum(alpha_g * alpha_l * rhog * rhol, 0.0)) / denom,
        0.0,
    )
    return pbar, qbar

def case_pair_speeds(case_id: int, pbar: np.ndarray, qbar: np.ndarray):
    """
    Your Case 3–7 mappings (real surrogates of the complex pair).
    NOTE: Case 6 can become invalid when pbar^2 < qbar^2 (radicand negative);
          here we clamp that part to 0, but in research code you should
          treat it as "not realizable" and fall back (e.g., Case 7).
    """
    if case_id == 3:          # double root
        return pbar, pbar
    if case_id == 4:
        r = np.sqrt(np.maximum(pbar * pbar + qbar * qbar, 0.0))
        return r, -r
    if case_id == 5:
        r = np.sqrt(np.maximum(2.0 * pbar * pbar - qbar * qbar, 0.0))
        return r, -r
    if case_id == 6:
        r4 = np.sign(pbar) * np.sqrt(np.maximum(pbar * pbar - qbar * qbar, 0.0))
        return pbar + qbar, r4
    if case_id == 7:
        return pbar + qbar, pbar - qbar
    raise ValueError("case_id must be 3..7")


def vanleer_limiter(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)


def limited_slopes(arr):
    """Van Leer limited slopes (vector)."""
    dL = arr[1:-1] - arr[:-2]
    dR = arr[2:] - arr[1:-1]
    r = dR / (dL + 1e-30)
    phi = vanleer_limiter(r)
    slopes = np.zeros_like(arr)
    slopes[1:-1] = phi * dL
    return slopes


# ---------------------------
# Main solver
# ---------------------------

def faucet_run(
    case_id: int,
    ip_mode: str,                    # "none" | "incomp" | "comp"
    *,
    # domain/time
    N: int = 250,
    L: float = 12.0,
    T: float = 0.5,
    CFL: float = 0.35,
    # benchmark BCs (Ransom faucet style)
    alpha_in: float = 0.2,
    #ug_in: float = 5.0,
    #ul_in: float = 5.0,
    ug_in: float = 1e-8,
    ul_in: float = 10.0,
    p_out: float = 7.5e6,            # absolute outlet pressure (paper uses 7.5 or 15 MPa)
    # barotropic EOS anchors
    #rho_g_ref: float = 40.0,
    rho_g_ref: float = 1.0,
    #rho_l_ref: float = 950.0,
    rho_l_ref: float = 1000.0,
    #cg: float = 400.0,
    cg: float = 300.0,
    cl: float = 1500.0,
    # sources
    g: float = 9.81,
    #Kd: float = 2000.0,              # drag strength
    Kd: float = 0.0,              # drag strength
    # compressibility adjustment for pi
    iota: float = 1.0,
    cap_pi_factor: float = 2.0,      # pi <= cap_pi_factor * p  (numerical safety cap)
    flux_scheme: str = "rusanov",    # rus'anov | muscl-vanleer | central-evm | ku | nt
    evm_const: float = 0.5,
):
    """
    Conservative two-phase (m_g, m_g u_g, m_l, m_l u_l) with barotropic p from volume constraint.
    """

    dx = L / N
    x = np.linspace(dx / 2, L - dx / 2, N)
    p_ref = p_out  # we reference EOS at outlet pressure

    ng = 2  # ghost cells
    M = N + 2 * ng

    # Initial primitives (uniform)
    alpha = np.full(N, alpha_in, dtype=float)
    p = np.full(N, p_ref, dtype=float)
    rhog = rho_barotropic_linear(p, rho_g_ref, p_ref, cg)
    rhol = rho_barotropic_linear(p, rho_l_ref, p_ref, cl)

    mg = alpha * rhog
    ml = (1.0 - alpha) * rhol
    momg = mg * ug_in
    moml = ml * ul_in

    def primitives_from_conserved(mg_arr, momg_arr, ml_arr, moml_arr):
        p_arr = solve_p_from_masses(mg_arr, ml_arr, rho_g_ref, rho_l_ref, p_ref, cg, cl)
        rhog_arr = rho_barotropic_linear(p_arr, rho_g_ref, p_ref, cg)
        rhol_arr = rho_barotropic_linear(p_arr, rho_l_ref, p_ref, cl)
        alpha_arr = np.clip(mg_arr / np.maximum(rhog_arr, 1e-14), 1e-10, 1.0 - 1e-10)
        ug_arr = momg_arr / np.maximum(mg_arr, 1e-14)
        ul_arr = moml_arr / np.maximum(ml_arr, 1e-14)
        return p_arr, rhog_arr, rhol_arr, alpha_arr, ug_arr, ul_arr

    def extend_with_bc(mg, momg, ml, moml):
        # Start with edge padding
        Mg = np.pad(mg, (ng, ng), mode="edge")
        Ml = np.pad(ml, (ng, ng), mode="edge")
        Momg = np.pad(momg, (ng, ng), mode="edge")
        Moml = np.pad(moml, (ng, ng), mode="edge")

        # ---- Left boundary (inlet): fix alpha, ug, ul; extrapolate p
        p0, *_ = primitives_from_conserved(
            Mg[ng:ng+1], Momg[ng:ng+1], Ml[ng:ng+1], Moml[ng:ng+1]
        )
        pL = float(p0[0])
        rhogL = float(rho_barotropic_linear(np.array([pL]), rho_g_ref, p_ref, cg)[0])
        rholL = float(rho_barotropic_linear(np.array([pL]), rho_l_ref, p_ref, cl)[0])
        mgL = alpha_in * rhogL
        mlL = (1.0 - alpha_in) * rholL
        momgL = mgL * ug_in
        momlL = mlL * ul_in
        Mg[:ng] = mgL
        Ml[:ng] = mlL
        Momg[:ng] = momgL
        Moml[:ng] = momlL

        # ---- Right boundary (outlet): fix p; extrapolate alpha, ug, ul
        pN, rhogN, rholN, alphaN, ugN, ulN = primitives_from_conserved(
            Mg[ng+N-1:ng+N], Momg[ng+N-1:ng+N], Ml[ng+N-1:ng+N], Moml[ng+N-1:ng+N]
        )
        alphaR = float(alphaN[0])
        ugR = float(ugN[0])
        ulR = float(ulN[0])
        rhogR = float(rho_barotropic_linear(np.array([p_out]), rho_g_ref, p_ref, cg)[0])
        rholR = float(rho_barotropic_linear(np.array([p_out]), rho_l_ref, p_ref, cl)[0])
        mgR = alphaR * rhogR
        mlR = (1.0 - alphaR) * rholR
        momgR = mgR * ugR
        momlR = mlR * ulR
        Mg[ng+N:] = mgR
        Ml[ng+N:] = mlR
        Momg[ng+N:] = momgR
        Moml[ng+N:] = momlR

        return Mg, Momg, Ml, Moml

    t = 0.0
    while t < T - 1e-14:
        Mg, Momg, Ml, Moml = extend_with_bc(mg, momg, ml, moml)
        p_e, rhog_e, rhol_e, alpha_e, ug_e, ul_e = primitives_from_conserved(Mg, Momg, Ml, Moml)

        # Interfacial pressure correction
        if ip_mode == "none":
            pi_e = np.zeros_like(p_e)
        else:
            pi0 = pi_min(alpha_e, ug_e, ul_e, rhog_e, rhol_e)
            if ip_mode == "incomp":
                pi_e = iota * pi0
            elif ip_mode == "comp":
                # simple slip-Mach scaling (paper: compressibility effect small when Ms is small)
                cm = np.sqrt(
                    (alpha_e * rhog_e * cg * cg + (1.0 - alpha_e) * rhol_e * cl * cl)
                    / np.maximum(alpha_e * rhog_e + (1.0 - alpha_e) * rhol_e, 1e-14)
                )
                Ms = np.abs(ug_e - ul_e) / np.maximum(cm, 1e-14)
                pi_e = iota * pi0 * (1.0 + 0.5 * Ms * Ms)
            else:
                raise ValueError("ip_mode must be 'none', 'incomp', or 'comp'")

            # numeric safety cap: prevents huge pi when (ug-ul)^2 grows spuriously
            pi_cap = cap_pi_factor * np.maximum(p_e, 1.0)
            pi_e = np.minimum(pi_e, pi_cap)

        pg_e = p_e + pi_e
        pl_e = p_e - pi_e

        # Lax-Friedrichs / Rusanov dissipation speed bound
        s_cell = np.maximum(np.abs(ug_e) + cg, np.abs(ul_e) + cl)

        # Case 3–7: include their pair speeds as extra bounds (changes dissipation, not conservation)
        pbar_e, qbar_e = compute_pq(alpha_e, ug_e, ul_e, rhog_e, rhol_e)
        r3_e, r4_e = case_pair_speeds(case_id, pbar_e, qbar_e)
        s_cell = np.maximum(s_cell, np.maximum(np.abs(r3_e), np.abs(r4_e)))

        smax = np.max(s_cell[ng:-ng])
        dt = CFL * dx / max(smax, 1e-12)
        if t + dt > T:
            dt = T - t

        # ---------------------------
        # Interface states (reconstruction per scheme)
        # ---------------------------
        if flux_scheme == "rusanov":
            MgL, MgR = Mg[:-1], Mg[1:]
            MlL, MlR = Ml[:-1], Ml[1:]
            MomgL, MomgR = Momg[:-1], Momg[1:]
            MomlL, MomlR = Moml[:-1], Moml[1:]
        else:
            smg = limited_slopes(Mg)
            sml = limited_slopes(Ml)
            smomg = limited_slopes(Momg)
            smoml = limited_slopes(Moml)
            MgL = Mg[:-1] + 0.5 * smg[:-1]
            MgR = Mg[1:] - 0.5 * smg[1:]
            MlL = Ml[:-1] + 0.5 * sml[:-1]
            MlR = Ml[1:] - 0.5 * sml[1:]
            MomgL = Momg[:-1] + 0.5 * smomg[:-1]
            MomgR = Momg[1:] - 0.5 * smomg[1:]
            MomlL = Moml[:-1] + 0.5 * smoml[:-1]
            MomlR = Moml[1:] - 0.5 * smoml[1:]

        pL, rhogL, rholL, alphaL, ugL, ulL = primitives_from_conserved(MgL, MomgL, MlL, MomlL)
        pR, rhogR, rholR, alphaR, ugR, ulR = primitives_from_conserved(MgR, MomgR, MlR, MomlR)

        pgL = pL
        pgR = pR
        plL = pL
        plR = pR

        # Physical fluxes
        FmgL = MomgL
        FmgR = MomgR
        FmlL = MomlL
        FmlR = MomlR

        FmomgL = MomgL * ugL + alphaL * pgL
        FmomgR = MomgR * ugR + alphaR * pgR

        FmomlL = MomlL * ulL + (1.0 - alphaL) * plL
        FmomlR = MomlR * ulR + (1.0 - alphaR) * plR

        if flux_scheme in ("rusanov", "muscl-vanleer"):
            s_int = np.maximum.reduce([np.abs(ugL) + cg, np.abs(ugR) + cg, np.abs(ulL) + cl, np.abs(ulR) + cl])
            r3L, r4L = r3_e[:-1], r4_e[:-1]
            r3R, r4R = r3_e[1:], r4_e[1:]
            s_int = np.maximum(s_int, np.maximum.reduce([np.abs(r3L), np.abs(r4L), np.abs(r3R), np.abs(r4R)]))
            Fmg = 0.5 * (FmgL + FmgR) - 0.5 * s_int * (MgR - MgL)
            Fml = 0.5 * (FmlL + FmlR) - 0.5 * s_int * (MlR - MlL)
            Fmomg = 0.5 * (FmomgL + FmomgR) - 0.5 * s_int * (MomgR - MomgL)
            Fmoml = 0.5 * (FmomlL + FmomlR) - 0.5 * s_int * (MomlR - MomlL)
        elif flux_scheme == "central-evm":
            a_face = np.maximum.reduce([np.abs(ugL) + cg, np.abs(ugR) + cg, np.abs(ulL) + cl, np.abs(ulR) + cl])
            grad_alpha = np.abs(alphaR - alphaL)
            nu_face = evm_const * dx * grad_alpha
            Fmg = 0.5 * (FmgL + FmgR) - 0.5 * (a_face + nu_face / dx) * (MgR - MgL)
            Fml = 0.5 * (FmlL + FmlR) - 0.5 * (a_face + nu_face / dx) * (MlR - MlL)
            Fmomg = 0.5 * (FmomgL + FmomgR) - 0.5 * (a_face + nu_face / dx) * (MomgR - MomgL)
            Fmoml = 0.5 * (FmomlL + FmomlR) - 0.5 * (a_face + nu_face / dx) * (MomlR - MomlL)
        elif flux_scheme == "ku":
            s_plus = np.maximum.reduce([ugL + cg, ugR + cg, ulL + cl, ulR + cl, np.zeros_like(ugL)])
            s_minus = np.minimum.reduce([ugL - cg, ugR - cg, ulL - cl, ulR - cl, np.zeros_like(ugL)])
            denom = s_plus - s_minus + 1e-30
            Fmg = (s_plus * FmgL - s_minus * FmgR + s_plus * s_minus * (MgR - MgL)) / denom
            Fml = (s_plus * FmlL - s_minus * FmlR + s_plus * s_minus * (MlR - MlL)) / denom
            Fmomg = (s_plus * FmomgL - s_minus * FmomgR + s_plus * s_minus * (MomgR - MomgL)) / denom
            Fmoml = (s_plus * FmomlL - s_minus * FmomlR + s_plus * s_minus * (MomlR - MomlL)) / denom
        elif flux_scheme == "nt":
            a_face = np.maximum.reduce([np.abs(ugL) + cg, np.abs(ugR) + cg, np.abs(ulL) + cl, np.abs(ulR) + cl])
            Fmg = 0.5 * (FmgL + FmgR) - 0.5 * a_face * (MgR - MgL)
            Fml = 0.5 * (FmlL + FmlR) - 0.5 * a_face * (MlR - MlL)
            Fmomg = 0.5 * (FmomgL + FmomgR) - 0.5 * a_face * (MomgR - MomgL)
            Fmoml = 0.5 * (FmomlL + FmomlR) - 0.5 * a_face * (MomlR - MomlL)
        else:
            raise ValueError(f"Unknown flux_scheme={flux_scheme}")

        # Update interior cells (indices ng..ng+N-1)
        j = np.arange(ng, ng + N)
        mg -= (dt / dx) * (Fmg[j] - Fmg[j - 1])
        ml -= (dt / dx) * (Fml[j] - Fml[j - 1])
        momg -= (dt / dx) * (Fmomg[j] - Fmomg[j - 1])
        moml -= (dt / dx) * (Fmoml[j] - Fmoml[j - 1])

        # Sources: gravity + drag
        p, rhog, rhol, alpha, ug, ul = primitives_from_conserved(mg, momg, ml, moml)
        drag = Kd * alpha * (1.0 - alpha)
        Sg = drag * (ul - ug)
        Sl = -Sg
        momg += dt * (mg * g + Sg)
        moml += dt * (ml * g + Sl)

        # Floors (strictly numerical safety)
        mg = np.maximum(mg, 1e-12)
        ml = np.maximum(ml, 1e-12)

        t += dt

    # final primitives
    p, rhog, rhol, alpha, ug, ul = primitives_from_conserved(mg, momg, ml, moml)
    return x, p, alpha, ug, ul


def compare_cases(
    *,
    ip_mode: str = "incomp",
    case_ids=(3, 4, 5, 6, 7),
    N: int = 250,
    L: float = 12.0,
    T: float = 0.5,
    p_out: float = 7.5e6,
    CFL: float = 0.5,
    flux_scheme: str = "rusanov",
):
    outputs = {}
    for cid in case_ids:
        x, p, a, ug, ul = faucet_run(
            cid, ip_mode,
            N=N, L=L, T=T,
            p_out=p_out,
            flux_scheme=flux_scheme,
            CFL=CFL,
        )
        outputs[cid] = (x, p, a, ug, ul)

    # Plot gauge pressure for readability
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axs = axs.ravel()

    for cid, (x, p, a, ug, ul) in outputs.items():
        axs[0].plot(x, p - p_out, label=f"Case {cid}")
        axs[1].plot(x, a, label=f"Case {cid}")
        axs[2].plot(x, ug, label=f"Case {cid}")
        axs[3].plot(x, ul, label=f"Case {cid}")

    axs[0].set_title("Pressure (gauge)")
    axs[1].set_title("alpha_g")
    axs[2].set_title("u_g")
    axs[3].set_title("u_l")
    for ax in axs:
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.show()


if __name__ == "__main__":
    # Match the paper-style faucet benchmark: L=12 m, inlet/IC alpha=0.2.
    #compare_cases(ip_mode="none", p_out=1e5, T=0.5, N=100, CFL=0.2, flux_scheme="rusanov")
    #compare_cases(ip_mode="none", p_out=1e5, T=0.5, N=100, CFL=0.2, flux_scheme="muscl-vanleer")
    #compare_cases(ip_mode="none", p_out=1e5, T=0.5, N=100, CFL=0.2, flux_scheme="central-evm")
    #compare_cases(ip_mode="none", p_out=1e5, T=0.5, N=100, CFL=0.2, flux_scheme="ku")
    compare_cases(ip_mode="none", p_out=1e5, T=0.5, N=100, CFL=0.2, flux_scheme="nt")
