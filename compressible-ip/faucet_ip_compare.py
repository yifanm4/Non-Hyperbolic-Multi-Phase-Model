"""
Water-faucet setup (alpha0=0.2, u_l0=10 m/s, g=9.81) using the compressible
primitive 4-eq model. We compare three interfacial-pressure modes with a
simple first-order upwind/Rusanov flux:
  - no IP
  - incompressible IP
  - compressible IP*

Defaults follow Entropy-Viscocity-TFM/test-water-faucet-simple-MUSCL-van-leer-main.py:
  L=12 m, N=100, t_final=0.5 s, inlet alpha=0.2, u_g≈0, u_l=10, outlet p=0.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------- EOS + helpers ----------------

def eos_gas_ideal_isentropic(p, p_ref, rho_ref, gamma_g, p_min=1.0):
    p = float(max(p, p_min))
    rho = rho_ref * (p / p_ref) ** (1.0 / gamma_g)
    c2 = gamma_g * p / max(rho, 1e-30)
    return float(rho), float(np.sqrt(max(c2, 0.0)))


def eos_liquid_stiffened_isentropic(p, p_ref, rho_ref, gamma_l, p_inf, p_min=1.0):
    p_eff = float(max(p + p_inf, p_min))
    pref_eff = float(max(p_ref + p_inf, p_min))
    rho = rho_ref * (p_eff / pref_eff) ** (1.0 / gamma_l)
    c2 = gamma_l * p_eff / max(rho, 1e-30)
    return float(rho), float(np.sqrt(max(c2, 0.0)))


def mixture_sound_speed(alpha_g, rho_g, rho_l, c_g, c_l):
    ag = float(np.clip(alpha_g, 1e-12, 1.0 - 1e-12))
    al = 1.0 - ag
    rho_m = ag * rho_g + al * rho_l
    denom = ag / max(rho_g * c_g * c_g, 1e-30) + al / max(rho_l * c_l * c_l, 1e-30)
    if denom <= 0.0 or rho_m <= 0.0:
        return 0.0
    return float(np.sqrt(1.0 / (rho_m * denom)))


def pi_incompressible(alpha_g, u_g, u_l, rho_g, rho_l):
    ag = float(np.clip(alpha_g, 1e-12, 1.0 - 1e-12))
    al = 1.0 - ag
    denom = ag * rho_l + al * rho_g
    du = u_g - u_l
    return float((ag * al * rho_g * rho_l / max(denom, 1e-30)) * (du * du))


def det3(M):
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# ---------------- Linearized operators ----------------

def build_A_B_C(F, par, p_i):
    p, ag, ug, ul = float(F[0]), float(F[1]), float(F[2]), float(F[3])
    ag = float(np.clip(ag, 1e-12, 1.0 - 1e-12))
    al = 1.0 - ag

    rho_g, c_g = eos_gas_ideal_isentropic(
        p, par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"]
    )
    rho_l, c_l = eos_liquid_stiffened_isentropic(
        p, par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"]
    )

    g = par["g"]
    Kd = par["K_drag"]

    A = np.zeros((4, 4), dtype=float)
    A[0, 0] = ag / (c_g * c_g)
    A[0, 1] = rho_g
    A[1, 0] = al / (c_l * c_l)
    A[1, 1] = -rho_l
    A[2, 2] = ag * rho_g
    A[3, 3] = al * rho_l

    B = np.zeros((4, 4), dtype=float)
    B[0, 0] = ag * ug / (c_g * c_g)
    B[0, 1] = rho_g * ug
    B[0, 2] = ag * rho_g

    B[1, 0] = al * ul / (c_l * c_l)
    B[1, 1] = -rho_l * ul
    B[1, 3] = al * rho_l

    B[2, 0] = ag
    B[2, 1] = +p_i
    B[2, 2] = ag * rho_g * ug

    B[3, 0] = al
    B[3, 1] = -p_i
    B[3, 3] = al * rho_l * ul

    C = np.zeros(4, dtype=float)
    # gravity acts only on liquid (gas nearly weightless), optional drag
    C[2] = Kd * (ul - ug)
    C[3] = al * rho_l * g - Kd * (ul - ug)

    return A, B, C, (rho_g, c_g, rho_l, c_l)


def P_and_S_closed_form(F, par, p_i):
    p, ag, ug, ul = float(F[0]), float(F[1]), float(F[2]), float(F[3])
    ag = float(np.clip(ag, 1e-12, 1.0 - 1e-12))
    al = 1.0 - ag

    rho_g, c_g = eos_gas_ideal_isentropic(
        p, par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"]
    )
    rho_l, c_l = eos_liquid_stiffened_isentropic(
        p, par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"]
    )

    a = ag / (c_g * c_g)
    b = rho_g
    c = al / (c_l * c_l)
    d = -rho_l
    det = a * d - b * c
    inv00 = d / det
    inv01 = -b / det
    inv10 = -c / det
    inv11 = a / det

    B0 = np.array([ag * ug / (c_g * c_g), rho_g * ug, ag * rho_g, 0.0], dtype=float)
    B1 = np.array([al * ul / (c_l * c_l), -rho_l * ul, 0.0, al * rho_l], dtype=float)

    P = np.zeros((4, 4), dtype=float)
    P[0, :] = inv00 * B0 + inv01 * B1
    P[1, :] = inv10 * B0 + inv11 * B1

    P[2, :] = np.array([1.0 / rho_g, p_i / (ag * rho_g), ug, 0.0], dtype=float)
    P[3, :] = np.array([1.0 / rho_l, -p_i / (al * rho_l), 0.0, ul], dtype=float)

    g = par["g"]
    Kd = par["K_drag"]
    S = np.zeros(4, dtype=float)
    # gravity on liquid only to mimic faucet reference acceleration
    S[2] = Kd * (ul - ug) / max(ag * rho_g, 1e-30)
    S[3] = g - Kd * (ul - ug) / max(al * rho_l, 1e-30)

    return P, S, (rho_g, c_g, rho_l, c_l)


def eig_imag_max(F, par, p_i):
    P, _, _ = P_and_S_closed_form(F, par, p_i)
    lam = np.linalg.eigvals(P)
    return float(np.max(np.abs(np.imag(lam))))


# ---------------- Compressible pi* (same as cip-test) ----------------

def P0_P1(lam, F, par):
    A, B0, _, _ = build_A_B_C(F, par, p_i=0.0)
    M0 = B0 - lam * A
    P0 = float(np.linalg.det(M0))
    M32 = M0[np.ix_([0, 1, 3], [0, 2, 3])]
    M42 = M0[np.ix_([0, 1, 2], [0, 2, 3])]
    P1 = float(-(det3(M32) + det3(M42)))
    return P0, P1


def dP_dlam(fn, lam, F, par, h):
    fp = fn(lam + h, F, par)
    fm = fn(lam - h, F, par)
    return (fp - fm) / (2.0 * h)


def pi_star_compressible(F, par, tol_im=1e-10):
    p, ag, ug, ul = float(F[0]), float(F[1]), float(F[2]), float(F[3])
    _, c_g = eos_gas_ideal_isentropic(p, par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"])
    _, c_l = eos_liquid_stiffened_isentropic(p, par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"])
    cmax = max(c_g, c_l, 1.0)
    lam_min = min(ug, ul) - 2.0 * cmax
    lam_max = max(ug, ul) + 2.0 * cmax
    grid = np.linspace(lam_min, lam_max, 241)
    h = 1e-6 * max(1.0, lam_max - lam_min)

    G = np.zeros_like(grid)
    for i, lam in enumerate(grid):
        P0, P1 = P0_P1(lam, F, par)
        P0p = dP_dlam(lambda x, F, par: P0_P1(x, F, par)[0], lam, F, par, h=h)
        P1p = dP_dlam(lambda x, F, par: P0_P1(x, F, par)[1], lam, F, par, h=h)
        G[i] = P0p * P1 - P0 * P1p

    cand_lams = []
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        fa, fb = G[i], G[i + 1]
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa == 0.0 or fa * fb < 0.0:
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(35):
                mid = 0.5 * (lo + hi)
                P0, P1 = P0_P1(mid, F, par)
                P0p = dP_dlam(lambda x, F, par: P0_P1(x, F, par)[0], mid, F, par, h=h)
                P1p = dP_dlam(lambda x, F, par: P0_P1(x, F, par)[1], mid, F, par, h=h)
                fm = P0p * P1 - P0 * P1p
                if flo * fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            cand_lams.append(0.5 * (lo + hi))

    idx = np.argsort(np.abs(G))
    for j in idx[:8]:
        cand_lams.append(grid[j])

    cand_pis = []
    for lam in cand_lams:
        P0, P1 = P0_P1(lam, F, par)
        if abs(P1) < 1e-20:
            continue
        pi = -P0 / P1
        if not np.isfinite(pi) or pi < 0.0:
            continue
        if eig_imag_max(F, par, pi * (1.0 + 1e-8)) <= tol_im:
            cand_pis.append(pi)

    if cand_pis:
        return float(min(cand_pis))

    if eig_imag_max(F, par, 0.0) <= tol_im:
        return 0.0

    rho_g, _ = eos_gas_ideal_isentropic(p, par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"])
    rho_l, _ = eos_liquid_stiffened_isentropic(p, par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"])
    pi0 = max(1e-16, pi_incompressible(ag, ug, ul, rho_g, rho_l))

    lo, hi = 0.0, pi0
    for _ in range(40):
        if eig_imag_max(F, par, hi) <= tol_im:
            break
        hi *= 2.0

    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if eig_imag_max(F, par, mid) <= tol_im:
            hi = mid
        else:
            lo = mid

    return float(hi)


def build_iota_table(par, ag_grid, Ms_grid, tol_im=1e-10):
    table = np.ones((len(ag_grid), len(Ms_grid)), dtype=float)
    for ia, ag in enumerate(ag_grid):
        rho_g, c_g = eos_gas_ideal_isentropic(par["p_ref"], par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"])
        rho_l, c_l = eos_liquid_stiffened_isentropic(par["p_ref"], par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"])
        cm = mixture_sound_speed(ag, rho_g, rho_l, c_g, c_l)
        for im, Ms in enumerate(Ms_grid):
            du = Ms * cm
            ug = +0.5 * du
            ul = -0.5 * du
            F = np.array([par["p_ref"], ag, ug, ul], dtype=float)
            pi_inc = pi_incompressible(ag, ug, ul, rho_g, rho_l)
            if pi_inc < 1e-18:
                table[ia, im] = 1.0
                continue
            pi_star = pi_star_compressible(F, par, tol_im=tol_im)
            table[ia, im] = float(pi_star / pi_inc)
    return table


def bilinear_interp(xg, yg, Z, x, y):
    x = float(np.clip(x, xg[0], xg[-1]))
    y = float(np.clip(y, yg[0], yg[-1]))
    ix = int(np.searchsorted(xg, x) - 1)
    iy = int(np.searchsorted(yg, y) - 1)
    ix = int(np.clip(ix, 0, len(xg) - 2))
    iy = int(np.clip(iy, 0, len(yg) - 2))
    x0, x1 = xg[ix], xg[ix + 1]
    y0, y1 = yg[iy], yg[iy + 1]
    tx = (x - x0) / max(x1 - x0, 1e-30)
    ty = (y - y0) / max(y1 - y0, 1e-30)
    z00 = Z[ix, iy]
    z10 = Z[ix + 1, iy]
    z01 = Z[ix, iy + 1]
    z11 = Z[ix + 1, iy + 1]
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11


# ---------------- Solver ----------------

class FaucetSolver:
    def __init__(self, N=100, L=12.0, T=0.5):
        self.N = int(N)
        self.L = float(L)
        self.dx = self.L / self.N
        self.x = np.linspace(self.dx / 2.0, self.L - self.dx / 2.0, self.N)
        self.T_final = float(T)

        self.alpha = np.ones(self.N) * 0.2
        self.u_l = np.ones(self.N) * 10.0
        self.u_g = np.ones(self.N) * 1e-10
        self.p = np.ones(self.N) * 1.0e5

        self.rho_l_ref = 1000.0
        self.rho_g_ref = 1.0
        self.gamma_g = 1.4
        self.gamma_l = 4.4
        self.p_inf_l = 6.0e8
        self.p_ref = 1.0e5
        self.p_min = 1.0
        self.gx = 9.81
        self.K_drag = 0.0

        self.CFL = 0.10  # match low CFL in reference Upwind1
        self.alpha_clip = (1e-12, 1.0 - 1e-12)
        self.u_clip = (-500.0, 500.0)
        self.p_clip = (0.1 * self.p_ref, 10.0 * self.p_ref)

        self.alpha_in = 0.2
        self.ug_in = 1e-10
        self.ul_in = 10.0
        self.p_out = 0.0

    def pack(self):
        U = np.zeros((self.N, 4), dtype=float)
        U[:, 0] = self.p
        U[:, 1] = self.alpha
        U[:, 2] = self.u_g
        U[:, 3] = self.u_l
        return U

    def apply_bc(self, U):
        U[0, 1] = self.alpha_in
        U[0, 2] = self.ug_in
        U[0, 3] = self.ul_in
        U[0, 0] = U[1, 0]

        U[-1, 0] = self.p_out
        U[-1, 1] = U[-2, 1]
        U[-1, 2] = U[-2, 2]
        U[-1, 3] = U[-2, 3]

        U[:, 1] = np.clip(U[:, 1], self.alpha_clip[0], self.alpha_clip[1])
        U[:, 2] = np.clip(U[:, 2], self.u_clip[0], self.u_clip[1])
        U[:, 3] = np.clip(U[:, 3], self.u_clip[0], self.u_clip[1])
        U[:, 0] = np.clip(U[:, 0], self.p_clip[0], self.p_clip[1])
        return U

    def params(self):
        return {
            "p_ref": self.p_ref,
            "p_min": self.p_min,
            "rho_g_ref": self.rho_g_ref,
            "rho_l_ref": self.rho_l_ref,
            "gamma_g": self.gamma_g,
            "gamma_l": self.gamma_l,
            "p_inf_l": self.p_inf_l,
            "g": self.gx,
            "K_drag": self.K_drag,
        }


def choose_pi(F, par, mode, iota_table=None, ag_grid=None, Ms_grid=None):
    p, ag, ug, ul = float(F[0]), float(F[1]), float(F[2]), float(F[3])
    if mode == "none":
        return 0.0
    rho_g, c_g = eos_gas_ideal_isentropic(p, par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"])
    rho_l, c_l = eos_liquid_stiffened_isentropic(p, par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"])
    pi_inc = pi_incompressible(ag, ug, ul, rho_g, rho_l)
    if mode == "incomp":
        return pi_inc
    if mode == "comp":
        cm = mixture_sound_speed(ag, rho_g, rho_l, c_g, c_l)
        Ms = abs(ug - ul) / max(cm, 1e-30)
        iota = bilinear_interp(ag_grid, Ms_grid, iota_table, ag, Ms)
        return float(iota * pi_inc)
    raise ValueError("mode must be 'none', 'incomp', or 'comp'")


def run_case(solver, mode, iota_table=None, ag_grid=None, Ms_grid=None):
    par = solver.params()
    U = solver.apply_bc(solver.pack())

    t = 0.0
    times = []
    max_imag = []

    while t < solver.T_final - 1e-15:
        # CFL based on max wave speed estimate
        smax = 0.0
        for k in range(0, solver.N, max(1, solver.N // 25)):
            p, ag, ug, ul = float(U[k, 0]), float(U[k, 1]), float(U[k, 2]), float(U[k, 3])
            rho_g, c_g = eos_gas_ideal_isentropic(p, par["p_ref"], par["rho_g_ref"], par["gamma_g"], p_min=par["p_min"])
            rho_l, c_l = eos_liquid_stiffened_isentropic(p, par["p_ref"], par["rho_l_ref"], par["gamma_l"], par["p_inf_l"], p_min=par["p_min"])
            smax = max(smax, max(abs(ug), abs(ul)) + max(c_g, c_l))
        dt = solver.CFL * solver.dx / max(smax, 1e-30)
        dt = min(dt, solver.T_final - t)

        # First-order upwind using sign of mixture velocity, with small LF viscosity
        Fhat = np.zeros((solver.N - 1, 4), dtype=float)
        for i in range(solver.N - 1):
            UL = U[i]
            UR = U[i + 1]
            Um = 0.5 * (UL + UR)
            p_i = choose_pi(Um, par, mode, iota_table, ag_grid, Ms_grid)
            Pm, Sm, (rho_g, c_g, rho_l, c_l) = P_and_S_closed_form(Um, par, p_i)
            u_mix = Um[1] * Um[2] + (1.0 - Um[1]) * Um[3]
            if u_mix >= 0.0:
                base_flux = Pm @ UL
            else:
                base_flux = Pm @ UR
            alpha_num = 0.3 * (max(abs(Um[2]), abs(Um[3])) + max(c_g, c_l))
            Fhat[i] = base_flux - 0.5 * alpha_num * (UR - UL)

        Un = U.copy()
        for i in range(1, solver.N - 1):
            p_i_c = choose_pi(U[i], par, mode, iota_table, ag_grid, Ms_grid)
            Pc, Sc, _ = P_and_S_closed_form(U[i], par, p_i_c)
            Un[i] = U[i] - (dt / solver.dx) * (Fhat[i] - Fhat[i - 1]) + dt * Sc

        # enforce target volumetric flux locally (keeps far field near alpha0)
        j_target = solver.alpha_in * solver.ug_in + (1.0 - solver.alpha_in) * solver.ul_in
        j_curr = Un[:, 1] * Un[:, 2] + (1.0 - Un[:, 1]) * Un[:, 3]
        delta_j = j_target - j_curr
        Un[:, 2] += delta_j
        Un[:, 3] += delta_j

        U = solver.apply_bc(Un)
        t += dt

        if (len(times) == 0) or (times and t - times[-1] >= 0.01):
            imax = 0.0
            idxs = np.linspace(0, solver.N - 1, 20).astype(int)
            for j in idxs:
                p_i_j = choose_pi(U[j], par, mode, iota_table, ag_grid, Ms_grid)
                imax = max(imax, eig_imag_max(U[j], par, p_i_j))
            times.append(t)
            max_imag.append(imax)

    return solver.x.copy(), U, np.array(times), np.array(max_imag)


def main():
    solver = FaucetSolver(N=100, L=12.0, T=0.5)
    par = solver.params()

    ag_grid = np.linspace(0.05, 0.95, 13)
    Ms_grid = np.linspace(0.0, 1.5, 21)
    iota_table = build_iota_table(par, ag_grid, Ms_grid, tol_im=1e-10)

    cases = [
        ("No IP", "none"),
        ("Incompressible IP", "incomp"),
        ("Compressible IP*", "comp"),
    ]

    results = {}
    for name, mode in cases:
        solver = FaucetSolver(N=solver.N, L=solver.L, T=solver.T_final)
        print(f"Running: {name}")
        x, Ufin, t_hist, im_hist = run_case(
            solver,
            mode=mode,
            iota_table=iota_table,
            ag_grid=ag_grid,
            Ms_grid=Ms_grid,
        )
        results[name] = (x, Ufin, t_hist, im_hist)

    plt.figure()
    for name, _ in cases:
        x, Ufin, _, _ = results[name]
        plt.plot(x, Ufin[:, 1], label=name)
    plt.xlabel("x (m)")
    plt.ylabel(r"$\alpha_g$")
    plt.title("Void fraction, Upwind1 faucet setup")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("void_fraction_upwind_ip.png", dpi=200)

    plt.figure()
    for name, _ in cases:
        x, Ufin, _, _ = results[name]
        plt.plot(x, Ufin[:, 2], label=name)
    plt.xlabel("x (m)")
    plt.ylabel(r"$u_g$ (m/s)")
    plt.title("Gas velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("ug_upwind_ip.png", dpi=200)

    plt.figure()
    for name, _ in cases:
        x, Ufin, _, _ = results[name]
        plt.plot(x, Ufin[:, 3], label=name)
    plt.xlabel("x (m)")
    plt.ylabel(r"$u_l$ (m/s)")
    plt.title("Liquid velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("ul_upwind_ip.png", dpi=200)

    plt.figure()
    for name, _ in cases:
        _, _, t_hist, im_hist = results[name]
        plt.semilogy(t_hist, np.maximum(im_hist, 1e-16), label=name)
        plt.xlabel("t (s)")
        plt.ylabel("max |Im(lambda)| (sampled)")
        plt.title("Hyperbolicity diagnostic")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.savefig("hyperbolicity_upwind_ip.png", dpi=200)

    print("Saved plots: void_fraction_upwind_ip.png, ug_upwind_ip.png, ul_upwind_ip.png, hyperbolicity_upwind_ip.png")


if __name__ == "__main__":
    main()
