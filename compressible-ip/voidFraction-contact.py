#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1-D barotropic compressible 1-pressure two-fluid model (4-eq):
  U = [m_g, m_l, q_g, q_l]
  m_k = alpha_k * rho_k(p)
  q_k = m_k * u_k

Mass:
  ∂t m_g + ∂x (m_g u_g) = 0
  ∂t m_l + ∂x (m_l u_l) = 0

Momentum (conservative flux + nonconservative void-gradient coupling):
  ∂t q_g + ∂x (q_g u_g + alpha_g p)  - (p - p_i)*∂x alpha_g = S_g
  ∂t q_l + ∂x (q_l u_l + alpha_l p)  + (p - p_i)*∂x alpha_g = S_l
  (These nonconservative terms cancel in mixture momentum.)

EOS (barotropic, constant sound speeds):
  rho_g(p) = p / c_g^2
  rho_l(p) = rho_l0 + (p - p0)/c_l^2 = a + b p

Pressure recovery in each cell:
  m_g/rho_g(p) + m_l/rho_l(p) = 1  -> quadratic in p (solved analytically)

Numerics:
  - Finite-volume, forward Euler
  - Rusanov (local LF) flux
  - reconstruction: "first" or "vanleer" (MUSCL)
  - FIX B implemented: discretize (p - p_i)*∂x alpha as a FACE-JUMP divergence:
        term_i = [ (p-p_i)_{i+1/2} (alpha_{i+1}-alpha_i)
                 - (p-p_i)_{i-1/2} (alpha_i-alpha_{i-1}) ] / dx
    instead of central alpha_x.

Three runs compared:
  1) no IP: p_i = 0
  2) incompressible IP: Pi_inc = (1-r^2)/4, p_i = (alpha_l rho_g + alpha_g rho_l)*(ug-ul)^2*Pi_inc
  3) compressible IP: Pi_comp(r,Ms^2) precomputed table + bilinear interpolation

Dependencies:
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# EOS + pressure recovery
# -------------------------
class BarotropicEOS:
    def __init__(self, c_g=450.0, c_l=1500.0, rho_l0=1000.0, p0=1e5):
        self.c_g = float(c_g)
        self.c_l = float(c_l)
        self.c_g2 = self.c_g**2
        self.c_l2 = self.c_l**2
        self.rho_l0 = float(rho_l0)
        self.p0 = float(p0)

        # rho_l(p) = a + b p
        self.b = 1.0 / self.c_l2
        self.a = self.rho_l0 - self.p0 / self.c_l2

    def rho_g(self, p):
        return p / self.c_g2

    def rho_l(self, p):
        return self.a + self.b * p

    def pressure_from_masses_vec(self, m_g, m_l, p_floor=1.0):
        """
        Vectorized solve for p > 0 from:
          m_g/(p/c_g^2) + m_l/(a + b p) = 1

        Quadratic:
          b p^2 + (a - m_g c_g^2 b - m_l) p - (m_g c_g^2 a) = 0
        """
        mg = np.maximum(m_g, 1e-14)
        ml = np.maximum(m_l, 1e-14)

        A = self.b
        B = self.a - (mg * self.c_g2 * self.b) - ml
        C = -(mg * self.c_g2 * self.a)

        D = np.maximum(B*B - 4.0*A*C, 0.0)
        sqrtD = np.sqrt(D)

        # two roots
        p1 = (-B + sqrtD) / (2.0*A)
        p2 = (-B - sqrtD) / (2.0*A)

        # choose positive root above floor; prefer the smaller positive
        p1_ok = np.isfinite(p1) & (p1 > p_floor)
        p2_ok = np.isfinite(p2) & (p2 > p_floor)

        p = np.full_like(p1, max(p_floor, 1.0), dtype=float)
        # if both ok: take min; if only one ok: take that one
        both = p1_ok & p2_ok
        p[both] = np.minimum(p1[both], p2[both])
        only1 = p1_ok & (~p2_ok)
        p[only1] = p1[only1]
        only2 = p2_ok & (~p1_ok)
        p[only2] = p2[only2]

        return p

    def mixture_cm2(self, alpha_g, rho_g, alpha_l, rho_l):
        """
        c_m^2 = c_g^2 c_l^2 (alpha_l rho_g + alpha_g rho_l) /
               (c_g^2 alpha_l rho_g + c_l^2 alpha_g rho_l)
        """
        num = self.c_g2 * self.c_l2 * (alpha_l * rho_g + alpha_g * rho_l)
        den = self.c_g2 * (alpha_l * rho_g) + self.c_l2 * (alpha_g * rho_l)
        return num / (den + 1e-30)


# -------------------------
# Primitive recovery
# -------------------------
def primitives_from_U(U, eos: BarotropicEOS):
    """
    U: (N,4) [m_g, m_l, q_g, q_l]
    Returns dict: p, rho_g, rho_l, alpha_g, alpha_l, u_g, u_l
    """
    m_g = U[:, 0]
    m_l = U[:, 1]
    q_g = U[:, 2]
    q_l = U[:, 3]

    p = eos.pressure_from_masses_vec(m_g, m_l, p_floor=1.0)
    rho_g = eos.rho_g(p)
    rho_l = eos.rho_l(p)

    alpha_g = m_g / (rho_g + 1e-30)
    alpha_g = np.clip(alpha_g, 1e-8, 1.0 - 1e-8)
    alpha_l = 1.0 - alpha_g

    u_g = q_g / (m_g + 1e-30)
    u_l = q_l / (m_l + 1e-30)

    return dict(p=p, rho_g=rho_g, rho_l=rho_l,
                alpha_g=alpha_g, alpha_l=alpha_l,
                u_g=u_g, u_l=u_l)


# -------------------------
# IP models: Pi tables
# -------------------------
def Pi_incompressible(r):
    return 0.25 * (1.0 - r*r)


def build_Pi_comp_table(r_min=-0.999, r_max=0.999, nr=161,
                        Ms2_max=1.0, nm=161,
                        tol_imag=1e-10, verbose=True):
    """
    Precompute Pi_comp(r,Ms2) = minimal Pi that makes the quartic have all-real roots.
    Uses tangency candidates from extrema cubic; falls back to bisection if needed.

    The quartic we test is for the characteristic function:
      G(Z;Pi) = (Z^2 - 2 r Z + 1 - 4 Pi) - Ms^2 (1 - Z^2)^2 = 0

    Returns (r_grid, Ms2_grid, Pi_grid).
    """
    r_grid = np.linspace(r_min, r_max, nr)
    Ms2_grid = np.linspace(0.0, Ms2_max, nm)
    Pi_grid = np.zeros((nr, nm), dtype=float)

    def all_real_roots(Ms2, r, Pi):
        # -Ms2 Z^4 + (2Ms2+1) Z^2 - 2r Z + (1 - Ms2 - 4Pi) = 0
        coefs = np.array([-Ms2, 0.0, (2.0*Ms2 + 1.0), (-2.0*r), (1.0 - Ms2 - 4.0*Pi)], float)
        zz = np.roots(coefs)
        return np.max(np.abs(np.imag(zz))) <= tol_imag

    def Pi_comp_min(Ms2, r):
        if Ms2 <= 1e-16:
            return Pi_incompressible(r)

        # extrema cubic: -2 Ms2 Z^3 + (1+2 Ms2) Z - r = 0
        cubic = np.array([-2.0*Ms2, 0.0, (1.0 + 2.0*Ms2), -r], float)
        zc_all = np.roots(cubic)

        candidates = []
        for zc in zc_all:
            if abs(np.imag(zc)) > 1e-10:
                continue
            z = float(np.real(zc))
            Pi = 0.25 * ((z*z - 2.0*r*z + 1.0) - Ms2*(1.0 - z*z)**2)
            if Pi < -1e-12:
                continue
            Pi = max(0.0, float(Pi))
            if all_real_roots(Ms2, r, Pi):
                candidates.append(Pi)

        if candidates:
            return float(min(candidates))

        # fallback: bisection
        Pi_lo = 0.0
        Pi_hi = max(Pi_incompressible(r), 1e-10)
        for _ in range(60):
            if all_real_roots(Ms2, r, Pi_hi):
                break
            Pi_hi *= 2.0

        for _ in range(70):
            Pi_mid = 0.5 * (Pi_lo + Pi_hi)
            if all_real_roots(Ms2, r, Pi_mid):
                Pi_hi = Pi_mid
            else:
                Pi_lo = Pi_mid
        return float(Pi_hi)

    total = nr * nm
    done = 0
    for i, rv in enumerate(r_grid):
        for j, mv in enumerate(Ms2_grid):
            Pi_grid[i, j] = Pi_comp_min(mv, rv)
            done += 1
        if verbose and (i % max(1, nr//10) == 0):
            print(f"[Pi_comp table] {done}/{total} entries computed...")

    return r_grid, Ms2_grid, Pi_grid


def Pi_comp_interp_vec(r, Ms2, r_grid, Ms2_grid, Pi_grid):
    """
    Vectorized bilinear interpolation of Pi_comp over (r,Ms2).
    r, Ms2 arrays of same shape.
    """
    r = np.clip(r, r_grid[0], r_grid[-1])
    Ms2 = np.clip(Ms2, Ms2_grid[0], Ms2_grid[-1])

    ir = np.searchsorted(r_grid, r) - 1
    im = np.searchsorted(Ms2_grid, Ms2) - 1
    ir = np.clip(ir, 0, len(r_grid)-2).astype(int)
    im = np.clip(im, 0, len(Ms2_grid)-2).astype(int)

    r0 = r_grid[ir]
    r1 = r_grid[ir+1]
    m0 = Ms2_grid[im]
    m1 = Ms2_grid[im+1]

    t = (r - r0) / (r1 - r0 + 1e-30)
    s = (Ms2 - m0) / (m1 - m0 + 1e-30)

    P00 = Pi_grid[ir, im]
    P10 = Pi_grid[ir+1, im]
    P01 = Pi_grid[ir, im+1]
    P11 = Pi_grid[ir+1, im+1]

    return (1-t)*(1-s)*P00 + t*(1-s)*P10 + (1-t)*s*P01 + t*s*P11


# -------------------------
# Reconstruction: first / MUSCL van-Leer
# -------------------------
def vanleer_slope(dL, dR, eps=1e-14):
    a = np.abs(dL)
    b = np.abs(dR)
    return (np.sign(dL) + np.sign(dR)) * (a*b) / (a + b + eps)


def reconstruct(U, recon="first"):
    """
    Reconstruct left/right states at faces for U with ghost cells.
    Faces count = N-1 for N cells (full grid including ghosts).
    Returns (UL, UR) each shape (N-1,4).
    """
    if recon == "first":
        return U[:-1].copy(), U[1:].copy()

    if recon.lower() in ["vanleer", "van-leer", "muscl"]:
        slope = np.zeros_like(U)
        dR = U[2:] - U[1:-1]
        dL = U[1:-1] - U[:-2]
        slope[1:-1] = vanleer_slope(dL, dR)
        UL = (U[:-1] + 0.5*slope[:-1]).copy()
        UR = (U[1:]  - 0.5*slope[1:]).copy()
        return UL, UR

    raise ValueError(f"Unknown recon={recon}")


# -------------------------
# Solver
# -------------------------
class TwoFluid1D:
    def __init__(self, N=600, L=1.0, ng=2, eos=None, CFL=0.35, recon="vanleer", gravity=0.0):
        self.N = int(N)
        self.L = float(L)
        self.ng = int(ng)
        self.Ntot = self.N + 2*self.ng
        self.dx = self.L / self.N
        self.x = (np.arange(self.N) + 0.5) * self.dx

        self.eos = eos if eos is not None else BarotropicEOS()
        self.CFL = float(CFL)
        self.recon = recon
        self.g = float(gravity)

        self.U = np.zeros((self.Ntot, 4), dtype=float)

    def apply_bc_outflow(self):
        self.U[:self.ng] = self.U[self.ng:self.ng+1]
        self.U[-self.ng:] = self.U[-self.ng-1:-self.ng]

    def set_initial_contact(self, p0=7.5e6, x0=0.5,
                            left=dict(alpha_g=0.2, u_g=400.0, u_l=100.0),
                            right=dict(alpha_g=0.8, u_g=400.0, u_l=100.0)):
        x_full = (np.arange(self.Ntot) - self.ng + 0.5) * self.dx
        p = np.full(self.Ntot, p0, float)

        alpha_g = np.where(x_full < x0*self.L, left["alpha_g"], right["alpha_g"])
        alpha_g = np.clip(alpha_g, 1e-8, 1.0-1e-8)
        alpha_l = 1.0 - alpha_g

        u_g = np.where(x_full < x0*self.L, left["u_g"], right["u_g"])
        u_l = np.where(x_full < x0*self.L, left["u_l"], right["u_l"])

        rho_g = self.eos.rho_g(p)
        rho_l = self.eos.rho_l(p)

        m_g = alpha_g * rho_g
        m_l = alpha_l * rho_l
        q_g = m_g * u_g
        q_l = m_l * u_l

        self.U[:, 0] = m_g
        self.U[:, 1] = m_l
        self.U[:, 2] = q_g
        self.U[:, 3] = q_l

    def flux_from_state(self, U_state):
        """
        Conservative flux:
          F(m_g)=q_g
          F(m_l)=q_l
          F(q_g)=q_g*u_g + alpha_g*p
          F(q_l)=q_l*u_l + alpha_l*p
        """
        prim = primitives_from_U(U_state, self.eos)
        m_g = U_state[:, 0]
        m_l = U_state[:, 1]
        q_g = U_state[:, 2]
        q_l = U_state[:, 3]
        u_g = prim["u_g"]
        u_l = prim["u_l"]
        p = prim["p"]
        a_g = prim["alpha_g"]
        a_l = prim["alpha_l"]

        F = np.zeros_like(U_state)
        F[:, 0] = q_g
        F[:, 1] = q_l
        F[:, 2] = q_g*u_g + a_g*p
        F[:, 3] = q_l*u_l + a_l*p
        return F, prim

    def max_wave_speed(self, prim):
        ug = prim["u_g"][self.ng:-self.ng]
        ul = prim["u_l"][self.ng:-self.ng]
        return float(max(np.max(np.abs(ug) + self.eos.c_g),
                         np.max(np.abs(ul) + self.eos.c_l)))

    def compute_p_i_cell(self, prim, model, correction_strength=1.0, Pi_table=None):
        """
        Compute p_i at cell centers for a given model.
        model in {"none","inc","comp"}.
        """
        a_g = prim["alpha_g"]
        a_l = prim["alpha_l"]
        rg = prim["rho_g"]
        rl = prim["rho_l"]
        ug = prim["u_g"]
        ul = prim["u_l"]

        slip = ug - ul
        denom = a_l*rg + a_g*rl + 1e-30
        r = (a_l*rg - a_g*rl) / denom
        cm2 = self.eos.mixture_cm2(a_g, rg, a_l, rl)
        Ms2 = (0.5*slip)**2 / (cm2 + 1e-30)

        if model == "none":
            Pi = np.zeros_like(a_g)
        elif model == "inc":
            Pi = Pi_incompressible(r) * correction_strength
        elif model == "comp":
            if Pi_table is None:
                raise ValueError("model='comp' requires Pi_table=(r_grid,Ms2_grid,Pi_grid)")
            r_grid, Ms2_grid, Pi_grid = Pi_table
            Pi = Pi_comp_interp_vec(r, Ms2, r_grid, Ms2_grid, Pi_grid) *correction_strength
        else:
            raise ValueError(f"Unknown model={model}")

        # map Pi -> p_i (same scaling used before)
        p_i = denom * (slip*slip) * Pi
        return p_i

    def step(self, dt, ip_model="none", correction_strength=1.0, Pi_table=None):
        self.apply_bc_outflow()

        # Cell-centered flux + primitives
        F, prim = self.flux_from_state(self.U)

        # Reconstruction & face fluxes
        UL, UR = reconstruct(self.U, recon=self.recon)
        FL, _ = self.flux_from_state(UL)
        FR, _ = self.flux_from_state(UR)

        smax = self.max_wave_speed(prim)
        Fhat = 0.5*(FL + FR) - 0.5*smax*(UR - UL)  # faces: Ntot-1

        # ---------- Sources ----------
        S = np.zeros_like(self.U)

        # gravity (optional)
        if abs(self.g) > 0.0:
            S[:, 2] += self.U[:, 0] * self.g
            S[:, 3] += self.U[:, 1] * self.g

        # FIX B: nonconservative term as divergence of face jump contribution
        # Compute p_i at cell centers
        p_i_cell = self.compute_p_i_cell(prim, ip_model, correction_strength=correction_strength, Pi_table=Pi_table)
        pp_cell = prim["p"] - p_i_cell  # (p - p_i) at centers

        # Face coefficient: average (p-p_i) to faces
        pp_face = 0.5*(pp_cell[:-1] + pp_cell[1:])  # length Ntot-1

        # alpha jumps across faces
        a = prim["alpha_g"]
        da_face = a[1:] - a[:-1]  # length Ntot-1

        # Face contribution: C_{i+1/2} * (a_{i+1}-a_i)
        J_face = pp_face * da_face  # length Ntot-1

        # Divergence for each cell i: (J_{i+1/2} - J_{i-1/2})/dx
        nc = np.zeros(self.Ntot, dtype=float)
        # valid for interior i = 1..Ntot-2
        nc[1:-1] = (J_face[1:] - J_face[:-1]) / self.dx

        # Apply: gas gets -nc, liquid gets +nc
        S[:, 2] += -nc
        S[:, 3] += +nc

        # ---------- FV update ----------
        Un = self.U.copy()
        for i in range(self.ng, self.Ntot - self.ng):
            self.U[i] = (Un[i]
                         - (dt/self.dx) * (Fhat[i] - Fhat[i-1])
                         + dt * S[i])

        # Floors for safety
        self.U[:, 0] = np.maximum(self.U[:, 0], 1e-14)
        self.U[:, 1] = np.maximum(self.U[:, 1], 1e-14)

    def run(self, T=0.0015, correction_strength=1.0, ip_model="none", Pi_table=None):
        t = 0.0
        steps = 0
        while t < T:
            prim = primitives_from_U(self.U, self.eos)
            smax = self.max_wave_speed(prim)
            dt = self.CFL * self.dx / (smax + 1e-30)
            if t + dt > T:
                dt = T - t
            self.step(dt, ip_model=ip_model, correction_strength=correction_strength,Pi_table=Pi_table)
            t += dt
            steps += 1

        prim = primitives_from_U(self.U, self.eos)
        sl = slice(self.ng, self.Ntot - self.ng)
        out = {k: v[sl].copy() for k, v in prim.items()}
        out["x"] = self.x.copy()
        out["steps"] = steps
        return out


# -------------------------
# Run & compare 3 cases
# -------------------------
def summarize(label, sol):
    print(f"\n[{label}] steps={sol['steps']}")
    for nm in ["p", "alpha_g", "u_g", "u_l"]:
        v = sol[nm]
        print(f"  {nm:7s}: min={v.min(): .4e}, max={v.max(): .4e}")


def main():
    # --- Grid / time ---
    N = 1000
    L = 1.0
    T = 0.0015
    #recon = "first"   # "first" or "vanleer"
    recon = "vanleer"   # "first" or "vanleer"
    CFL = 0.15          # lower CFL helps reduce residual wiggles
    gravity = 0.0
    correction_strength = 2.0


    # --- EOS (steam-water-ish magnitudes; tune as you like) ---
    eos = BarotropicEOS(c_g=450.0, c_l=1500.0, rho_l0=1000.0, p0=1e5)

    # --- IC: slip contact (same p, jump in alpha) ---
    p_init = 7.5e6
    #p_init = 15e6

    left_state  = dict(alpha_g=0.2, u_g=400.0, u_l=100.0)
    right_state = dict(alpha_g=0.8, u_g=400.0, u_l=100.0)

    # --- Precompute Pi_comp table ---
    print("Building Pi_comp(r,Ms^2) table (one-time cost)...")
    Pi_table = build_Pi_comp_table(nr=161, nm=161, Ms2_max=1.0, verbose=True)

    def run_case(ip_model):
        sim = TwoFluid1D(N=N, L=L, ng=2, eos=eos, CFL=CFL, recon=recon, gravity=gravity)
        sim.set_initial_contact(p0=p_init, x0=0.5, left=left_state, right=right_state)
        return sim.run(T=T, ip_model=ip_model, correction_strength=correction_strength,Pi_table=Pi_table)

    sol_no   = run_case("none")
    sol_inc  = run_case("inc")
    sol_comp = run_case("comp")

    summarize("NO IP", sol_no)
    summarize("INCOMPRESSIBLE IP", sol_inc)
    summarize("COMPRESSIBLE IP", sol_comp)

    x = sol_no["x"]

    def plot_var(name, ylabel):
        plt.figure()
        plt.plot(x, sol_no[name], label="no IP")
        plt.plot(x, sol_inc[name], label="incompressible IP")
        plt.plot(x, sol_comp[name], label="compressible IP")
        plt.xlabel("x")
        plt.ylabel(ylabel)
        plt.title(f"Final {name} at t={T}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"voidFraction_contact_{name}_ip_strength_{correction_strength:.1f}.png")

    plot_var("p", "pressure [Pa]")
    plot_var("alpha_g", "alpha_g [-]")
    plot_var("u_g", "u_g [m/s]")
    plot_var("u_l", "u_l [m/s]")

    # Simple difference metrics vs compressible-IP reference
    def norms(a, b):
        d = a - b
        return np.mean(np.abs(d)), np.max(np.abs(d))

    for nm in ["p", "alpha_g", "u_g", "u_l"]:
        L1_no, Linf_no = norms(sol_no[nm], sol_comp[nm])
        L1_inc, Linf_inc = norms(sol_inc[nm], sol_comp[nm])
        print(f"\n[{nm}] vs compressible IP:")
        print(f"  no-IP:   L1={L1_no:.4e}, Linf={Linf_no:.4e}")
        print(f"  inc-IP:  L1={L1_inc:.4e}, Linf={Linf_inc:.4e}")

    plt.show()


if __name__ == "__main__":
    main()
