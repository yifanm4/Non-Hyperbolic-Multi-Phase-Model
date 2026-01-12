#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
water_faucet_compare.py

A simple 1-D "water faucet" benchmark for a barotropic compressible
1-pressure two-fluid model (4 equations: two masses + two momenta).

Compares primitive-variable profiles at final time for three IP models:
  (A) no IP
  (B) incompressible IP (minimum)
  (C) compressible IP (minimum; via precomputed Pi_comp(r,Ms^2) table)

Boundary conditions (as requested):
  Inlet (x=0):  alpha_g, u_g, u_l fixed; pressure extrapolated
  Outlet (x=L): pressure fixed (p_out); alpha_g, u_g, u_l extrapolated

Numerics:
  - finite-volume, forward Euler
  - Rusanov flux
  - FIX B: nonconservative term discretized as face-jump divergence
  - reconstruction: "first" (robust default) or "vanleer"

Run:
  python water_faucet_compare.py

Dependencies:
  pip install numpy matplotlib
"""

from __future__ import annotations
import os
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
        Solve p>0 from: m_g/(p/c_g^2) + m_l/(a + b p) = 1
        Quadratic: b p^2 + (a - m_g c_g^2 b - m_l) p - (m_g c_g^2 a) = 0
        """
        mg = np.maximum(m_g, 1e-14)
        ml = np.maximum(m_l, 1e-14)

        A = self.b
        B = self.a - (mg * self.c_g2 * self.b) - ml
        C = -(mg * self.c_g2 * self.a)

        D = np.maximum(B*B - 4.0*A*C, 0.0)
        sqrtD = np.sqrt(D)

        p1 = (-B + sqrtD) / (2.0*A)
        p2 = (-B - sqrtD) / (2.0*A)

        p = np.full_like(p1, max(p_floor, 1.0), dtype=float)
        p1_ok = np.isfinite(p1) & (p1 > p_floor)
        p2_ok = np.isfinite(p2) & (p2 > p_floor)

        both = p1_ok & p2_ok
        p[both] = np.minimum(p1[both], p2[both])
        only1 = p1_ok & (~p2_ok)
        p[only1] = p1[only1]
        only2 = p2_ok & (~p1_ok)
        p[only2] = p2[only2]
        return p

    def mixture_cm2(self, alpha_g, rho_g, alpha_l, rho_l):
        num = self.c_g2 * self.c_l2 * (alpha_l * rho_g + alpha_g * rho_l)
        den = self.c_g2 * (alpha_l * rho_g) + self.c_l2 * (alpha_g * rho_l)
        return num / (den + 1e-30)


def primitives_from_U(U, eos: BarotropicEOS):
    m_g, m_l, q_g, q_l = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    p = eos.pressure_from_masses_vec(m_g, m_l, p_floor=1.0)
    rho_g = eos.rho_g(p)
    rho_l = eos.rho_l(p)
    alpha_g = np.clip(m_g / (rho_g + 1e-30), 1e-8, 1.0 - 1e-8)
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


def build_Pi_comp_table(r_min=-0.999, r_max=0.999, nr=121,
                        Ms2_max=1.0, nm=121,
                        tol_imag=1e-10, verbose=True):
    """
    Precompute Pi_comp(r,Ms^2) = minimal Pi that makes the characteristic quartic all-real.
    Quartic in Z:
      (Z^2 - 2 r Z + 1 - 4 Pi) - Ms^2 (1 - Z^2)^2 = 0
    """
    r_grid = np.linspace(r_min, r_max, nr)
    Ms2_grid = np.linspace(0.0, Ms2_max, nm)
    Pi_grid = np.zeros((nr, nm), dtype=float)

    def all_real_roots(Ms2, r, Pi):
        # -Ms^2 Z^4 + (2Ms^2+1) Z^2 - 2 r Z + (1 - Ms^2 - 4 Pi) = 0
        coefs = np.array([-Ms2, 0.0, (2.0*Ms2 + 1.0), (-2.0*r), (1.0 - Ms2 - 4.0*Pi)], float)
        z = np.roots(coefs)
        return np.max(np.abs(np.imag(z))) <= tol_imag

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

        # bisection fallback
        Pi_lo = 0.0
        Pi_hi = max(Pi_incompressible(r), 1e-10)
        for _ in range(60):
            if all_real_roots(Ms2, r, Pi_hi):
                break
            Pi_hi *= 2.0
        for _ in range(70):
            Pi_mid = 0.5*(Pi_lo + Pi_hi)
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
# Reconstruction: first / MUSCL van-Leer (conserved)
# -------------------------
def vanleer_slope(dL, dR, eps=1e-14):
    a, b = np.abs(dL), np.abs(dR)
    return (np.sign(dL) + np.sign(dR)) * (a*b) / (a + b + eps)


def reconstruct(U, recon="first"):
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
# Faucet solver
# -------------------------
class WaterFaucet1D:
    def __init__(self, N=200, L=12.0, ng=2, eos=None,
                 CFL=0.25, recon="first", g=9.81,
                 alpha_g_in=0.2, u_g_in=5.0, u_l_in=5.0,
                 p_out=7.5e6,
                 Pi_table=None):
        self.N = int(N)
        self.L = float(L)
        self.ng = int(ng)
        self.Ntot = self.N + 2*self.ng
        self.dx = self.L / self.N
        self.x = (np.arange(self.N) + 0.5) * self.dx

        self.eos = eos if eos is not None else BarotropicEOS()
        self.CFL = float(CFL)
        self.recon = recon
        self.g = float(g)

        self.alpha_g_in = float(alpha_g_in)
        self.u_g_in = float(u_g_in)
        self.u_l_in = float(u_l_in)
        self.p_out = float(p_out)

        self.Pi_table = Pi_table

        self.U = np.zeros((self.Ntot, 4), dtype=float)

    def _state_from_primitives(self, p, alpha_g, u_g, u_l):
        alpha_g = np.clip(alpha_g, 1e-8, 1.0-1e-8)
        alpha_l = 1.0 - alpha_g
        rho_g = self.eos.rho_g(p)
        rho_l = self.eos.rho_l(p)
        m_g = alpha_g * rho_g
        m_l = alpha_l * rho_l
        q_g = m_g * u_g
        q_l = m_l * u_l
        return np.array([m_g, m_l, q_g, q_l], dtype=float)

    def apply_bc_faucet(self, prim):
        # interior indices
        iL = self.ng
        iR = self.Ntot - self.ng - 1

        # inlet: p extrapolated from first interior, others fixed
        p_in = float(prim["p"][iL])
        ag_in = self.alpha_g_in
        ug_in = self.u_g_in
        ul_in = self.u_l_in

        # outlet: p fixed, others extrapolated
        p_out = self.p_out
        ag_out = float(prim["alpha_g"][iR])
        ug_out = float(prim["u_g"][iR])
        ul_out = float(prim["u_l"][iR])

        for j in range(self.ng):
            self.U[self.ng-1-j] = self._state_from_primitives(p_in, ag_in, ug_in, ul_in)
            self.U[self.Ntot-self.ng+j] = self._state_from_primitives(p_out, ag_out, ug_out, ul_out)

    def set_initial_uniform(self, p0=None, alpha_g0=None, ug0=None, ul0=None):
        if p0 is None:
            p0 = self.p_out
        if alpha_g0 is None:
            alpha_g0 = self.alpha_g_in
        if ug0 is None:
            ug0 = self.u_g_in
        if ul0 is None:
            ul0 = self.u_l_in

        p = np.full(self.Ntot, float(p0))
        ag = np.full(self.Ntot, float(alpha_g0))
        ug = np.full(self.Ntot, float(ug0))
        ul = np.full(self.Ntot, float(ul0))

        rg = self.eos.rho_g(p)
        rl = self.eos.rho_l(p)

        self.U[:, 0] = ag * rg
        self.U[:, 1] = (1.0 - ag) * rl
        self.U[:, 2] = self.U[:, 0] * ug
        self.U[:, 3] = self.U[:, 1] * ul

    def flux_from_state(self, U_state):
        prim = primitives_from_U(U_state, self.eos)
        m_g, m_l, q_g, q_l = U_state[:, 0], U_state[:, 1], U_state[:, 2], U_state[:, 3]
        u_g, u_l = prim["u_g"], prim["u_l"]
        p = prim["p"]
        a_g, a_l = prim["alpha_g"], prim["alpha_l"]

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

    def compute_p_i_cell(self, prim, ip_model):
        a_g, a_l = prim["alpha_g"], prim["alpha_l"]
        rg, rl = prim["rho_g"], prim["rho_l"]
        ug, ul = prim["u_g"], prim["u_l"]

        slip = ug - ul
        denom = a_l*rg + a_g*rl + 1e-30
        r = (a_l*rg - a_g*rl) / denom
        cm2 = self.eos.mixture_cm2(a_g, rg, a_l, rl)
        Ms2 = (0.5*slip)**2 / (cm2 + 1e-30)

        if ip_model == "none":
            Pi = np.zeros_like(a_g)
        elif ip_model == "inc":
            Pi = Pi_incompressible(r)
        elif ip_model == "comp":
            if self.Pi_table is None:
                raise ValueError("ip_model='comp' requires Pi_table")
            r_grid, Ms2_grid, Pi_grid = self.Pi_table
            Pi = Pi_comp_interp_vec(r, Ms2, r_grid, Ms2_grid, Pi_grid)
        else:
            raise ValueError(f"Unknown ip_model={ip_model}")

        p_i = denom * (slip*slip) * Pi
        return p_i

    def step(self, dt, ip_model):
        # primitives, BC
        prim = primitives_from_U(self.U, self.eos)
        self.apply_bc_faucet(prim)

        # fluxes
        F, prim = self.flux_from_state(self.U)
        UL, UR = reconstruct(self.U, recon=self.recon)
        FL, _ = self.flux_from_state(UL)
        FR, _ = self.flux_from_state(UR)

        smax = self.max_wave_speed(prim)
        Fhat = 0.5*(FL + FR) - 0.5*smax*(UR - UL)

        # sources
        S = np.zeros_like(self.U)
        S[:, 2] += self.U[:, 0] * self.g
        S[:, 3] += self.U[:, 1] * self.g

        # FIX B: face-jump divergence for (p - p_i) ∂x alpha_g
        p_i = self.compute_p_i_cell(prim, ip_model=ip_model)
        pp_cell = prim["p"] - p_i
        pp_face = 0.5*(pp_cell[:-1] + pp_cell[1:])

        a = prim["alpha_g"]
        da_face = a[1:] - a[:-1]
        J_face = pp_face * da_face

        nc = np.zeros(self.Ntot, dtype=float)
        nc[1:-1] = (J_face[1:] - J_face[:-1]) / self.dx

        S[:, 2] += -nc
        S[:, 3] += +nc

        # update
        Un = self.U.copy()
        for i in range(self.ng, self.Ntot - self.ng):
            self.U[i] = (Un[i]
                         - (dt/self.dx)*(Fhat[i] - Fhat[i-1])
                         + dt*S[i])

        self.U[:, 0] = np.maximum(self.U[:, 0], 1e-14)
        self.U[:, 1] = np.maximum(self.U[:, 1], 1e-14)

    def run(self, T=0.05, ip_model="none"):
        t = 0.0
        steps = 0
        while t < T:
            prim = primitives_from_U(self.U, self.eos)
            smax = self.max_wave_speed(prim)
            dt = self.CFL * self.dx / (smax + 1e-30)
            if t + dt > T:
                dt = T - t
            self.step(dt, ip_model=ip_model)
            t += dt
            steps += 1

        prim = primitives_from_U(self.U, self.eos)
        sl = slice(self.ng, self.Ntot - self.ng)
        out = {k: v[sl].copy() for k, v in prim.items()}
        out["x"] = self.x.copy()
        out["steps"] = steps
        return out


# -------------------------
# plotting / reporting
# -------------------------
def summarize(label, sol):
    print(f"\n[{label}] steps={sol['steps']}")
    for nm in ["p", "alpha_g", "u_g", "u_l"]:
        v = sol[nm]
        print(f"  {nm:7s}: min={v.min(): .4e}, max={v.max(): .4e}")


def plot_profiles(x, sols, title_prefix):
    def plot_one(var, ylabel):
        plt.figure()
        for lab, sol in sols.items():
            plt.plot(x, sol[var], label=lab)
        plt.xlabel("x [m]")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} | {var}(x) at final time")
        plt.legend()
        plt.tight_layout()

    plot_one("p", "pressure [Pa]")
    plot_one("alpha_g", "alpha_g [-]")
    plot_one("u_g", "u_g [m/s]")
    plot_one("u_l", "u_l [m/s]")


def main():
    # --- Faucet numerical setup ---
    N = 200
    L = 12.0
    T = 0.05
    recon = "first"   # start robust; try "vanleer" after
    CFL = 0.25
    g = 9.81

    # inlet (fixed)
    alpha_g_in = 0.2
    u_g_in = 5.0
    u_l_in = 5.0

    # outlet pressures to compare (common in literature)
    outlet_pressures = [7.5e6, 15e6]

    eos = BarotropicEOS(c_g=450.0, c_l=1500.0, rho_l0=1000.0, p0=1e5)

    # --- Build/load Pi_comp table (cache) ---
    cache = "Pi_comp_cache.npz"
    if os.path.exists(cache):
        print(f"Loading cached Pi_comp table: {cache}")
        data = np.load(cache)
        Pi_table = (data["r_grid"], data["Ms2_grid"], data["Pi_grid"])
    else:
        print("Building Pi_comp(r,Ms^2) table (one-time cost)...")
        Pi_table = build_Pi_comp_table(nr=121, nm=121, Ms2_max=1.0, verbose=True)
        r_grid, Ms2_grid, Pi_grid = Pi_table
        np.savez(cache, r_grid=r_grid, Ms2_grid=Ms2_grid, Pi_grid=Pi_grid)
        print(f"Saved cache to {cache}")

    for p_out in outlet_pressures:
        print("\n" + "="*80)
        print(f"Water faucet comparison: p_out = {p_out/1e6:.2f} MPa")
        print("="*80)

        def run_case(ip_model):
            sim = WaterFaucet1D(
                N=N, L=L, ng=2, eos=eos,
                CFL=CFL, recon=recon, g=g,
                alpha_g_in=alpha_g_in, u_g_in=u_g_in, u_l_in=u_l_in,
                p_out=p_out, Pi_table=Pi_table
            )
            sim.set_initial_uniform(p0=p_out, alpha_g0=alpha_g_in, ug0=u_g_in, ul0=u_l_in)
            return sim.run(T=T, ip_model=ip_model)

        sol_no = run_case("none")
        sol_inc = run_case("inc")
        sol_comp = run_case("comp")

        summarize("NO IP", sol_no)
        summarize("INCOMPRESSIBLE IP", sol_inc)
        summarize("COMPRESSIBLE IP", sol_comp)

        x = sol_no["x"]
        sols = {"no IP": sol_no, "inc IP": sol_inc, "comp IP": sol_comp}
        plot_profiles(x, sols, title_prefix=f"Faucet p_out={p_out/1e6:.2f}MPa, T={T}s, recon={recon}")

    plt.show()


if __name__ == "__main__":
    main()
