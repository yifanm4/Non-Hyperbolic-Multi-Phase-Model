import numpy as np
import matplotlib.pyplot as plt

class FullPressureTwoFluidSolver:
    def __init__(self, n_cells=100, L=12.0, t_final=0.5):
        self.N = n_cells
        self.L = L
        self.dx = L / n_cells
        self.t_final = t_final
        self.x = np.linspace(self.dx/2, L - self.dx/2, self.N)

        # Physics Constants
        self.rho_l = 1000.0
        self.g = 9.81
        self.alpha0 = 0.2
        self.v0 = 10.0

        # State Arrays
        self.alpha = np.ones(self.N) * self.alpha0
        self.u_l = np.ones(self.N) * self.v0

        # History for Entropy Viscosity
        self.alpha_hist = [self.alpha.copy()] * 3

    def van_leer(self, r):
        # Van Leer Slope Limiter
        # phi(r) = (r + |r|) / (1 + |r|)
        return (r + np.abs(r)) / (1.0 + np.abs(r))

    def limited_slope(self, Q):
        """
        Computes limited slopes for second-order reconstructions (Van Leer limiter).
        Falls back to first-order near boundaries.
        """
        slope = np.zeros_like(Q)
        for i in range(1, self.N - 1):
            d_minus = Q[i] - Q[i-1]
            d_plus  = Q[i+1] - Q[i]
            if abs(d_minus) < 1e-12 or abs(d_plus) < 1e-12:
                continue
            r = d_plus / d_minus
            phi = self.van_leer(r)
            slope[i] = phi * d_minus
        return slope

    def pp_limit_slopes_mass(self, mass, slope_mass, slope_mom, m_min=1e-10, m_max=None):
        """
        Positivity/bound-preserving slope limiter for MUSCL-type reconstructions.

        Ensures the reconstructed interface values
            mass_i^- = mass_i - 0.5*slope_mass_i
            mass_i^+ = mass_i + 0.5*slope_mass_i
        satisfy:  m_min <= mass_i^-, mass_i^+ <= m_max  (when m_max is not None).

        We rescale BOTH mass and momentum slopes by the SAME factor so that face velocities
        mom/mass do not blow up when mass is limited.
        """
        if m_max is None:
            m_max = np.inf

        sm = slope_mass.copy()
        sp = slope_mom.copy()

        for i in range(self.N):
            m = mass[i]
            if not np.isfinite(m) or m <= m_min:
                sm[i] = 0.0
                sp[i] = 0.0
                continue

            m_minus = m - 0.5 * sm[i]
            m_plus  = m + 0.5 * sm[i]

            theta = 1.0

            # Lower bound enforcement
            if m_minus < m_min:
                denom = m - m_minus
                if denom > 0:
                    theta = min(theta, (m - m_min) / denom)
            if m_plus < m_min:
                denom = m - m_plus
                if denom > 0:
                    theta = min(theta, (m - m_min) / denom)

            # Upper bound enforcement (optional, helps keep alpha in [0,1])
            if m_minus > m_max:
                denom = m_minus - m
                if denom > 0:
                    theta = min(theta, (m_max - m) / denom)
            if m_plus > m_max:
                denom = m_plus - m
                if denom > 0:
                    theta = min(theta, (m_max - m) / denom)

            theta = max(0.0, min(1.0, theta))
            sm[i] *= theta
            sp[i] *= theta

        return sm, sp

    def compute_entropy_viscosity(self, dt, v_field):
        # 1. Update History
        self.alpha_hist.append(self.alpha.copy())
        if len(self.alpha_hist) > 3: self.alpha_hist.pop(0)

        if len(self.alpha_hist) < 3:
            return np.zeros(self.N + 1)

        # 2. Compute Residual
        if dt < 1e-12: return np.zeros(self.N + 1)

        d_dt = (3*self.alpha_hist[-1] - 4*self.alpha_hist[-2] + self.alpha_hist[-3]) / (2*dt)

        f = self.alpha * v_field
        df_dx = np.zeros(self.N)
        df_dx[1:-1] = (f[2:] - f[:-2]) / (2*self.dx)
        df_dx[0] = (f[1] - f[0]) / self.dx
        df_dx[-1] = (f[-1] - f[-2]) / self.dx

        Residual = np.abs(d_dt + df_dx)

        # 3. Normalization
        norm = np.max(np.abs(self.alpha - np.mean(self.alpha))) + 1e-10

        # 4. Coefficients
        #c_E = 1.0 # works great already
        c_E = 1.0
        nu_E = c_E * (self.dx**2) * Residual / norm

        c_max = 0.5
        nu_max = c_max * self.dx * np.abs(v_field)
        nu_cell = np.minimum(nu_max, nu_E)

        # Map to Faces
        nu_face = np.zeros(self.N + 1)
        nu_face[1:-1] = 0.5 * (nu_cell[1:] + nu_cell[:-1])
        nu_face[0] = nu_cell[0]
        nu_face[-1] = nu_cell[-1]

        return nu_face

    def check_stability(self):
        if np.any(np.isnan(self.u_l)) or np.any(np.isinf(self.u_l)):
            return False
        if np.any(np.isnan(self.alpha)) or np.any(np.isinf(self.alpha)):
            return False
        if np.max(np.abs(self.u_l)) > 1e4:
            return False
        return True

    def reconstruct_MUSCL(self, Q):
        """
        Reconstructs Left state at face i using MUSCL with Van Leer Limiter
        """
        Q_L = np.zeros(self.N + 1)
        # Loop faces
        for i in range(1, self.N):
            if i < 2:
                # Fallback to 1st order at inlet boundary
                Q_L[i] = Q[i-1]
            else:
                # Calc slopes
                d_minus = Q[i-1] - Q[i-2]
                d_plus  = Q[i] - Q[i-1]

                if abs(d_minus) < 1e-10: r = 0.0
                else: r = d_plus / d_minus

                # Apply Limiter
                phi = self.van_leer(r)

                # Reconstruct
                Q_L[i] = Q[i-1] + 0.5 * phi * d_minus

        Q_L[0] = Q[0]
        Q_L[-1] = Q[-1]
        return Q_L

    def reconstruct_QUICK(self, Q):
        """
        Quadratic Upwind Interpolation (QUICK) assuming predominantly positive flow.
        Falls back to 1st order near boundaries or if velocity reverses.
        """
        Q_face = np.zeros(self.N + 1)
        # Positive-flow-biased stencil: face i between cells i-1 and i
        for i in range(1, self.N):
            if i < 3:
                Q_face[i] = Q[i-1]  # fallback near inlet
            else:
                Q_face[i] = (3*Q[i-1] + 6*Q[i-2] - Q[i-3]) / 8.0
        Q_face[0] = Q[0]
        Q_face[-1] = Q[-1]
        return Q_face

    def reconstruct_ENO2(self, Q):
        """
        Second-order ENO reconstruction for left state at face i (assumes positive flow).
        Chooses the smoother stencil between left and right differences.
        """
        Q_face = np.zeros(self.N + 1)
        for i in range(1, self.N):
            if i < 2 or i > self.N - 2:
                Q_face[i] = Q[i-1]  # fallback near boundaries
                continue

            d_left = Q[i-1] - Q[i-2]
            d_right = Q[i] - Q[i-1]
            d_more = Q[i+1] - Q[i]

            # Choose smoother slope between {d_left, d_right} vs {d_right, d_more}
            smooth_left = abs(d_right - d_left)
            smooth_right = abs(d_more - d_right)

            if smooth_left <= smooth_right:
                slope = d_right
                Q_face[i] = Q[i-1] + 0.5 * slope
            else:
                slope = d_more
                Q_face[i] = Q[i] - 0.5 * slope

        Q_face[0] = Q[0]
        Q_face[-1] = Q[-1]
        return Q_face

    def reconstruct_WENO3(self, Q):
        """
        Third-order WENO (2-point stencils) for left state at face i.
        Bias assumes positive advection direction.
        """
        eps = 1e-6
        Q_face = np.zeros(self.N + 1)
        for i in range(1, self.N):
            if i < 2 or i > self.N - 2:
                Q_face[i] = Q[i-1]
                continue

            v0 = 0.5 * (Q[i-2] + Q[i-1])      # stencil S0
            v1 = 0.5 * (-Q[i-1] + 3*Q[i])     # stencil S1

            beta0 = (Q[i-1] - Q[i-2])**2
            beta1 = (Q[i] - Q[i-1])**2

            alpha0 = 1.0 / (eps + beta0)**2
            alpha1 = 2.0 / (eps + beta1)**2  # give downstream stencil larger linear weight
            w0 = alpha0 / (alpha0 + alpha1)
            w1 = alpha1 / (alpha0 + alpha1)

            Q_face[i] = w0 * v0 + w1 * v1

        Q_face[0] = Q[0]
        Q_face[-1] = Q[-1]
        return Q_face

    def reconstruct_CWENO3(self, Q):
        """
        Symmetric central WENO-3: average left/right-biased WENO3 reconstructions.
        Provides a less upwind-biased smooth profile.
        """
        left = self.reconstruct_WENO3(Q)

        # Right-biased WENO3 by flipping indices around the face
        eps = 1e-6
        right = np.zeros(self.N + 1)
        for i in range(1, self.N):
            if i < 2 or i > self.N - 2:
                right[i] = Q[i]
                continue

            v0 = 0.5 * (Q[i+1] + Q[i])        # stencil to the right
            v1 = 0.5 * (-Q[i] + 3*Q[i-1])     # stencil to the left (mirrored)

            beta0 = (Q[i+1] - Q[i])**2
            beta1 = (Q[i] - Q[i-1])**2

            alpha0 = 1.0 / (eps + beta0)**2
            alpha1 = 2.0 / (eps + beta1)**2
            w0 = alpha0 / (alpha0 + alpha1)
            w1 = alpha1 / (alpha0 + alpha1)

            right[i] = w0 * v0 + w1 * v1

        right[0] = Q[0]
        right[-1] = Q[-1]

        return 0.5 * (left + right)

    def solve(self, scheme='Upwind1'):
        # Reset State
        self.alpha = np.ones(self.N) * self.alpha0
        self.u_l = np.ones(self.N) * self.v0
        self.alpha_hist = [self.alpha.copy()] * 3

        t = 0.0
        CFL = 0.1
        if any(tag in scheme for tag in ['ENO', 'WENO', 'CWENO', 'KU', 'PCU']):
            CFL = 0.05  # tighten timestep for high-order / central-upwind schemes
        visc_record = np.zeros(self.N + 1)

        while t < self.t_final:
            if not self.check_stability():
                print(f"  [Warning] Simulation blew up at t={t:.4f}s with scheme {scheme}")
                break

            # 1. Adaptive Time Step
            max_v = np.max(np.abs(self.u_l)) + 1e-6
            dt = CFL * self.dx / max_v
            if t + dt > self.t_final: dt = self.t_final - t

            # --- 2. Compute Viscosity (if EVM enabled) ---
            if 'EVM' in scheme:
                nu_visc = self.compute_entropy_viscosity(dt, self.u_l)
                visc_record = np.maximum(visc_record, nu_visc)
            else:
                nu_visc = np.zeros(self.N + 1)

            # --- 3. Compute Fluxes ---
            flux_mass = np.zeros(self.N + 1)
            flux_mom  = np.zeros(self.N + 1)

            mass = (1 - self.alpha) * self.rho_l
            mom  = mass * self.u_l

            # Pre-calculate Reconstructions if needed
            if 'MUSCL' in scheme:
                rho_muscl = self.reconstruct_MUSCL(mass)
                u_muscl   = self.reconstruct_MUSCL(self.u_l)
            if 'QUICK' in scheme:
                rho_quick = self.reconstruct_QUICK(mass)
                u_quick   = self.reconstruct_QUICK(self.u_l)
            if ('NT' in scheme) or ('KU' in scheme) or ('PCU' in scheme):
                slope_mass = self.limited_slope(mass)
                slope_mom  = self.limited_slope(mom)
                # Make the reconstruction truly bound/positivity-preserving for mass
                slope_mass, slope_mom = self.pp_limit_slopes_mass(
                    mass, slope_mass, slope_mom,
                    m_min=1e-10,
                    m_max=self.rho_l - 1e-10,
                )
            if 'ENO' in scheme:
                rho_eno = self.reconstruct_ENO2(mass)
                u_eno   = self.reconstruct_ENO2(self.u_l)
            if 'WENO' in scheme and 'CWENO' not in scheme:
                rho_weno = self.reconstruct_WENO3(mass)
                u_weno   = self.reconstruct_WENO3(self.u_l)
            if 'CWENO' in scheme:
                rho_cweno = self.reconstruct_CWENO3(mass)
                u_cweno   = self.reconstruct_CWENO3(self.u_l)

            # Add a small Rusanov-like baseline viscosity for high-order schemes to prevent blow-up
            if any(tag in scheme for tag in ['ENO', 'WENO', 'CWENO', 'KU', 'PCU']):
                face_speed = np.zeros(self.N + 1)
                face_speed[1:-1] = 0.5 * (np.abs(self.u_l[:-1]) + np.abs(self.u_l[1:]))
                face_speed[0] = np.abs(self.v0)
                face_speed[-1] = face_speed[-2]
                c_base = 0.2
                nu_visc = np.maximum(nu_visc, c_base * self.dx * face_speed)

            for i in range(1, self.N):
                # Basic Neighbors
                rho_L = mass[i-1]
                rho_R = mass[i]
                u_L = self.u_l[i-1]
                u_R = self.u_l[i]
                # Defaults (used by non-CU schemes and for diffusion term)
                mass_L = rho_L
                mass_R = rho_R
                mom_L  = mom[i-1]
                mom_R  = mom[i]

                # --- FLUX SELECTION ---

                if 'Upwind1' in scheme:
                    rho_face = rho_L
                    u_face   = u_L

                elif 'Central' in scheme:
                    rho_face = 0.5 * (rho_L + rho_R)
                    u_face   = 0.5 * (u_L + u_R)

                elif 'MUSCL' in scheme:
                    # Use the limited reconstructed values
                    rho_face = rho_muscl[i]
                    u_face   = u_muscl[i]

                elif 'QUICK' in scheme:
                    rho_face = rho_quick[i]
                    u_face   = u_quick[i]

                elif 'ENO' in scheme:
                    rho_face = rho_eno[i]
                    u_face   = u_eno[i]

                elif 'CWENO' in scheme:
                    rho_face = rho_cweno[i]
                    u_face   = u_cweno[i]

                elif 'WENO' in scheme:
                    rho_face = rho_weno[i]
                    u_face   = u_weno[i]

                elif 'KU' in scheme:
                    # Positivity-preserving central-upwind (Kurganov-type) flux
                    mass_L = mass[i-1] + 0.5 * slope_mass[i-1]
                    mass_R = mass[i]   - 0.5 * slope_mass[i]
                    mom_L  = mom[i-1]  + 0.5 * slope_mom[i-1]
                    mom_R  = mom[i]    - 0.5 * slope_mom[i]

                    u_L_face = mom_L / mass_L
                    u_R_face = mom_R / mass_R

                    f_mass_L = mom_L
                    f_mass_R = mom_R
                    f_mom_L  = mom_L * u_L_face
                    f_mom_R  = mom_R * u_R_face

                    a_plus  = max(u_L_face, u_R_face, 0.0)
                    a_minus = min(u_L_face, u_R_face, 0.0)
                    denom = a_plus - a_minus

                    if denom < 1e-14:
                        # fallback to local Lax-Friedrichs at nearly-stationary interfaces
                        s = max(abs(u_L_face), abs(u_R_face)) + 1e-14
                        f_mass_conv = 0.5 * (f_mass_L + f_mass_R) - 0.5 * s * (mass_R - mass_L)
                        f_mom_conv  = 0.5 * (f_mom_L  + f_mom_R ) - 0.5 * s * (mom_R  - mom_L )
                    else:
                        f_mass_conv = (a_plus * f_mass_L - a_minus * f_mass_R + a_plus * a_minus * (mass_R - mass_L)) / denom
                        f_mom_conv  = (a_plus * f_mom_L  - a_minus * f_mom_R  + a_plus * a_minus * (mom_R  - mom_L )) / denom
                elif 'PCU' in scheme:
                    # True positivity-preserving central-upwind FV flux.
                    # NOTE: In this simplified faucet model the PDE is conservative; there is no
                    # nonconservative product to integrate along a path, so a genuine path term is zero.
                    mass_L = mass[i-1] + 0.5 * slope_mass[i-1]
                    mass_R = mass[i]   - 0.5 * slope_mass[i]
                    mom_L  = mom[i-1]  + 0.5 * slope_mom[i-1]
                    mom_R  = mom[i]    - 0.5 * slope_mom[i]

                    u_L_face = mom_L / mass_L
                    u_R_face = mom_R / mass_R

                    f_mass_L = mom_L
                    f_mass_R = mom_R
                    f_mom_L  = mom_L * u_L_face
                    f_mom_R  = mom_R * u_R_face

                    a_plus  = max(u_L_face, u_R_face, 0.0)
                    a_minus = min(u_L_face, u_R_face, 0.0)
                    denom = a_plus - a_minus

                    if denom < 1e-14:
                        s = max(abs(u_L_face), abs(u_R_face)) + 1e-14
                        f_mass_conv = 0.5 * (f_mass_L + f_mass_R) - 0.5 * s * (mass_R - mass_L)
                        f_mom_conv  = 0.5 * (f_mom_L  + f_mom_R ) - 0.5 * s * (mom_R  - mom_L )
                    else:
                        f_mass_conv = (a_plus * f_mass_L - a_minus * f_mass_R + a_plus * a_minus * (mass_R - mass_L)) / denom
                        f_mom_conv  = (a_plus * f_mom_L  - a_minus * f_mom_R  + a_plus * a_minus * (mom_R  - mom_L )) / denom
                elif 'NT' in scheme:
                    # Nessyahu-Tadmor 2nd-order central flux (Lax-Friedrichs form)
                    mass_minus = mass[i-1] + 0.5 * slope_mass[i-1]
                    mass_plus  = mass[i]   - 0.5 * slope_mass[i]
                    mom_minus  = mom[i-1]  + 0.5 * slope_mom[i-1]
                    mom_plus   = mom[i]    - 0.5 * slope_mom[i]
                    # Use reconstructed states for the viscosity term as well
                    mass_L = mass_minus
                    mass_R = mass_plus
                    mom_L  = mom_minus
                    mom_R  = mom_plus

                    u_minus = mom_minus / mass_minus if mass_minus > 1e-8 else 0.0
                    u_plus  = mom_plus  / mass_plus  if mass_plus  > 1e-8 else 0.0

                    f_mass_minus = mass_minus * u_minus
                    f_mass_plus  = mass_plus  * u_plus
                    f_mom_minus  = mom_minus  * u_minus
                    f_mom_plus   = mom_plus   * u_plus

                    a_face = max(abs(u_minus), abs(u_plus)) + 1e-6

                    f_mass_conv = 0.5 * (f_mass_minus + f_mass_plus) - 0.5 * a_face * (mass_plus - mass_minus)
                    f_mom_conv  = 0.5 * (f_mom_minus + f_mom_plus)   - 0.5 * a_face * (mom_plus - mom_minus)

                # Convective Flux
                if ('NT' not in scheme) and ('KU' not in scheme) and ('PCU' not in scheme):
                    f_mass_conv = rho_face * u_face
                    f_mom_conv  = rho_face * u_face**2

                # --- ADD DIFFUSION / ARTIFICIAL VISCOSITY (conservative form) ---
                # Interpret nu_visc as a face diffusivity and write it as an additional
                # Rusanov/Lax-Friedrichs dissipation term using the (possibly reconstructed) jump.
                s_visc = 2.0 * nu_visc[i] / self.dx  # [m/s]

                flux_mass[i] = f_mass_conv - 0.5 * s_visc * (mass_R - mass_L)
                flux_mom[i]  = f_mom_conv  - 0.5 * s_visc * (mom_R  - mom_L )


            # Boundary Fluxes
            rho_in = (1.0 - self.alpha0)*self.rho_l
            flux_mass[0] = rho_in * self.v0
            flux_mom[0]  = rho_in * self.v0**2
            flux_mass[-1] = flux_mass[-2]
            flux_mom[-1]  = flux_mom[-2]

            # --- 4. Update Equations ---
            div_mass = (flux_mass[1:] - flux_mass[:-1]) / self.dx
            div_mom  = (flux_mom[1:] - flux_mom[:-1]) / self.dx

            force_grav = (1.0 - self.alpha) * self.rho_l * self.g

            mass_new = (1-self.alpha)*self.rho_l - dt * div_mass
            mom_new  = (1-self.alpha)*self.rho_l*self.u_l + dt * (-div_mom + force_grav)

            self.alpha = 1.0 - (mass_new / self.rho_l)
            self.alpha = np.clip(self.alpha, 0.0, 1.0)

            self.u_l = np.zeros_like(mass_new)
            mask = mass_new > 1e-6
            self.u_l[mask] = mom_new[mask] / mass_new[mask]

            t += dt

        visc_centers = 0.5 * (visc_record[1:] + visc_record[:-1])
        return self.x, self.alpha, visc_centers

    def analytical_solution(self, t_current):
        alpha_exact = np.zeros_like(self.x)
        x_front = self.v0 * t_current + 0.5 * self.g * t_current**2
        for i, xi in enumerate(self.x):
            if xi < x_front:
                v = np.sqrt(self.v0**2 + 2 * self.g * xi)
                alpha_exact[i] = 1.0 - ((1.0 - self.alpha0) * self.v0) / v
            else:
                alpha_exact[i] = self.alpha0
        return self.x, alpha_exact


import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# Utilities (ported/adapted from your water-faucet test driver)
# ============================================================
def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def vanleer_slope(dL, dR, eps=1e-30):
    # vectorized van Leer slope limiter (componentwise)
    return (np.sign(dL) + np.sign(dR)) * (np.abs(dL)*np.abs(dR)) / (np.abs(dL) + np.abs(dR) + eps)

# ============================================================
# One-pressure isothermal Two-Fluid Model (Evje & Flåtten 2005)
# U = [m_g, m_l, I_g, I_l]
# EOS:
#   rho_g = p / a_g^2
#   rho_l = rho_l0 + (p - p0)/a_l^2
# Closure:
#   m_g/rho_g(p) + m_l/rho_l(p) = 1
#
# Conservative flux part:
#   F = [I_g, I_l, I_g^2/m_g + alpha_g p, I_l^2/m_l + alpha_l p]
#
# Nonconservative term in momentum:
#   -p_i d(alpha_k)/dx, p_i = p - Δp
# with Δp = σ * ag*al*rg*rl/(rg*al + rl*ag) * (vg-vl)^2
#
# Numerical options kept in the same "scheme names" style:
#   upwind1, MUSCL, Central, Central_EVM, MUSCL_EVM,
#   QUICK, NT, ENO2, WENO3, CWENO3, KU, SDCU61, PCU
# (Some schemes share the same Rusanov/KU backend but differ in reconstruction/diffusion.)
# ============================================================
class TFMShockTubeSolver:
    def __init__(self, N=200, L=100.0, t_final=0.1, sigma=1.2,
                 scheme="PCU", dt_mode="fixed_dxdt", dxdt=1000.0, CFL=0.25):
        self.N = int(N)
        self.L = float(L)
        self.dx = self.L / self.N
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.N)

        self.t_final = float(t_final)
        self.sigma = float(sigma)

        self.scheme = str(scheme).strip()
        self.dt_mode = str(dt_mode).strip().lower()
        self.dxdt = float(dxdt)
        self.CFL = float(CFL)

        # EOS params (Evje & Flåtten 2005)
        self.p0 = 1.0e5
        self.rho_l0 = 1000.0
        self.a_l = 1.0e3
        self.a_g = math.sqrt(1.0e5)

        # Conservative state U = [m_g, m_l, I_g, I_l]
        self.U = np.zeros((self.N, 4), dtype=float)

        # floors
        self.m_floor = 1e-12
        self.alpha_floor = 1e-12

        # For EVM
        self.alpha_hist = []  # store alpha_l snapshots

    # ---------- EOS & closures ----------
    def rho_g(self, p):
        return p / (self.a_g**2)

    def rho_l(self, p):
        return self.rho_l0 + (p - self.p0) / (self.a_l**2)

    def pressure_from_masses(self, mg, ml):
        # Solve mg/rho_g(p) + ml/rho_l(p) = 1 with linear EOS
        A = self.a_l**2
        B = self.a_g**2
        c0 = self.rho_l0 - self.p0 / A
        mg = max(mg, self.m_floor)
        ml = max(ml, self.m_floor)
        b = A*c0 - mg*B - ml*A
        c = -mg * B * A * c0
        disc = b*b - 4.0*c
        disc = max(disc, 0.0)
        p = 0.5 * (-b + math.sqrt(disc))
        return max(p, 1.0)

    def prim_from_U(self, U):
        mg, ml, Ig, Il = U
        mg = max(mg, self.m_floor)
        ml = max(ml, self.m_floor)
        p = self.pressure_from_masses(mg, ml)
        rg = self.rho_g(p)
        rl = self.rho_l(p)
        ag = mg / rg
        al = ml / rl
        vg = Ig / mg
        vl = Il / ml
        return p, ag, al, vg, vl, rg, rl

    def U_from_prim(self, p, alpha_l, vg, vl):
        alpha_l = float(alpha_l)
        alpha_l = min(max(alpha_l, self.alpha_floor), 1.0 - self.alpha_floor)
        alpha_g = 1.0 - alpha_l
        rg = self.rho_g(p)
        rl = self.rho_l(p)
        mg = rg * alpha_g
        ml = rl * alpha_l
        Ig = mg * vg
        Il = ml * vl
        return np.array([mg, ml, Ig, Il], dtype=float)

    def delta_p(self, p, ag, al, vg, vl, rg, rl):
        denom = rg*al + rl*ag
        if denom <= 0:
            return 0.0
        return self.sigma * (ag*al*rg*rl/denom) * (vg - vl)**2

    # ---------- Derived fields ----------
    def alpha_l_field(self):
        al = np.zeros(self.N)
        for j in range(self.N):
            p, ag, alj, vg, vl, rg, rl = self.prim_from_U(self.U[j])
            al[j] = alj
        return al

    def mixture_velocity(self):
        # u_mix = (I_g + I_l) / (m_g + m_l)
        mg = np.maximum(self.U[:,0], self.m_floor)
        ml = np.maximum(self.U[:,1], self.m_floor)
        return (self.U[:,2] + self.U[:,3]) / (mg + ml)

    def max_speed_cell(self, Uj):
        p, ag, al, vg, vl, rg, rl = self.prim_from_U(Uj)
        return max(abs(vg) + self.a_g, abs(vl) + self.a_l)

    # ---------- BC ----------
    def extend_extrapolate(self, A):
        # A: (N, m) -> (N+2, m)
        AE = np.empty((self.N+2, A.shape[1]), dtype=float)
        AE[1:-1] = A
        AE[0] = A[0]
        AE[-1] = A[-1]
        return AE

    # ---------- Flux & PCU jump ----------
    def flux_conservative(self, U):
        mg, ml, Ig, Il = U
        mg = max(mg, self.m_floor)
        ml = max(ml, self.m_floor)
        p, ag, al, vg, vl, rg, rl = self.prim_from_U(U)
        return np.array([
            Ig,
            Il,
            Ig*vg + ag*p,
            Il*vl + al*p
        ], dtype=float)

    def p_i_mid(self, UL, UR):
        pL, agL, alL, vgL, vlL, rgL, rlL = self.prim_from_U(UL)
        pR, agR, alR, vgR, vlR, rgR, rlR = self.prim_from_U(UR)
        pm = 0.5*(pL + pR)
        agm = 0.5*(agL + agR)
        alm = 0.5*(alL + alR)
        vgm = 0.5*(vgL + vgR)
        vlm = 0.5*(vlL + vlR)
        rgm = self.rho_g(pm)
        rlm = self.rho_l(pm)
        dpm = self.delta_p(pm, agm, alm, vgm, vlm, rgm, rlm)
        return pm - dpm

    def pcu_jump(self, UL, UR):
        # momentum jumps from -p_i * d(alpha_k)
        p_i = self.p_i_mid(UL, UR)
        _, agL, alL, *_ = self.prim_from_U(UL)
        _, agR, alR, *_ = self.prim_from_U(UR)
        D = np.zeros(4, dtype=float)
        D[2] = -p_i * (agR - agL)
        D[3] = -p_i * (alR - alL)
        return D

    # ---------- Reconstruction methods (component-wise, direction by u_mix face) ----------
    def reconstruct_piecewise_constant(self, UE):
        # UE: (N+2,4) with ghosts -> UL/UR at faces (N+1,4)
        UL = UE[:-1].copy()
        UR = UE[1:].copy()
        return UL, UR

    def reconstruct_MUSCL(self, UE):
        dL = UE[1:-1] - UE[0:-2]
        dR = UE[2:  ] - UE[1:-1]
        slope = vanleer_slope(dL, dR)

        UL = np.empty((self.N+1, 4), dtype=float)
        UR = np.empty((self.N+1, 4), dtype=float)

        # left boundary
        UL[0] = UE[0]
        UR[0] = UE[1] - 0.5*slope[0]

        for i in range(1, self.N):
            UL[i] = UE[i] + 0.5*slope[i-1]
            UR[i] = UE[i+1] - 0.5*slope[i]
        # right boundary face
        UL[self.N] = UE[self.N] + 0.5*slope[self.N-1]
        UR[self.N] = UE[self.N+1]

        # positivity fallback on masses
        for i in range(self.N+1):
            if UL[i,0] < self.m_floor or UR[i,0] < self.m_floor or UL[i,1] < self.m_floor or UR[i,1] < self.m_floor:
                UL[i] = UE[i]
                UR[i] = UE[i+1]
        return UL, UR

    def _reconstruct_positive_QUICK_left(self, Q):
        # Q: (N+2,) with ghosts, returns left state at faces i=0..N (N+1)
        Qf = np.zeros(self.N+1)
        Qf[0] = Q[0]
        Qf[-1] = Q[-2]
        for i in range(1, self.N):
            # face i between cells i and i+1 in UE indexing
            # match your original QUICK: face i uses cells (i),(i-1),(i-2) in interior indexing
            if i < 3:
                Qf[i] = Q[i]  # fallback
            else:
                Qf[i] = (3*Q[i] + 6*Q[i-1] - Q[i-2]) / 8.0
        return Qf

    def _reconstruct_positive_ENO2_left(self, Q):
        Qf = np.zeros(self.N+1)
        Qf[0] = Q[0]
        Qf[-1] = Q[-2]
        for i in range(1, self.N):
            if i < 2 or i > self.N-2:
                Qf[i] = Q[i]
                continue
            d_left = Q[i] - Q[i-1]
            d_right = Q[i+1] - Q[i]
            d_more = Q[i+2] - Q[i+1]
            smooth_left = abs(d_right - d_left)
            smooth_right = abs(d_more - d_right)
            if smooth_left <= smooth_right:
                slope = d_right
                Qf[i] = Q[i] + 0.5*slope
            else:
                slope = d_more
                Qf[i] = Q[i+1] - 0.5*slope
        return Qf

    def _reconstruct_positive_WENO3_left(self, Q):
        eps = 1e-6
        Qf = np.zeros(self.N+1)
        Qf[0] = Q[0]
        Qf[-1] = Q[-2]
        for i in range(1, self.N):
            if i < 2 or i > self.N-2:
                Qf[i] = Q[i]
                continue
            v0 = 0.5 * (Q[i-1] + Q[i])       # S0
            v1 = 0.5 * (-Q[i] + 3*Q[i+1])    # S1
            beta0 = (Q[i] - Q[i-1])**2
            beta1 = (Q[i+1] - Q[i])**2
            alpha0 = 1.0 / (eps + beta0)**2
            alpha1 = 2.0 / (eps + beta1)**2
            w0 = alpha0 / (alpha0 + alpha1)
            w1 = alpha1 / (alpha0 + alpha1)
            Qf[i] = w0*v0 + w1*v1
        return Qf

    def reconstruct_face_states(self, UE, method):
        # UE: (N+2,4). Returns UL/UR at faces (N+1,4).
        method = method.upper()
        if method in ["UPWIND1", "CENTRAL", "KU", "PCU", "SDCU61", "NT"]:
            return self.reconstruct_piecewise_constant(UE)
        if method in ["MUSCL", "MUSCL_EVM", "CENTRAL_EVM"]:
            return self.reconstruct_MUSCL(UE)

        # For QUICK/ENO/WENO/CWENO: do componentwise directional recon using u_mix
        u = self.mixture_velocity()
        # face velocity (N+1): average of neighbor cell u, extrapolate at boundaries
        uf = np.zeros(self.N+1)
        uf[0] = u[0]
        uf[-1] = u[-1]
        uf[1:-1] = 0.5*(u[:-1] + u[1:])

        UL = np.zeros((self.N+1,4))
        UR = np.zeros((self.N+1,4))

        for k in range(4):
            Q = UE[:,k]
            # positive-direction left states
            if method == "QUICK":
                QL_pos = self._reconstruct_positive_QUICK_left(Q)
            elif method == "ENO2":
                QL_pos = self._reconstruct_positive_ENO2_left(Q)
            elif method == "WENO3":
                QL_pos = self._reconstruct_positive_WENO3_left(Q)
            elif method == "CWENO3":
                QL_pos = self._reconstruct_positive_WENO3_left(Q)
            else:
                QL_pos = self._reconstruct_positive_WENO3_left(Q)

            # negative-direction: mirror
            Qrev = Q[::-1].copy()
            if method == "QUICK":
                QR_pos_rev = self._reconstruct_positive_QUICK_left(Qrev)
            elif method == "ENO2":
                QR_pos_rev = self._reconstruct_positive_ENO2_left(Qrev)
            else:
                QR_pos_rev = self._reconstruct_positive_WENO3_left(Qrev)
            # map back to right states
            QR_neg = QR_pos_rev[::-1]

            # build UL/UR based on face velocity direction
            for i in range(self.N+1):
                if uf[i] >= 0:
                    UL[i,k] = QL_pos[i]
                    UR[i,k] = UE[i+1,k]  # fallback right state
                else:
                    UL[i,k] = UE[i,k]    # fallback left state
                    UR[i,k] = QR_neg[i]

        # CWENO3: average of left/right biased WENO3 (very approximate here)
        if method == "CWENO3":
            # just smooth the states a bit by averaging
            UL = 0.5*(UL + UE[:-1])
            UR = 0.5*(UR + UE[1:])

        # positivity fallback on masses
        for i in range(self.N+1):
            if UL[i,0] < self.m_floor or UR[i,0] < self.m_floor or UL[i,1] < self.m_floor or UR[i,1] < self.m_floor:
                UL[i] = UE[i]
                UR[i] = UE[i+1]
        return UL, UR

    # ---------- EVM (entropy-viscosity-like sensor using alpha_l) ----------
    def compute_entropy_viscosity(self, alpha, dt, C_ev=0.05, C_max=0.25):
        # same spirit as in your faucet driver; use alpha_l as "entropy"
        a = np.array(alpha, dtype=float)
        a_old = self.alpha_hist[-1] if len(self.alpha_hist) > 0 else a.copy()

        da_dt = (a - a_old) / max(dt, 1e-14)
        da_dx = np.zeros_like(a)
        da_dx[1:-1] = (a[2:] - a[:-2]) / (2*self.dx)
        da_dx[0] = (a[1] - a[0]) / self.dx
        da_dx[-1] = (a[-1] - a[-2]) / self.dx

        # residual r = a_t + u_mix a_x
        u = self.mixture_velocity()
        r = da_dt + u*da_dx

        # local scale
        r_scale = np.max(np.abs(r)) + 1e-14
        nu = C_ev * self.dx*self.dx * (np.abs(r)/r_scale)

        # cap
        umax = np.max(np.abs(u)) + max(self.a_g, self.a_l)
        nu_max = C_max * umax * self.dx
        nu = np.minimum(nu, nu_max)
        return nu

    # ---------- One step ----------
    def step(self, dt):
        UE = self.extend_extrapolate(self.U)

        scheme = self.scheme.strip()
        schemeU = scheme.upper()

        # base reconstruction selector
        recon_method = schemeU
        if schemeU in ["KU", "PCU", "SDCU61"]:
            recon_method = "MUSCL"  # default: 2nd-order MUSCL for these
        if schemeU == "UPWIND1":
            recon_method = "UPWIND1"

        UL, UR = self.reconstruct_face_states(UE, recon_method)

        # entropy-viscosity coefficient (cell-based), used as extra diffusion
        nu = None
        if schemeU in ["CENTRAL_EVM", "MUSCL_EVM"]:
            alpha = self.alpha_l_field()
            nu = self.compute_entropy_viscosity(alpha, dt)

        # fluxes
        F = np.zeros((self.N+1, 4), dtype=float)

        for i in range(self.N+1):
            FL = self.flux_conservative(UL[i])
            FR = self.flux_conservative(UR[i])

            if schemeU in ["CENTRAL", "CENTRAL_EVM"]:
                a_face = self.dxdt  # fixed diffusion speed, like LF
            else:
                # KU/Rusanov diffusion
                a_face = max(self.max_speed_cell(UL[i]), self.max_speed_cell(UR[i]))

            # Local Lax-Friedrichs / Rusanov
            Fhat = 0.5*(FL + FR) - 0.5*a_face*(UR[i] - UL[i])

            # Add EVM diffusion (treat as additional speed)
            if nu is not None and i > 0 and i < self.N:
                nu_face = 0.5*(nu[i-1] + nu[i])
                Fhat -= nu_face * (UR[i] - UL[i]) / self.dx

            # PCU jump
            if schemeU in ["PCU"]:
                Fhat += self.pcu_jump(UL[i], UR[i])

            # SDCU61: keep same backend here (you can upgrade later)
            # NT/QUICK/ENO/WENO/CWENO mainly differ by recon_method
            F[i] = Fhat

        Unew = self.U.copy()
        for j in range(self.N):
            Unew[j] -= (dt/self.dx)*(F[j+1] - F[j])

        # floors + keep velocities consistent to avoid blow-ups when masses are floored
        for j in range(self.N):
            mg, ml, Ig, Il = Unew[j]
            if mg < self.m_floor:
                vg = Ig / max(mg, self.m_floor)
                mg = self.m_floor
                Ig = mg * vg
            if ml < self.m_floor:
                vl = Il / max(ml, self.m_floor)
                ml = self.m_floor
                Il = ml * vl
            Unew[j] = [mg, ml, Ig, Il]

        self.U = Unew

        # store alpha for EVM
        self.alpha_hist.append(self.alpha_l_field().copy())
        if len(self.alpha_hist) > 2:
            self.alpha_hist.pop(0)

    # ---------- Driver ----------
    def compute_dt(self):
        # fixed dt from dx/dt
        dt_fixed = self.dx / max(self.dxdt, 1e-14)
        # CFL dt
        smax = 0.0
        for j in range(self.N):
            smax = max(smax, self.max_speed_cell(self.U[j]))
        dt_cfl = self.CFL * self.dx / max(smax, 1e-14)

        if self.dt_mode == "cfl":
            return dt_cfl
        if self.dt_mode == "fixed_dxdt":
            # safety: don't exceed CFL
            return min(dt_fixed, dt_cfl)
        return dt_cfl

    def run(self):
        t = 0.0
        # init alpha history for EVM
        self.alpha_hist = [self.alpha_l_field().copy()]
        while t < self.t_final - 1e-14:
            dt = self.compute_dt()
            if t + dt > self.t_final:
                dt = self.t_final - t
            self.step(dt)
            t += dt
        return self.get_primitives()

    def get_primitives(self):
        p  = np.zeros(self.N)
        al = np.zeros(self.N)
        vg = np.zeros(self.N)
        vl = np.zeros(self.N)
        for j in range(self.N):
            pj, agj, alj, vgj, vlj, *_ = self.prim_from_U(self.U[j])
            p[j] = pj
            al[j] = alj
            vg[j] = vgj
            vl[j] = vlj
        return p, al, vg, vl

    def plot(self, title=""):
        p, al, vg, vl = self.get_primitives()
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

        axs[0,0].plot(self.x, al)
        axs[0,0].set_title("Liquid fraction alpha_l")
        axs[0,0].set_xlabel("x"); axs[0,0].set_ylabel("alpha_l")

        axs[0,1].plot(self.x, p)
        axs[0,1].set_title("Pressure p (Pa)")
        axs[0,1].set_xlabel("x"); axs[0,1].set_ylabel("p")

        axs[1,0].plot(self.x, vl)
        axs[1,0].set_title("Liquid velocity v_l (m/s)")
        axs[1,0].set_xlabel("x"); axs[1,0].set_ylabel("v_l")

        axs[1,1].plot(self.x, vg)
        axs[1,1].set_title("Gas velocity v_g (m/s)")
        axs[1,1].set_xlabel("x"); axs[1,1].set_ylabel("v_g")

        if title:
            fig.suptitle(title)
        return fig



# ============================================================
# Combined driver: Faucet demo OR Evje & Flåtten (2005) shock tubes
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run either the original faucet demo or Evje & Flåtten (2005) shock-tube benchmarks."
    )
    parser.add_argument("--problem", choices=["faucet", "LRV", "MLRV", "TOUMI"], default="LRV",
                        help="Select which problem to run.")
    parser.add_argument("--scheme", default="PCU",
                        help=("Scheme/flux option. Faucet solver uses e.g. Upwind1, Central, MUSCL, QUICK, "
                              "ENO2, WENO3, CWENO3, KU, PCU, Central_EVM, MUSCL_EVM. "
                              "Shock-tube solver uses e.g. upwind1, Central, MUSCL, QUICK, NT, ENO2, "
                              "WENO3, CWENO3, KU, SDCU61, PCU."))
    # Common plotting toggle
    parser.add_argument("--no_plot", action="store_true")

    # Faucet options
    parser.add_argument("--N_faucet", type=int, default=100)
    parser.add_argument("--L_faucet", type=float, default=12.0)
    parser.add_argument("--t_final_faucet", type=float, default=0.5)

    # Shock-tube options
    parser.add_argument("--N", type=int, default=None, help="Override number of shock-tube cells.")
    parser.add_argument("--L", type=float, default=None, help="Override shock-tube domain length.")
    parser.add_argument("--t_final", type=float, default=None, help="Override shock-tube final time.")
    parser.add_argument("--dt_mode", choices=["fixed_dxdt", "cfl"], default="fixed_dxdt")
    parser.add_argument("--dxdt", type=float, default=None, help="Override Δx/Δt when dt_mode=fixed_dxdt.")
    parser.add_argument("--CFL", type=float, default=0.25)

    args = parser.parse_args()

    if args.problem == "faucet":
        solver = FullPressureTwoFluidSolver(n_cells=args.N_faucet, L=args.L_faucet, t_final=args.t_final_faucet)
        print(f"[Faucet] scheme={args.scheme}")
        alpha, u_l, visc = solver.solve(scheme=args.scheme)

        if not args.no_plot:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
            ax.plot(solver.x, alpha)
            ax.set_title(f"Faucet: alpha(x) | scheme={args.scheme}")
            ax.set_xlabel("x"); ax.set_ylabel("alpha")
            ax.grid(True, alpha=0.3)
            plt.show()

    else:
        # --- Evje & Flåtten (2005) shock-tube benchmarks ---
        CASES = {
            "LRV": {
                "title": "LRV shock (Evje & Flåtten 2005 §7.1)",
                "WL": {"p": 265000.0, "alpha_l": 0.71, "vg": 65.0, "vl": 1.0},
                "WR": {"p": 265000.0, "alpha_l": 0.70, "vg": 50.0, "vl": 1.0},
                "N": 100, "L": 100.0, "t_final": 0.1, "dxdt": 1000.0, "sigma": 1.2,
            },
            "MLRV": {
                "title": "Modified LRV shock (Evje & Flåtten 2005 §7.2)",
                "WL": {"p": 265000.0, "alpha_l": 0.70, "vg": 65.0, "vl": 10.0},
                "WR": {"p": 265000.0, "alpha_l": 0.10, "vg": 50.0, "vl": 15.0},
                "N": 100, "L": 100.0, "t_final": 0.1, "dxdt": 750.0, "sigma": 1.2,
            },
            "TOUMI": {
                "title": "Toumi water-air shock (Evje & Flåtten 2005 §7.3)",
                "WL": {"p": 2.0e7, "alpha_l": 0.75, "vg": 0.0, "vl": 0.0},
                "WR": {"p": 1.0e7, "alpha_l": 0.90, "vg": 0.0, "vl": 0.0},
                "N": 200, "L": 100.0, "t_final": 0.08, "dxdt": 1000.0, "sigma": 2.0,
            },
        }
        cfg = CASES[args.problem]

        N = args.N if args.N is not None else cfg["N"]
        L = args.L if args.L is not None else cfg["L"]
        t_final = args.t_final if args.t_final is not None else cfg["t_final"]
        dxdt = args.dxdt if args.dxdt is not None else cfg["dxdt"]
        sigma = cfg["sigma"]

        solver = TFMShockTubeSolver(
            N=N, L=L, t_final=t_final,
            sigma=sigma,
            scheme=args.scheme,
            dt_mode=args.dt_mode,
            dxdt=dxdt,
            CFL=args.CFL,
        )

        WL = cfg["WL"]; WR = cfg["WR"]
        x0 = 0.5 * L
        for j, x in enumerate(solver.x):
            W = WL if x < x0 else WR
            solver.U[j] = solver.U_from_prim(W["p"], W["alpha_l"], W["vg"], W["vl"])

        print(f"[ShockTube {args.problem}] scheme={args.scheme} | dt_mode={args.dt_mode} | "
              f"{('dx/dt='+str(dxdt)) if args.dt_mode=='fixed_dxdt' else ('CFL='+str(args.CFL))} | sigma={sigma}")

        solver.run()

        if not args.no_plot:
            title = f'{cfg["title"]} | scheme={args.scheme} | dt_mode={args.dt_mode}'
            if args.dt_mode == "fixed_dxdt":
                title += f" | dx/dt={dxdt:g}"
            fig = solver.plot(title=title)
            plt.show()
