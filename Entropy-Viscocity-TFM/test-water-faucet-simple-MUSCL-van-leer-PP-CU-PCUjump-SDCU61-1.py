import numpy as np
import math
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

    # ---------------------------------------------------------------------
    # Nonconservative products (DLM / path-conservative interface terms)
    # ---------------------------------------------------------------------
    def nonconservative_matrix(self, U):
        """
        Return the nonconservative coefficient matrix B(U) for a term of the form
            U_t + F(U)_x + B(U) U_x = S(U).
        In the current faucet model we do NOT include any nonconservative products,
        so the default is zero.

        If you later add a nonconservative term (e.g., in a 1-pressure TFM with
        interfacial-pressure / relaxation terms written as B(U)U_x), override this
        function accordingly.

        Parameters
        ----------
        U : array_like, shape (2,)
            State vector [mass, mom].

        Returns
        -------
        B : ndarray, shape (2,2)
        """
        return np.zeros((2, 2), dtype=float)

    def _path_integral_BdU(self, U_L, U_R, n_quad=2):
        """
        Compute the DLM path integral
            D(U_L,U_R) = ∫_0^1 B(φ(s)) φ_s(s) ds
        along a straight-line path φ(s) = U_L + s (U_R-U_L).

        Notes
        -----
        - For n_quad=2, we use 2-pt Gauss–Legendre on [0,1].
        - If B(U)=0 (default), D=0 and the method reduces to the conservative CU/HLL form.
        """
        dU = U_R - U_L
        if n_quad == 1:
            s_nodes = [0.5]
            w_nodes = [1.0]
        elif n_quad == 2:
            a = 1.0 / math.sqrt(3.0)
            s_nodes = [0.5 * (1.0 - a), 0.5 * (1.0 + a)]
            w_nodes = [0.5, 0.5]
        else:
            s_nodes = [(k + 0.5) / n_quad for k in range(n_quad)]
            w_nodes = [1.0 / n_quad] * n_quad

        D = np.zeros_like(dU, dtype=float)
        for s, w in zip(s_nodes, w_nodes):
            U = U_L + s * dU
            B = self.nonconservative_matrix(U)
            D += w * (B @ dU)
        return D

    def _pcu_fluctuations(self, U_L, U_R, F_L, F_R, a_minus, a_plus):
        """
        Compute path-conservative HLL-type fluctuations (M^- , M^+) satisfying:
            M^- + M^+ = (F_R - F_L) + D(U_L,U_R),
        where D is the DLM path integral for nonconservative products.

        Returns
        -------
        M_minus, M_plus : ndarrays shape (2,)
            Left-going and right-going fluctuations at the interface.
        """
        dU = U_R - U_L
        D = self._path_integral_BdU(U_L, U_R)
        C = (F_R - F_L) + D  # total jump consistent with the chosen path

        if a_minus >= 0.0:
            return np.zeros_like(C), C
        if a_plus <= 0.0:
            return C, np.zeros_like(C)

        denom = (a_plus - a_minus)
        M_minus = (a_plus / denom) * (C - a_minus * dU)
        M_plus  = (-a_minus / denom) * (C - a_plus  * dU)
        return M_minus, M_plus

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

    def minmod(self, a, b):
        """Componentwise minmod limiter."""
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    def _cu_semidiscrete_flux(self, U_minus, U_plus, F_minus, F_plus, a_minus, a_plus):
        """
        Semidiscrete central-upwind numerical flux with built-in anti-diffusion:
            H = (a^+ f(U^-) - a^- f(U^+))/(a^+ - a^-) + (a^+ a^-)/(a^+ - a^-)(U^+ - U^-) - d/2
        where
            d = minmod( (U^+ - U*)/(a^+ - a^-), (U* - U^-)/(a^+ - a^-) )
            U* = (a^+ U^+ - a^- U^- - (f(U^+) - f(U^-)))/(a^+ - a^-)
        (See Handbook of Numerical Analysis, Chapter 20, Eqs. (46)-(49).)
        """
        denom = a_plus - a_minus
        if denom < 1e-14:
            # Degenerate case (no upwind direction); use centered flux
            return 0.5 * (F_minus + F_plus)

        U_star = (a_plus * U_plus - a_minus * U_minus - (F_plus - F_minus)) / denom
        d = self.minmod((U_plus - U_star) / denom, (U_star - U_minus) / denom)
        H = (a_plus * F_minus - a_minus * F_plus + a_plus * a_minus * (U_plus - U_minus)) / denom - 0.5 * d
        return H

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
        if any(tag in scheme for tag in ['ENO', 'WENO', 'CWENO', 'KU', 'PCU', 'SDCU']):
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

            # Artificial-viscosity dissipation kept as a separate conservative flux.
            flux_mass_diss = np.zeros(self.N + 1)
            flux_mom_diss  = np.zeros(self.N + 1)

            mass = (1 - self.alpha) * self.rho_l
            mom  = mass * self.u_l

            # Pre-calculate Reconstructions if needed
            if 'MUSCL' in scheme:
                rho_muscl = self.reconstruct_MUSCL(mass)
                u_muscl   = self.reconstruct_MUSCL(self.u_l)
            if 'QUICK' in scheme:
                rho_quick = self.reconstruct_QUICK(mass)
                u_quick   = self.reconstruct_QUICK(self.u_l)
            if ('NT' in scheme) or ('KU' in scheme) or ('PCU' in scheme) or ('SDCU' in scheme):
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
            if any(tag in scheme for tag in ['ENO', 'WENO', 'CWENO', 'KU', 'PCU', 'SDCU']):
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
                elif 'SDCU' in scheme:
                    # Semidiscrete central-upwind flux (Section 6.1) with built-in anti-diffusion.
                    # See Handbook of Numerical Analysis, Chapter 20, Eqs. (46)-(49).
                    mass_L = mass[i-1] + 0.5 * slope_mass[i-1]
                    mass_R = mass[i]   - 0.5 * slope_mass[i]
                    mom_L  = mom[i-1]  + 0.5 * slope_mom[i-1]
                    mom_R  = mom[i]    - 0.5 * slope_mom[i]

                    mass_L = max(mass_L, 1e-12)
                    mass_R = max(mass_R, 1e-12)

                    U_minus = np.array([mass_L, mom_L], dtype=float)
                    U_plus  = np.array([mass_R, mom_R], dtype=float)

                    u_minus = U_minus[1] / U_minus[0]
                    u_plus  = U_plus[1]  / U_plus[0]

                    F_minus = np.array([U_minus[1], U_minus[1] * u_minus], dtype=float)
                    F_plus  = np.array([U_plus[1],  U_plus[1]  * u_plus ], dtype=float)

                    a_plus  = max(u_minus, u_plus, 0.0)
                    a_minus = min(u_minus, u_plus, 0.0)

                    H = self._cu_semidiscrete_flux(U_minus, U_plus, F_minus, F_plus, a_minus, a_plus)
                    f_mass_conv = H[0]
                    f_mom_conv  = H[1]

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
                if ('NT' not in scheme) and ('KU' not in scheme) and ('PCU' not in scheme) and ('SDCU' not in scheme):
                    f_mass_conv = rho_face * u_face
                    f_mom_conv  = rho_face * u_face**2

                # --- ADD DIFFUSION / ARTIFICIAL VISCOSITY (conservative form) ---
                # Interpret nu_visc as a face diffusivity and write it as an additional
                # Rusanov/Lax-Friedrichs dissipation term using the (possibly reconstructed) jump.
                s_visc = 2.0 * nu_visc[i] / self.dx  # [m/s]

                # Conservative artificial-viscosity dissipation term.
                flux_mass_diss[i] = -0.5 * s_visc * (mass_R - mass_L)
                flux_mom_diss[i]  = -0.5 * s_visc * (mom_R  - mom_L )

                flux_mass[i] = f_mass_conv + flux_mass_diss[i]
                flux_mom[i]  = f_mom_conv  + flux_mom_diss[i]

            # Boundary Fluxes
            rho_in = (1.0 - self.alpha0) * self.rho_l

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

# --- RUNNING THE CASES ---
N_cells=50
solver = FullPressureTwoFluidSolver(n_cells=N_cells, t_final=0.5)

x_ref, alpha_ref = solver.analytical_solution(0.5)
results = {}

print("1. Running 1st Order Upwind...")
results['Up1'] = solver.solve(scheme='Upwind1')

print("2. Running MUSCL (Van Leer) [No EVM]...")
results['MUSCL'] = solver.solve(scheme='MUSCL')

print("3. Running Central [No EVM]...")
results['Cen'] = solver.solve(scheme='Central')

print("4. Running Central + EVM...")
results['Cen_EVM'] = solver.solve(scheme='Central_EVM')

print("5. Running MUSCL + EVM...")
results['MUSCL_EVM'] = solver.solve(scheme='MUSCL_EVM')

print("6. Running QUICK (3rd order upwind, no EVM)...")
results['QUICK'] = solver.solve(scheme='QUICK')

print("7. Running Nessyahu-Tadmor 2nd Order Central...")
results['NT'] = solver.solve(scheme='NT')

print("8. Running ENO-2 (no EVM)...")
results['ENO2'] = solver.solve(scheme='ENO')

print("9. Running WENO-3 (no EVM)...")
results['WENO3'] = solver.solve(scheme='WENO')

print("10. Running Central WENO-3 (CWENO)...")
results['CWENO3'] = solver.solve(scheme='CWENO')

print("11. Running Kurganov Central-Upwind...")
results['KU'] = solver.solve(scheme='KU')

print("11b. Running Semidiscrete Central-Upwind (Section 6.1, with anti-diffusion)...")
results['SDCU'] = solver.solve(scheme='SDCU')


print("12. Running Path-Conservative Central-Upwind...")
results['PCU'] = solver.solve(scheme='PCU')

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Void Fraction
ax1.set_title("Water Faucet: Flux Scheme Comparison")
ax1.plot(x_ref, alpha_ref, 'k--', linewidth=2, label='Analytical')

# Baseline / Unstable
ax1.plot(solver.x, results['Up1'][1], 'g--', label='1st Order Upwind')
#ax1.plot(solver.x, results['Cen'][1], 'm:',  label='Central (Unstable)')

# MUSCL (Standard)
#ax1.plot(solver.x, results['MUSCL'][1], 'y-', label='MUSCL (Van Leer)')
#ax1.plot(solver.x, results['QUICK'][1], 'k-', linewidth=1.5, label='QUICK')
#ax1.plot(solver.x, results['NT'][1], color='blue',linestyle='--',linewidth=2, label='Nessyahu-Tadmor 2nd Order')
ax1.plot(solver.x, results['KU'][1], color='tab:purple', linestyle='-', linewidth=2.0, label='Kurganov CU')
ax1.plot(solver.x, results['SDCU'][1], color='tab:orange', linestyle='-', linewidth=2.0, label='Semidiscrete CU (Sec 6.1)')
ax1.plot(solver.x, results['PCU'][1], color='tab:cyan', linestyle='-', linewidth=2.0, label='Path-Cons. CU')
#ax1.plot(solver.x, results['ENO2'][1], color='brown', linestyle='--', linewidth=2.0, label='ENO-2')
#ax1.plot(solver.x, results['WENO3'][1], color='red', linestyle='-.', linewidth=2.0, label='WENO-3')
#ax1.plot(solver.x, results['CWENO3'][1], color='orange', linestyle=':', linewidth=2.0, label='CWENO-3')

# EVM Stabilized
#ax1.plot(solver.x, results['Cen_EVM'][1], 'b-', linewidth=2, label='Central + EVM')
ax1.plot(solver.x, results['MUSCL_EVM'][1], 'r-', linewidth=2, label='MUSCL + EVM')

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Void Fraction")
#ax1.set_ylim(0.0, 0.35)
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Viscosity Profile
ax2.set_title("Artificial Viscosity Activation")
#ax2.plot(solver.x, results['Cen_EVM'][2], 'b-', label='Viscosity (Central Base)')
ax2.plot(solver.x, results['MUSCL_EVM'][2], 'r-', label='Viscosity (MUSCL Base)')
ax2.fill_between(solver.x, 0, results['MUSCL_EVM'][2], color='r', alpha=0.1)

ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Viscosity (m^2/s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'Water_Faucet_Flux_Scheme_Comparison_nCells_{N_cells}.png')
plt.show()
