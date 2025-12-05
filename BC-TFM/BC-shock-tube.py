import numpy as np
import matplotlib.pyplot as plt

class ShockTubeSolver:
    def __init__(self, N=50, L=100.0, T_end=0.1, dt=1e-4):
        # --- 1. Geometry & Time ---
        self.N = N
        self.L = L
        self.dx = L / N
        self.x = np.linspace(self.dx/2, L-self.dx/2, N)
        self.dt = dt
        self.T_end = T_end

        # --- 2. Phase Properties ---
        # Liquid (Water-like)
        self.rho_l = 1000.0
        # Gas (Air-like) - Compressible
        # Assuming P = rho * c^2 relation for simplicity in this demo
        self.P_init = 2.65e5
        self.rho_g_init = 3.0 # Approx density at 2.65 bar
        self.c_g_sq = self.P_init / self.rho_g_init * 1.4 # Speed of sound squared

        # --- 3. Paper Parameters ---
        self.lm = 0.05       # Mixing length
        self.nu_k = 1e-5     # Kinematic viscosity

        # --- 4. Initialization (Left/Right State) ---
        self.alpha = np.zeros(N)
        self.u_g = np.zeros(N)
        self.u_l = np.zeros(N)
        self.P = np.zeros(N)

        mid = N // 2

        # Left State
        self.alpha[:mid] = 0.29
        self.u_g[:mid]   = 65.0
        self.u_l[:mid]   = 1.0
        self.P[:mid]     = 2.65e5

        # Right State
        self.alpha[mid:] = 0.30
        self.u_g[mid:]   = 50.0
        self.u_l[mid:]   = 1.0
        self.P[mid:]     = 2.65e5

        # Conserved Variables
        self.rho_g = np.ones(N) * self.rho_g_init

        # U = [m_g, m_l, mom_g, mom_l]
        self.m_g = self.alpha * self.rho_g
        self.m_l = (1 - self.alpha) * self.rho_l
        self.mom_g = self.m_g * self.u_g
        self.mom_l = self.m_l * self.u_l

    def get_viscous_forces(self):
        # Turbulent Viscosity: nu_t = lm * |u_r| [cite: 13, 14, 15]
        u_r = self.u_g - self.u_l
        nu_t = self.lm * np.abs(u_r)
        nu_eff = self.nu_k + nu_t

        # 1. Gas Viscosity Gradient
        dug_dx = np.gradient(self.u_g, self.dx)
        stress_g = self.alpha * nu_eff * dug_dx
        F_visc_g = np.gradient(stress_g, self.dx)

        # 2. Liquid Viscosity Gradient
        dul_dx = np.gradient(self.u_l, self.dx)
        stress_l = (1 - self.alpha) * nu_eff * dul_dx
        F_visc_l = np.gradient(stress_l, self.dx)

        return F_visc_g, F_visc_l

    def solve(self):
        t = 0.0
        times = [0.0]
        # History storage
        h_alpha = [self.alpha.copy()]
        h_P     = [self.P.copy()]
        h_ug    = [self.u_g.copy()]
        h_ul    = [self.u_l.copy()]

        while t < self.T_end:
            # --- 1. Primitives Update ---
            self.rho_g = self.m_g / (self.alpha + 1e-9)
            # Pressure EOS
            self.P = self.P_init + self.c_g_sq * (self.rho_g - self.rho_g_init)

            # Reconstruct Velocities
            self.u_g = self.mom_g / (self.m_g + 1e-9)
            self.u_l = self.mom_l / (self.m_l + 1e-9)

            # --- 2. Flux Calculation (Rusanov) ---
            # State U
            U = np.stack((self.m_g, self.m_l, self.mom_g, self.mom_l))

            # Flux F
            F_mg = self.m_g * self.u_g
            F_ml = self.m_l * self.u_l
            F_momg = self.mom_g * self.u_g + self.alpha * self.P
            F_moml = self.mom_l * self.u_l + (1-self.alpha) * self.P
            F = np.stack((F_mg, F_ml, F_momg, F_moml))

            # Wave Speeds
            lambda_g = np.abs(self.u_g) + np.sqrt(self.c_g_sq)
            lambda_l = np.abs(self.u_l) + 10.0 # Pseudo-speed for liquid
            max_wave = np.maximum(lambda_g, lambda_l)

            # Flux at Faces
            F_L = F[:, :-1]; F_R = F[:, 1:]
            U_L = U[:, :-1]; U_R = U[:, 1:]
            wave_speed = np.maximum(max_wave[:-1], max_wave[1:])

            Flux = 0.5 * (F_L + F_R) - 0.5 * wave_speed * (U_R - U_L)

            # Divergence dF/dx
            dFdx = np.zeros_like(U)
            dFdx[:, 1:-1] = (Flux[:, 1:] - Flux[:, :-1]) / self.dx

            # Wall BCs (Flux = 0 for mass, P for momentum)
            F_wall_L = np.array([0, 0, self.alpha[0]*self.P[0], (1-self.alpha[0])*self.P[0]])
            F_wall_R = np.array([0, 0, self.alpha[-1]*self.P[-1], (1-self.alpha[-1])*self.P[-1]])

            dFdx[:, 0] = (Flux[:, 0] - F_wall_L) / self.dx
            dFdx[:, -1] = (F_wall_R - Flux[:, -1]) / self.dx

            # --- 3. Source Terms (Viscosity) ---
            F_visc_g, F_visc_l = self.get_viscous_forces()

            Source = np.zeros_like(U)
            Source[2, :] += F_visc_g
            Source[3, :] += F_visc_l

            # --- 4. Update ---
            U_new = U - self.dt * dFdx + self.dt * Source

            # Unpack
            self.m_g = U_new[0]
            self.m_l = U_new[1]
            self.mom_g = U_new[2]
            self.mom_l = U_new[3]

            # Recalculate Alpha from Mass
            # vol_g = m_g / rho_g(approx); vol_l = m_l / rho_l
            # This is iterative in compressible flow, simplified here:
            vol_l = self.m_l / self.rho_l
            self.alpha = 1.0 - vol_l

            t += self.dt
            if len(times) < 10 or t >= times[-1] + 0.01:
                times.append(t)
                h_alpha.append(self.alpha.copy())
                h_P.append(self.P.copy())
                h_ug.append(self.u_g.copy())
                h_ul.append(self.u_l.copy())

        return self.x, h_alpha, h_P, h_ug, h_ul, times

# --- Run & Plot ---
solver = ShockTubeSolver(N=50, L=100.0, T_end=0.1, dt=1e-4)
x, h_a, h_P, h_ug, h_ul, t_vec = solver.solve()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Void Fraction
axes[0, 0].plot(x, h_a[0], 'k--', label='Initial')
axes[0, 0].plot(x, h_a[-1], 'r-', lw=2, label='Final (0.1s)')
axes[0, 0].set_title(r"Gas Void Fraction $\alpha$")
axes[0, 0].set_ylabel("Alpha [-]")
axes[0, 0].grid(True)
axes[0, 0].legend()

# 2. Pressure
axes[0, 1].plot(x, h_P[0]/1e5, 'k--', label='Initial')
axes[0, 1].plot(x, h_P[-1]/1e5, 'b-', lw=2, label='Final (0.1s)')
axes[0, 1].set_title("Pressure")
axes[0, 1].set_ylabel("Pressure [bar]")
axes[0, 1].grid(True)

# 3. Gas Velocity
axes[1, 0].plot(x, h_ug[0], 'k--', label='Initial')
axes[1, 0].plot(x, h_ug[-1], 'g-', lw=2, label='Final (0.1s)')
axes[1, 0].set_title("Gas Velocity $u_g$")
axes[1, 0].set_xlabel("Position [m]")
axes[1, 0].set_ylabel("Velocity [m/s]")
axes[1, 0].grid(True)

# 4. Liquid Velocity
axes[1, 1].plot(x, h_ul[0], 'k--', label='Initial')
axes[1, 1].plot(x, h_ul[-1], 'm-', lw=2, label='Final (0.1s)')
axes[1, 1].set_title("Liquid Velocity $u_l$")
axes[1, 1].set_xlabel("Position [m]")
axes[1, 1].set_ylabel("Velocity [m/s]")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
