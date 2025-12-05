import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Set general plot style
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['font.size'] = 10

# --- PART 1: LINEAR STABILITY (Figures 1 & 2) ---
def plot_linear_stability():
    # Parameters from Section IV
    alpha = 0.5
    W = 1.0
    sigma = 1e-4
    H = 1e-2
    Q = 0.25      # Simple inertia Q = alpha(1-alpha) at alpha=0.5
    Q_d2 = -2.0   # Q'' = -2

    k = np.logspace(0.5, 4.5, 500)

    # Eq 25: Pure Advection Growth Rate
    # Im(w) = sqrt(alpha(1-alpha)) * |W k|
    im_w_adv = np.sqrt(alpha*(1-alpha)) * np.abs(W * k)

    # Eq 41: With Surface Tension
    # Delta term determines stability
    radicand = Q * (0.5 * Q_d2 * W**2 * k**2 + sigma * H * k**4)
    im_w_st = np.zeros_like(k)
    mask = radicand < 0
    im_w_st[mask] = np.sqrt(-radicand[mask])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Figure 1: Pure Advection
    ax = axes[0]
    ax.loglog(k, im_w_adv, 'k-', label='Theoretical rate')
    ax.set_title('Fig 1: Pure Advection Growth Rate')
    ax.set_xlabel('k [1/m]'); ax.set_ylabel('Im(omega) [1/s]')
    ax.set_ylim(bottom=1e0)

    # Figure 2: Surface Tension
    ax = axes[1]
    ax.loglog(k, im_w_st, 'k-', label='Theoretical rate')
    k_cut = np.abs(W) * np.sqrt(np.abs(Q_d2)/(2*sigma*H))
    ax.axvline(k_cut, color='red', linestyle=':', label='Linear Cutoff')
    ax.set_title('Fig 2: Surface Tension Growth Rate')
    ax.set_xlabel('k [1/m]'); ax.set_ylim(bottom=1e0)
    ax.legend()
    return fig

# --- PART 2: ANALYTICAL WAVEFORMS (Figures 20 & 21) ---
def plot_waveforms():
    theta = np.linspace(0, 4*np.pi, 200)

    # Linear (Eq 125-126): 90 deg phase shift
    da_lin = np.cos(theta)
    dW_lin = -np.sin(theta)

    # Nonlinear (Heuristic based on Eq 127): Stokes-like steepening
    a2 = 0.3; w2 = 0.4 # Coeffs for visual representation
    da_nl = np.cos(theta) - a2 * np.sin(2*theta)
    dW_nl = -np.sin(theta) - w2 * np.cos(2*theta)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(theta, da_lin, 'r-', label=r'$\Delta \alpha$')
    axes[0].plot(theta, dW_lin, 'b-', label=r'$\Delta W$')
    axes[0].set_title('Fig 20: Linear Waveforms')
    axes[0].legend(loc='upper right')

    axes[1].plot(theta, da_nl, 'r-', label=r'$\Delta \alpha$')
    axes[1].plot(theta, dW_nl, 'b-', label=r'$\Delta W$')
    axes[1].set_title('Fig 21: Nonlinear Waveforms (Shock-Spike)')
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    return fig

# --- PART 3: TFM SOLVER (Figures 3, 4, 8, 19) ---
class TFMSolver:
    def __init__(self, L, N, sigma=0.0, viscosity_model=None, lm=0.0, gx=0.0, wall_drag=False):
        self.L = L; self.N = N; self.dx = L/N
        self.x = np.linspace(self.dx/2, L-self.dx/2, N)
        self.sigma = sigma; self.visc_model = viscosity_model; self.lm = lm
        self.gx = gx; self.wall_drag = wall_drag
        self.H = 1e-2
        self.rho1 = 1.1 if gx != 0 else 1.0
        self.rho2 = 0.9 if gx != 0 else 1.0
        self.d_rho = self.rho1 - self.rho2

        # State
        self.alpha = np.ones(N) * 0.5
        self.W = np.ones(N) * 1.0

    def step(self, dt):
        # Flux calc (Simplified Central + Dissipation for stability)
        Q = self.alpha * (1 - self.alpha)
        Qp = 1.0 - 2.0 * self.alpha

        F_a = Q * self.W
        F_W = 0.5 * Qp * self.W**2

        # Gradients
        dFa_dx = (np.roll(F_a, -1) - np.roll(F_a, 1)) / (2*self.dx)
        dFW_dx = (np.roll(F_W, -1) - np.roll(F_W, 1)) / (2*self.dx)

        # Artificial dissipation to stabilize the base solver (replaces complex Flux Limiter)
        d2a = (np.roll(self.alpha, -1) - 2*self.alpha + np.roll(self.alpha, 1)) / (self.dx**2)
        d2W = (np.roll(self.W, -1) - 2*self.W + np.roll(self.W, 1)) / (self.dx**2)
        dFa_dx -= 0.001 * self.dx * d2a
        dFW_dx -= 0.001 * self.dx * d2W

        Forces_W = np.zeros(self.N)
        Forces_W += -self.gx * self.d_rho # Gravity

        if self.sigma > 0: # Surface Tension (Linearized)
            dK = (np.roll(d2a, -1) - np.roll(d2a, 1)) / (2*self.dx)
            Forces_W += self.sigma * self.H * dK

        if self.visc_model == 'turbulent': # Turbulent Viscosity
            nu = self.lm * np.abs(self.W)
            stress = nu * (np.roll(self.W, -1) - np.roll(self.W, 1)) / (2*self.dx)
            Forces_W += (np.roll(stress, -1) - np.roll(stress, 1)) / (2*self.dx)

        if self.wall_drag: # Simple Drag
            Forces_W -= 1.0 * np.abs(self.W) * self.W

        # Update
        self.alpha += dt * (-dFa_dx)
        self.W += dt * (-dFW_dx + Forces_W)

# Run Simulations
fig1_2 = plot_linear_stability()
fig20_21 = plot_waveforms()

# Inviscid (Fig 3) & Viscous (Fig 8)
solver_inv = TFMSolver(L=0.02, N=100, sigma=1e-4) # Inviscid
solver_inv.alpha += 1e-3 * np.exp(-(solver_inv.x - 0.01)**2 / (2*(2e-3)**2)) # Perturb

solver_vis = TFMSolver(L=0.02, N=100, sigma=1e-4, viscosity_model='turbulent', lm=1e-3) # Viscous
solver_vis.alpha += 1e-3 * np.exp(-(solver_vis.x - 0.01)**2 / (2*(2e-3)**2))

# Fig 4: Blowup Time
Ns = [50, 100, 200]; blowup_times = []
for N in Ns:
    s = TFMSolver(0.02, N, sigma=1e-4); s.alpha += 1e-3 * np.exp(-(s.x - 0.01)**2 / 8e-6)
    t=0; dt=1e-5
    for _ in range(10000):
        s.step(dt); t+=dt
        if np.max(np.abs(s.W)) > 3.0: break
    blowup_times.append(t)

# Integrate Fig 3/8 Cases
hist_inv = []; hist_vis = []
for _ in range(3000):
    solver_inv.step(1e-5); solver_vis.step(1e-5)
    if _ % 1000 == 0: hist_inv.append(solver_inv.W.copy()); hist_vis.append(solver_vis.W.copy())

# Plotting PDE Results
fig_pde, axes = plt.subplots(2, 2, figsize=(12, 10))

# Fig 3 Equiv
axes[0,0].plot(hist_inv[-1]); axes[0,0].set_title('Fig 3: Inviscid Profile (Blowup)')
yf_inv = np.abs(fft(hist_inv[-1]-1.0))
axes[0,1].semilogy(yf_inv[:50]); axes[0,1].set_title('Fig 3: Flat Spectrum')

# Fig 8 Equiv
axes[1,0].plot(hist_vis[-1]); axes[1,0].set_title('Fig 8: Viscous Profile (Stable)')
yf_vis = np.abs(fft(hist_vis[-1]-1.0))
axes[1,1].semilogy(yf_vis[:50]); axes[1,1].set_title('Fig 8: Decaying Spectrum')

plt.tight_layout()
plt.show()

# Fig 4 Plot
plt.figure(figsize=(6,4))
plt.plot(Ns, blowup_times, 'ro-')
plt.title('Fig 4: Blowup Time vs Mesh Resolution')
plt.xlabel('Number of Nodes'); plt.ylabel('Time to Blowup [s]')
plt.gca().invert_yaxis()
plt.show()
