# Test the comparison of CASE-3..7 wave-speed modifications
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helpers: primitive <-> conservative
# U = [alpha_g, alpha_g*u_g, alpha_l*u_l, p_gauge]
# ------------------------------------------------------------
def primitives_from_U(U, eps=1e-12):
    ag = np.clip(U[:,0], eps, 1.0-eps)
    al = 1.0 - ag
    ug = U[:,1] / ag
    ul = U[:,2] / al
    p  = U[:,3]
    return p, ag, ug, ul

def U_from_primitives(p, ag, ug, ul):
    ag = np.asarray(ag)
    al = 1.0 - ag
    U = np.zeros((ag.size, 4), dtype=float)
    U[:,0] = ag
    U[:,1] = ag * ug
    U[:,2] = al * ul
    U[:,3] = p
    return U

# ------------------------------------------------------------
# Boundary conditions (water faucet): inlet Dirichlet ag,ug,ul;
# outlet Dirichlet p. (Singh & Mousseau faucet setup)
# ------------------------------------------------------------
def apply_bc(U, inlet_ag, inlet_ug, inlet_ul, outlet_p, ng=2):
    # left: ag,ug,ul fixed; p extrapolated
    U[:ng,0] = inlet_ag
    U[:ng,1] = inlet_ag * inlet_ug
    U[:ng,2] = (1.0-inlet_ag) * inlet_ul
    U[:ng,3] = U[ng:ng+1,3]  # extrapolate p

    # right: p fixed; others extrapolated
    U[-ng:,0] = U[-ng-1:-ng,0]
    U[-ng:,1] = U[-ng-1:-ng,1]
    U[-ng:,2] = U[-ng-1:-ng,2]
    U[-ng:,3] = outlet_p
    return U

# ------------------------------------------------------------
# Incompressible IP correction magnitude (Eq. 33)
# ------------------------------------------------------------
def pi_min(ag, ug, ul, rhog, rhol):
    al = 1.0 - ag
    denom = (ag*rhol + al*rhog) + 1e-30
    return (ag*al*rhog*rhol/denom) * (ug-ul)**2

# ------------------------------------------------------------
# Closure: choose physical p_i (no_ip / incompressible_ip / compressible_ip)
# compressible_ip here is a simple Mach^2 scaling placeholder.
# ------------------------------------------------------------
def compute_pi(closure, ag, ug, ul, rhog, rhol, cp, kappa=5.0):
    if closure == "no_ip":
        return np.zeros_like(ag)

    pim = pi_min(ag, ug, ul, rhog, rhol)

    if closure == "incompressible_ip":
        return pim

    if closure == "compressible_ip":
        Ms2 = (ug-ul)**2 / (4.0*cp*cp + 1e-30)  # u'=(ug-ul)/2
        return pim * (1.0 + kappa*Ms2)

    raise ValueError(f"unknown closure: {closure}")

# ------------------------------------------------------------
# CASE-3..7 wave-speed modifications from quartic-root study:
# You pass case_id in {"q0",3,4,5,6,7}
# q0 = "Study 1": set q->0  (double root at p)
# ------------------------------------------------------------
def modified_pair_speeds(ag, ug, ul, rhog, rhol, case_id):
    al = 1.0 - ag
    M  = al*rhog + ag*rhol
    p  = (ug*al*rhog + ul*ag*rhol)/M
    q  = np.abs(ug-ul)*np.sqrt(np.maximum(ag*al*rhog*rhol,0.0))/M

    if case_id == "q0":
        lam3 = p; lam4 = p
    elif case_id == 3:
        lam3 = np.sqrt(p*p + q*q); lam4 = lam3
    elif case_id == 4:
        lam3 = p + q; lam4 = lam3
    elif case_id == 5:
        lam3 = p - q; lam4 = lam3
    elif case_id == 6:
        lam3 = np.sqrt(np.maximum(p*p + q*q, 0.0))
        lam4 = np.sqrt(np.maximum(p*p - q*q, 0.0))
    elif case_id == 7:
        lam3 = p + q
        lam4 = p - q
    else:
        raise ValueError(case_id)

    return lam3, lam4

# ------------------------------------------------------------
# Fluxes (simple pseudo-compressibility pressure equation)
# ------------------------------------------------------------
def flux(U, rhog, rhol, cp):
    p, ag, ug, ul = primitives_from_U(U)
    al = 1.0 - ag
    F = np.zeros_like(U)
    F[:,0] = ag*ug
    F[:,1] = ag*ug*ug + ag*p/rhog
    F[:,2] = al*ul*ul + al*p/rhol
    F[:,3] = cp*cp*(ag*ug + al*ul)
    return F

# ------------------------------------------------------------
# One SSP-RK2 step using Rusanov flux + sources + diffusion
# ------------------------------------------------------------
def rusanov_step(U, dx, dt, closure, case_id,
                 rhog=1.0, rhol=1000.0, cp=80.0,
                 g=9.81, Kd=200.0, lm=0.02,
                 inlet=(0.2,1e-8,10.0), outlet_p=0.0, # alphag, ug, ul
                 kappa=5.0, vcap=600.0, ng=2):

    U = U.copy()
    U = apply_bc(U, inlet[0], inlet[1], inlet[2], outlet_p, ng=ng)

    # Interface states
    UL = U[ng-1:-ng]
    UR = U[ng:  -ng+1]
    FL = flux(UL, rhog, rhol, cp)
    FR = flux(UR, rhog, rhol, cp)

    # Wave speeds: include case-modified pair
    _, agL, ugL, ulL = primitives_from_U(UL)
    _, agR, ugR, ulR = primitives_from_U(UR)

    lam3L, lam4L = modified_pair_speeds(agL, ugL, ulL, rhog, rhol, case_id)
    lam3R, lam4R = modified_pair_speeds(agR, ugR, ulR, rhog, rhol, case_id)

    aL = np.maximum.reduce([np.abs(ugL)+cp, np.abs(ulL)+cp, np.abs(lam3L), np.abs(lam4L)])
    aR = np.maximum.reduce([np.abs(ugR)+cp, np.abs(ulR)+cp, np.abs(lam3R), np.abs(lam4R)])
    a  = np.maximum(aL, aR)

    num_flux = 0.5*(FL+FR) - 0.5*a[:,None]*(UR-UL)

    # Conservative update on interior
    Ui = U[ng:-ng]
    Unew = U.copy()
    Unew[ng:-ng] = Ui - (dt/dx)*(num_flux[1:]-num_flux[:-1])

    # Sources: gravity + drag + interfacial pressure correction term
    p, ag, ug, ul = primitives_from_U(Unew[ng:-ng])
    al = 1.0 - ag

    ag_ext = Unew[:,0]
    dag_dx = (ag_ext[ng+1:-ng+1] - ag_ext[ng-1:-ng-1])/(2.0*dx)

    pi = compute_pi(closure, ag, ug, ul, rhog, rhol, cp, kappa=kappa)

    Smg = ag*g - (Kd*(ug-ul)*ag*al)/rhog + (pi/rhog)*dag_dx
    Sml = al*g + (Kd*(ug-ul)*ag*al)/rhol - (pi/rhol)*dag_dx

    Unew[ng:-ng,1] += dt*Smg
    Unew[ng:-ng,2] += dt*Sml

    # explicit Laplacian diffusion (stabilizer)
    if lm and lm>0:
        Uc = Unew.copy()
        lap = (Uc[ng+1:-ng+1] - 2*Uc[ng:-ng] + Uc[ng-1:-ng-1])/(dx*dx)
        Unew[ng:-ng] += lm*dt*lap

    # clip
    p, ag, ug, ul = primitives_from_U(Unew[ng:-ng])
    ag = np.clip(ag, 1e-5, 1-1e-5)
    ug = np.clip(ug, -vcap, vcap)
    ul = np.clip(ul, -vcap, vcap)
    p  = np.clip(p, -5e6, 5e6)

    Unew[ng:-ng] = U_from_primitives(p, ag, ug, ul)
    Unew = apply_bc(Unew, inlet[0], inlet[1], inlet[2], outlet_p, ng=ng)
    return Unew

# ------------------------------------------------------------
# Solve wrapper
# ------------------------------------------------------------
def solve(closure="incompressible_ip", case_id=3,
          N=1000, L=12.0, T=0.5, CFL=0.1,
          rhog=1.0, rhol=1000.0, cp=80.0,
          inlet=(0.2,1e-8,10.0), outlet_p=0.0, # alphag, ug, ul
          g=9.81, Kd=0.0, lm=0.02, # Kd is drag coeff
          kappa=5.0, vcap=600.0):

    ng=2
    dx=L/N
    x=np.linspace(dx/2, L-dx/2, N)

    U = U_from_primitives(np.full(N+2*ng, outlet_p),
                          np.full(N+2*ng, inlet[0]),
                          np.full(N+2*ng, inlet[1]),
                          np.full(N+2*ng, inlet[2]))

    t=0.0
    while t < T-1e-12:
        p, ag, ug, ul = primitives_from_U(U[ng:-ng])
        amax = float(np.max(np.maximum(np.abs(ug)+cp, np.abs(ul)+cp)))
        dt = min(CFL*dx/max(amax,1e-12), T-t)

        U1 = rusanov_step(U, dx, dt, closure, case_id,
                          rhog,rhol,cp,g,Kd,lm,inlet,outlet_p,kappa,vcap,ng)
        U2 = rusanov_step(U1, dx, dt, closure, case_id,
                          rhog,rhol,cp,g,Kd,lm,inlet,outlet_p,kappa,vcap,ng)
        U = 0.5*(U + U2)
        t += dt

    p, ag, ug, ul = primitives_from_U(U[ng:-ng])
    return x, p, ag, ug, ul

# ------------------------------------------------------------
# Compare all Cases 3..7
# ------------------------------------------------------------
if __name__ == "__main__":
    cases = [3,4,5,6,7]
    closure = "incompressible_ip"   # set to "no_ip" or "compressible_ip"
    results = {}

    for cid in cases:
        x, p, ag, ug, ul = solve(closure=closure, case_id=cid, N=160, T=0.4)
        results[cid] = (p, ag, ug, ul)
        print(f"Done case {cid}: p[{p.min():.2e},{p.max():.2e}] ag[{ag.min():.3f},{ag.max():.3f}]")

    # plot
    fig, axs = plt.subplots(2,2, figsize=(12,8), sharex=True)
    for cid,(p,ag,ug,ul) in results.items():
        axs[0,0].plot(x,p,label=f"Case {cid}")
        axs[0,1].plot(x,ag,label=f"Case {cid}")
        axs[1,0].plot(x,ug,label=f"Case {cid}")
        axs[1,1].plot(x,ul,label=f"Case {cid}")

    axs[0,0].set_title("Pressure (gauge)")
    axs[0,1].set_title("alpha_g")
    axs[1,0].set_title("u_g")
    axs[1,1].set_title("u_l")
    for ax in axs.flat:
        ax.grid(True)
        ax.legend()
    axs[1,0].set_xlabel("x (m)")
    axs[1,1].set_xlabel("x (m)")
    plt.tight_layout()
    plt.savefig("compressible_ip_case_comparison.png")
    plt.show()
