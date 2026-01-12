import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Characteristic function G(Z; Pi)
# ----------------------------
def G(Z, Ms2, r, Pi):
    # G(Z;Pi) = (Z^2 - 2 r Z + 1) - Ms^2 (1 - Z^2)^2 - 4 Pi
    return (Z**2 - 2.0*r*Z + 1.0) - Ms2*(1.0 - Z**2)**2 - 4.0*Pi


def poly_coeffs(Ms2, r, Pi):
    """
    Expand G(Z;Pi)=0 into polynomial coefficients (highest power first).
    G(Z;Pi) = -Ms2 Z^4 + (2Ms2+1) Z^2 - 2r Z + (1 - Ms2 - 4Pi) = 0
    """
    return np.array([
        -Ms2,             # Z^4
        0.0,              # Z^3
        (2.0*Ms2 + 1.0),  # Z^2
        (-2.0*r),         # Z^1
        (1.0 - Ms2 - 4.0*Pi)  # Z^0
    ], dtype=float)


def roots_Z(Ms2, r, Pi):
    return np.roots(poly_coeffs(Ms2, r, Pi))


def split_roots(roots, tol=1e-10):
    real_mask = np.abs(np.imag(roots)) < tol
    return np.sort(np.real(roots[real_mask])), roots[~real_mask]


# ----------------------------
# IP models: Pi_inc and Pi_comp
# ----------------------------
def Pi_incompressible(r):
    # nondimensional incompressible minimum in this Z-form
    return 0.25*(1.0 - r*r)


def Pi_compressible_min(Ms2, r):
    """
    Minimal Pi such that G(Z;Pi)=0 has a double root (tangency):
       G(Z*)=0 and G'(Z*)=0
    G'(Z)= 2Z - 2r + 4 Ms2 Z (1 - Z^2)
    => -2 Ms2 Z^3 + (1+2 Ms2) Z - r = 0
    Choose the real root near r (continuous with Ms2->0).
    Then Pi = 1/4[(quad) - Ms2(1-Z^2)^2].
    """
    # cubic for Z*
    c = np.array([-2.0*Ms2, 0.0, (1.0 + 2.0*Ms2), -r], dtype=float)
    zc = np.roots(c)

    # pick a real root closest to r if available; else smallest imag
    real_roots = np.real(zc[np.abs(np.imag(zc)) < 1e-10])
    if len(real_roots) > 0:
        z_star = float(real_roots[np.argmin(np.abs(real_roots - r))])
    else:
        z_star = float(np.real(zc[np.argmin(np.abs(np.imag(zc)))]))

    Pi = 0.25*((z_star*z_star - 2.0*r*z_star + 1.0) - Ms2*(1.0 - z_star*z_star)**2)
    return max(0.0, float(Pi)), z_star


# ----------------------------
# Plot range selection (prevents divergence)
# ----------------------------
def choose_windows(roots_no_ip, z_star=None):
    """
    We return two windows:
      - inner: focuses on the "missing" pair location (where bowl/tangency happens)
      - outer: includes the two real crossings but clipped if huge
    """
    real_no, complex_no = split_roots(roots_no_ip)

    # inner window center:
    if z_star is not None:
        center = z_star
    elif len(complex_no) > 0:
        center = float(np.real(complex_no[0]))
    else:
        center = float(np.mean(real_no)) if len(real_no) else 0.0

    # inner width based on imag part of complex pair (if any), else default
    if len(complex_no) > 0:
        w = max(1.5, 3.0*float(np.max(np.abs(np.imag(complex_no)))))
    else:
        w = 2.0
    inner = (center - w, center + w)

    # outer window based on the two real roots of the no-IP case
    if len(real_no) >= 2:
        zL, zR = float(real_no[0]), float(real_no[-1])
        span = zR - zL
        pad = max(0.25, 0.08*span)
        outer_raw = (zL - pad, zR + pad)
    else:
        outer_raw = inner

    # clip outer window if it is enormous (common when Ms2 is tiny)
    max_span = 20.0 * (inner[1] - inner[0])
    if (outer_raw[1] - outer_raw[0]) > max_span:
        outer = (inner[0] - 10*(inner[1]-inner[0]), inner[1] + 10*(inner[1]-inner[0]))
    else:
        outer = outer_raw

    return inner, outer


def plot_case_set(Ms2, r, title_suffix=""):
    # 3 cases
    Pi0 = 0.0
    Pii = Pi_incompressible(r)
    Pic, z_star = Pi_compressible_min(Ms2, r)

    roots0 = roots_Z(Ms2, r, Pi0)
    rootsi = roots_Z(Ms2, r, Pii)
    rootsc = roots_Z(Ms2, r, Pic)

    real0, comp0 = split_roots(roots0)
    reali, compi = split_roots(rootsi)
    realc, compc = split_roots(rootsc)

    print("\n=== Parameters ===")
    print(f"Ms^2 = {Ms2:.6g}, r = {r:.6g}")
    print(f"Pi(no IP) = {Pi0:.6g}")
    print(f"Pi(inc IP) = {Pii:.6g}")
    print(f"Pi(comp IP, min) = {Pic:.6g}  (Z* = {z_star:.6g})")

    print("\n=== Roots in Z ===")
    print("No IP:          ", roots0)
    print("Incompressible: ", rootsi)
    print("Compressible:   ", rootsc)

    print("\nReal-root counts:",
          f"noIP={len(real0)}  inc={len(reali)}  comp={len(realc)}")

    inner, outer = choose_windows(roots0, z_star=z_star)

    # ---- Outer plot (shows the 2 crossings, but still safe) ----
    Z = np.linspace(outer[0], outer[1], 4000)
    y0 = G(Z, Ms2, r, Pi0)
    yi = G(Z, Ms2, r, Pii)
    yc = G(Z, Ms2, r, Pic)

    plt.figure()
    plt.axhline(0.0)
    plt.plot(Z, y0, label="No IP")
    plt.plot(Z, yi, label="Incompressible IP")
    plt.plot(Z, yc, label="Compressible IP (min, tangency)")
    # mark real roots
    for rr in real0: plt.plot(rr, 0.0, marker="o")
    for rr in reali: plt.plot(rr, 0.0, marker="o")
    for rr in realc: plt.plot(rr, 0.0, marker="o")
    plt.xlabel("Z")
    plt.ylabel("G(Z)")
    plt.title(f"Characteristic function (outer view){title_suffix}")
    plt.legend()
    plt.savefig("characteristic_function_Ms2_{Ms2:.3g}_r_{r:.3g}_outer.png")
    plt.show()

    # ---- Inner zoom (this is where the complex pair turns into real roots) ----
    Zz = np.linspace(inner[0], inner[1], 4000)
    y0z = G(Zz, Ms2, r, Pi0)
    yiz = G(Zz, Ms2, r, Pii)
    ycz = G(Zz, Ms2, r, Pic)

    plt.figure()
    plt.axhline(0.0)
    plt.plot(Zz, y0z, label="No IP")
    plt.plot(Zz, yiz, label="Incompressible IP")
    plt.plot(Zz, ycz, label="Compressible IP (min, tangency)")
    # mark relevant roots in zoom window
    for rr in reali:
        if inner[0] <= rr <= inner[1]:
            plt.plot(rr, 0.0, marker="o")
    for rr in realc:
        if inner[0] <= rr <= inner[1]:
            plt.plot(rr, 0.0, marker="o")
    plt.xlabel("Z")
    plt.ylabel("G(Z)")
    plt.title(f"Characteristic function (inner zoom){title_suffix}")
    plt.legend()
    plt.savefig("characteristic_function_Ms2_{Ms2:.3g}_r_{r:.3g}_inner.png")
    plt.show()


if __name__ == "__main__":
    # Pick a *moderate* Ms^2 for clean comparison (avoids huge outer roots)
    # This parameter choice reliably shows: no-IP -> 2 real + complex pair,
    # compressible-IP(min) -> double root (tangency),
    # incompressible-IP -> two distinct inner real roots.
    Ms2_demo = 0.1
    r_demo = 0.2
    plot_case_set(Ms2_demo, r_demo, title_suffix="  (demo Ms^2=0.1, r=0.2)")
