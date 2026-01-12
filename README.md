# Non-Hyperbolic Conservative Two-Equation Model — Overview

This workspace collects experiments and benchmarks for two-fluid models (barotropic/isentropic, interfacial pressure corrections, flux schemes, and faucet/shock tube tests). Key folders:

- `compressible-ip/` — Primitive 4-eq compressible tests; faucet comparisons (`water_faucet_compare.py`, `faucet_ip_compare.py`, `cip-test.py`) with no/IP/inc/comp IP options.
- `K-TFM/` — Keyfitz-type TFM solvers and analytical faucet reference.
- `BC-TFM/`, `Entropy-Viscocity-TFM/`, `TF-TFM/`, `K-TFM/` — Variants of two-fluid/TMF solvers, stability studies, and flux comparisons.
- `AUSM/4eq-model/` (external reference) — AUSM/LLF flux implementations reused in some faucet runs.
- `path-conservative/`, `action-study/`, `Entropy-Viscocity-TFM/` — Additional numerical experiments (central/path-conservative schemes, entropy viscosity).

Typical faucet setup used across scripts: α_g,in≈0.2, u_l,in≈10 m/s, u_g,in≈0, L≈12 m, g=9.81, p_out≈1e5 (sometimes 0), with variations per solver.

How to run (examples):
- `python compressible-ip/water_faucet_compare.py`
- `python compressible-ip/faucet_ip_compare.py`
- `python compressible-ip/cip-test.py`
- `python K-TFM/Keyfitz_explicit_solver_HO.py`

Notes:
- Some scripts rely on the AUSM repo in `/Users/yifanmao/Desktop/UIUC/AUSM/4eq-model/Restart Mission` for flux functions.
- Plots are written alongside scripts (e.g., `void_fraction_upwind_ip.png`). Check script headers for dependencies and IC/BC details.
