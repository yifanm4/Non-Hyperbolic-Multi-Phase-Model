# Compressible-IP Faucet Tests

This folder holds quick tests for the primitive 4‑equation two‑fluid model with three interfacial-pressure (IP) options:
- No IP correction
- Incompressible IP
- Compressible IP* (hyperbolicity-based)

The setup mirrors the water-faucet problem: α_g,in=0.2, u_l,in=10 m/s, u_g,in≈0, L=12 m, T=0.5 s, g=9.81, p_out=0.

## Scripts
- `faucet_ip_compare.py`: runs a simple first-order upwind/Rusanov scheme for all three IP modes and saves plots: `void_fraction_upwind_ip.png`, `ug_upwind_ip.png`, `ul_upwind_ip.png`, `hyperbolicity_upwind_ip.png`.
- `cip-test.py`: legacy richer demo (MUSCL, compressible IP* table build, etc.) with plots for α_g, u_g, u_l, and hyperbolicity.

## How to run
From the repo root:
```bash
python compressible-ip/faucet_ip_compare.py
# or the legacy demo:
python compressible-ip/cip-test.py
```

## Notes/limitations
- These tests use the hyperbolic primitive model (no elliptic pressure projection). Void-fraction profiles will differ from the faucet reference that includes an elliptic pressure/entropy-viscosity treatment.
- CFL is conservative; runs may take a minute. The compressible IP* table precompute happens once per run in both scripts.
