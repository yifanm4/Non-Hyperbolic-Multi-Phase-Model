# Entropy-Viscocity-TFM

Tests of two-fluid/two-phase models with MUSCL and entropy-viscosity (EVM) stabilization.

Key scripts:
- `test-water-faucet-simple-MUSCL-van-leer-main.py`, `test-water-faucet-simpleMUSCL+EVM.py`, `test-water-faucet-simple-MUSCL-only.py`: faucet benchmarks comparing flux schemes (Upwind, MUSCL, central, ENO/WENO/NT/KU/PCU) with/without EVM.
- `test-linear-advection.py`: linear advection sanity check.
- `test-shock-tube.py`: shock-tube cases.
- `test.py`: additional experiment harness.

Outputs: `Water_Faucet_Flux_Scheme_Comparison_*.png`, `Entropy_Viscosity_Comparison.png`, `shock_tube_results*.png` summarize scheme performance across grids/CFL/EVM settings.

Models: barotropic/isentropic two-fluid faucet and shock-tube setups; explores limiter choices, entropy-viscosity damping, and scheme stability/accuracy.***
