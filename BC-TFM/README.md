# BC-TFM

Two-fluid tests using the Bertodano–Clauss (BC) reduced model and variants.

- `BC-water-faucet.py`: faucet benchmark with coarse/fine grids; plots α, velocities, and pressure.
- `BC-shock-tube.py`: shock-tube setup for BC model.
- `BC_stability_study_test.py`: stability/regularization study.
- `Bertodano-Clauss-Model.py` / `Bertodano-Clauss-Model-HO.py`: core model implementations (base and higher-order).
- `BC_water_faucet_results.png`: reference faucet results.

Numerics: first-order upwind-style schemes (and a higher-order version) on the BC equations; faucet BCs (fixed inlet α/u, outlet pressure).

Typical problems: faucet rise/relaxation and shock-tube transients; compare coarse vs fine grids and stability variants.***
