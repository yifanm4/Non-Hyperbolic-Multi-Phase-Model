# K-TFM

Keyfitz-style two-fluid faucet solvers and comparisons.

- `Keyfitz_explicit_solver.py`, `Keyfitz_explicit_solver_HO.py`: explicit solvers (base and higher-order) for the Keyfitz faucet model with analytical reference.
- `Keyfitz_implicit_solver.py`: implicit variant.
- `5_solver_comparison.png`: comparison plot of solver outputs vs reference.
- `plots/`: saved figures from runs.

Model: faucet problem with single-velocity Keyfitz formulation; analytical void-fraction profile available for validation.
Numerics: explicit/implicit time integration, first/second-order reconstructions; gravity-driven acceleration; faucet BCs (fixed inlet α/u, outlet pressure).***
