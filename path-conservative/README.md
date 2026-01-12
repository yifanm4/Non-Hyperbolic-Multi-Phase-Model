# path-conservative

Path-conservative central-upwind experiments for two-fluid systems.

- `path-conservative-original.py`: baseline path-conservative solver.
- `path-conservative-water-faucet-gemini.py`: faucet-specific variant.

Models: two-fluid equations with nonconservative products handled via path integrals.
Numerics: central-upwind flux with path terms; faucet BCs for the faucet variant.***
