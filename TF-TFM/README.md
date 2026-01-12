# TF-TFM

Transitional/two-fluid model experiments (compressible and multiphase) and related PDE tests.

Representative scripts:
- `TF-shock-tube-multifluid.py`, `TF-shock-tube-multiphase.py`: multifluid/multiphase shock-tube cases with relaxation parameters (`tau_D`, `tau_T`); see saved result PNGs.
- `TF-compressible-subsonic.py`: compressible subsonic test.
- `Kinematic-Wave-Equation.py`, `Klein-Gordon-Equation.py`, `inverted-pendulum.py`: auxiliary PDE/instability studies (kinematic wave, Klein–Gordon, pendulum Lyapunov).
- Plots: `TF_Multifluid_Shock_Tube_Results_*.png`, `TF_Multiphase_Shock_Tube_Results_*.png`, `Klein_Gordon_Instability.png`, `Lyapunov_Instability_Comparison_Inverted_Pendulum.png`, `comparison_*`.

Models: barotropic/relaxation two-fluid equations; additional PDE stability tests.
Numerics: finite-volume style updates with relaxation terms (see scripts for specific schemes/parameters).***
