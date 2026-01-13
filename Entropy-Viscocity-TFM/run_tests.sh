#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
# ✅ Updated script to the version that includes PHASESEP + MANOMETER
SCRIPT=${SCRIPT:-test-multitest-multischeme.py} # test-water-faucet-plus-multitest-multischeme

OUTDIR=${OUTDIR:-out_plots}
N_faucet=${N_faucet:-100}

mkdir -p "$OUTDIR"

echo "Using: $PY $SCRIPT"
echo "Output dir: $OUTDIR"
echo

# -------------------------
# Faucet: overlay schemes
# -------------------------
echo "[1/6] Faucet overlay..."
$PY "$SCRIPT" \
  --problem faucet \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --N_faucet "$N_faucet" \
  --t_final_faucet 0.5 \
  --sigma 2.0 \
  --save "$OUTDIR/faucet_overlay_${N_faucet}.png"

# -------------------------
# LRV (Evje&Flåtten 2005 §7.1)
# Fixed dx/dt = 1000
# -------------------------
echo "[2/6] LRV overlay..."
$PY "$SCRIPT" \
  --problem LRV \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --dt_mode fixed_dxdt \
  --dxdt 1000 \
  --save "$OUTDIR/LRV_overlay.png"

# -------------------------
# MLRV (Evje&Flåtten 2005 §7.2)
# dx/dt = 750
# -------------------------
echo "[3/6] MLRV overlay..."
$PY "$SCRIPT" \
  --problem MLRV \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --dt_mode fixed_dxdt \
  --dxdt 750 \
  --save "$OUTDIR/MLRV_overlay.png"

# -------------------------
# Toumi water-air shock (§7.3)
# N=200, t_final=0.08, dx/dt=1000, sigma=2 (paper)
# -------------------------
echo "[4/6] Toumi overlay..."
$PY "$SCRIPT" \
  --problem TOUMI \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --dt_mode fixed_dxdt \
  --dxdt 1000 \
  --N 200 \
  --t_final 0.08 \
  --sigma 2.0 \
  --save "$OUTDIR/Toumi_overlay.png"

# -------------------------
# Phase separation (Paillere 2003 §4.4)
# Wall BC both sides; recommend small CFL
# -------------------------
echo "[5/6] Phase separation overlay..."
$PY "$SCRIPT" \
  --problem PHASESEP \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --dt_mode cfl \
  --CFL 0.5 \
  --save "$OUTDIR/PHASESEP_overlay.png"

# -------------------------
# Oscillating manometer (Paillere 2003 §4.5)
# Pressure BC both ends (1 bar); includes piecewise gx and drag
# -------------------------
echo "[6/6] Manometer overlay..."
$PY "$SCRIPT" \
  --problem MANOMETER \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --dt_mode cfl \
  --CFL 0.25 \
  --save "$OUTDIR/MANOMETER_overlay.png"

echo
echo "All done. Plots saved in: $OUTDIR"
ls -1 "$OUTDIR"
