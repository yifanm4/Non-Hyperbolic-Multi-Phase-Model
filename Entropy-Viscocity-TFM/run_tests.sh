#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
SCRIPT=${SCRIPT:-test-water-faucet-plus-Evje2005-shocktubes-multischeme.py}
OUTDIR=${OUTDIR:-out_plots}
N_faucet=${N_faucet:-100}

mkdir -p "$OUTDIR"

echo "Using: $PY $SCRIPT"
echo "Output dir: $OUTDIR"
echo

# -------------------------
# Faucet: overlay schemes
# -------------------------
echo "[1/4] Faucet overlay..."
$PY "$SCRIPT" \
  --problem faucet \
  --scheme "Upwind1,MUSCL_EVM,KU,PCU,SDCU61" \
  --N_faucet $N_faucet \
  --t_final_faucet 0.5 \
  --sigma 2.0 \
  --save "$OUTDIR/faucet_overlay_$N_faucet.png"

# -------------------------
# LRV (Evje&Flåtten 2005 §7.1)
# Fixed dx/dt = 1000 (as commonly used)
# -------------------------
echo "[2/4] LRV overlay..."
$PY "$SCRIPT" \
  --problem LRV \
  --scheme "KU,PCU,SDCU61" \
  --dt_mode fixed_dxdt \
  --dxdt 1000 \
  --save "$OUTDIR/LRV_overlay.png"

# -------------------------
# MLRV (Evje&Flåtten 2005 §7.2)
# dx/dt = 750
# -------------------------
echo "[3/4] MLRV overlay..."
$PY "$SCRIPT" \
  --problem MLRV \
  --scheme "KU,PCU,SDCU61" \
  --dt_mode fixed_dxdt \
  --dxdt 750 \
  --save "$OUTDIR/MLRV_overlay.png"

# -------------------------
# Toumi water-air shock (§7.3)
# sigma=2, N=200, t_final=0.08, dx/dt=1000
# -------------------------
echo "[4/4] Toumi overlay..."
$PY "$SCRIPT" \
  --problem TOUMI \
  --scheme "KU,PCU,SDCU61" \
  --dt_mode fixed_dxdt \
  --dxdt 1000 \
  --N 200 \
  --t_final 0.08 \
  --sigma 1.0 \
  --save "$OUTDIR/Toumi_overlay.png"

echo
echo "All done. Plots saved in: $OUTDIR"
ls -1 "$OUTDIR"
