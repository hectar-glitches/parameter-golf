#!/bin/bash
set -e

# ── 1. Install Python deps ────────────────────────────────────────────────────
pip install -q -r requirements.txt

# ── 2. Install Flash Attention 3 ─────────────────────────────────────────────
# The SOTA script uses flash_attn_interface (FA3 API).
# First try the cu130 wheel (works if NVIDIA driver >= 570 / CUDA 13.0 capable).
# Fall back to building FA2 if the driver is too old.
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
  || pip install flash-attn --no-build-isolation

# Verify FA3 interface is importable
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" \
  || { echo "ERROR: flash_attn_interface not available — need FA3. See README."; exit 1; }

# ── 3. Download SP8192 dataset ────────────────────────────────────────────────
# Dataset hosted by Kevin Clark (PR #1394 author) at kevclark/parameter-golf
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

echo "Setup complete. Ready to train."
