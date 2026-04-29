#!/bin/bash
set -e

# ── 1. Upgrade PyTorch to cu128 (matches SOTA environment) ───────────────────
# The SOTA (PR #1493) was run with PyTorch 2.9.1+cu128 + FA3 cu128 wheels.
# RunPod H100 SXM driver >= 520 supports CUDA 12.8.
pip install -q --upgrade \
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# ── 2. Install Python deps ────────────────────────────────────────────────────
pip install -q -r requirements.txt

# ── 3. Install Flash Attention 3 (FA3) ───────────────────────────────────────
# The SOTA script imports flash_attn_interface (FA3-only API).
# Using the community wheel index that the SOTA README specifies.
pip install flash_attn_3 --no-deps \
  --find-links http://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Verify FA3 interface is importable
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" \
  || { echo "ERROR: flash_attn_interface not available. Check driver / CUDA version."; exit 1; }

# ── 4. Download SP8192 dataset ────────────────────────────────────────────────
# Dataset hosted by Kevin Clark (PR #1394 author) at kevclark/parameter-golf.
# Must delete stale manifest.json first (default repo only has sp1024).
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

echo "Setup complete. Ready to train."
