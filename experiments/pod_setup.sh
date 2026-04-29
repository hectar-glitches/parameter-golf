#!/bin/bash
set -e

# ── 1. Install deps ──────────────────────────────────────────────────────────
pip install -q flash-attn --no-build-isolation
pip install -q -r requirements.txt

# ── 2. Download SP8192 dataset ───────────────────────────────────────────────
python3 data/cached_challenge_fineweb.py --variant sp8192

echo "Setup complete. Ready to train."
