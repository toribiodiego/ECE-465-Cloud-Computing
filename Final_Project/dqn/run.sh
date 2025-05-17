#!/usr/bin/env bash
# run.sh – sweep k∈{1,12} and dump everything into one text file + plot PNGs

set -euo pipefail
REPO_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$REPO_DIR"

# 1) Python env (reuse if it exists) ----------------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

# 2) Clean old artifacts ----------------------------------------------------
rm -rf .venv/sweep.txt sweep.txt checkpoints_k1 checkpoints_k12 plots

# 3) Sweep for k=1 and k=12 ------------------------------------------------
EPISODES=100000
WAVE=500

echo "=== Phase 3 Sweep (k=1,12) ===" > sweep.txt
for K in 1 12; do
  echo -e "\n--- Running actors=$K ---" | tee -a sweep.txt
  python phase3.py \
      --episodes $EPISODES \
      --wave $WAVE \
      --actors $K \
      --checkpoint-dir checkpoints_k$K \
      --plot-dir plots \
      >> sweep.txt 2>&1
done

echo -e "\nSweep complete. See:\n • sweep.txt (all console output)\n • plots/reward_curve_k1.png & reward_curve_k12.png" | tee -a sweep.txt
