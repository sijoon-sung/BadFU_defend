#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

for c in iid owner; do
  for s in 0 1 2; do
    out="logs/benign_departure/bd_${c}_r100w5_k20paper_excl_s$s.json"
    if [ -f "$out" ]; then
      echo "=== [SKIP] Already exists: $out ==="
      continue
    fi
    echo "=== Running R-010: $c seed $s -> $out ==="
    EXTRA_NI=""
    [ "$c" = "owner" ] && EXTRA_NI="--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1"
    python benign_departure.py --name "bd_${c}_r100w5_k20paper_s$s" --K 20 --rounds 100 \
      --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 \
      --methods retrain_benign --exclude 0 1 2 3 $EXTRA_NI --out "$out"
  done
done

echo "=== Aggregating R-010 ==="
python aggregate_benign_departure.py
echo "=== Done R-010 ==="
