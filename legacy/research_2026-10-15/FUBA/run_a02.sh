#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

echo "=== A-02: IID k20paper ==="
for s in 0 1 2; do
  python analysis_equilibrium.py --fmt fuba --name "bd_iid_r100w5_k20paper_s$s" --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 --cond iid_k20paper
done

echo "=== A-02: Owner k20paper ==="
for s in 0 1 2; do
  python analysis_equilibrium.py --fmt fuba --name "bd_owner_r100w5_k20paper_s$s" --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 --owner_client 19 --cond owner_k20paper
done

echo "=== A-02: Aggregation ==="
python analysis_equilibrium.py --aggregate --cond iid_r100w5_k20paper
python analysis_equilibrium.py --aggregate --cond owner_r100w5_k20paper
echo "=== A-02: Done ==="
