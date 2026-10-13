#!/bin/bash
# 균형 장부 방어 (ledger_defense.py) — 논문 설정 6 궤적 + 요약. 궤적당 1–2 분 (순전파만). 결과: logs/ledger/
# 사용: bash run_ledger.sh              # iid + owner (있는 궤적만)
#       CONDS="iid" bash run_ledger.sh  # 한 조건만
#       EXTRA="--h0 0.9" bash run_ledger.sh   # 문턱 바꿔 재판정 (두 번째부터는 --from_cache 로 순전파 없이)
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
CONDS="${CONDS:-iid owner}"; SEEDS="${SEEDS:-0 1 2}"; TAG="${TAG:-_r100w5_k20paper}"; EXTRA="${EXTRA:-}"
for c in $CONDS; do
  own=""; [ "$c" = "owner" ] && own="--owner_client 19"
  for s in $SEEDS; do
    n="bd_${c}${TAG}_s$s"
    [ -f "checkpoints/global_net_model_${n}_round_1.pkl" ] || { echo "skip $n (없음)"; continue; }
    cache=""; [ -f "logs/ledger/${n}_push.npz" ] && cache="--from_cache"
    python ledger_defense.py --fmt fuba --name "$n" --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 $own --cond "${c}${TAG}" $cache $EXTRA || exit 1
  done
  python ledger_defense.py --aggregate --cond "${c}${TAG}"
done
