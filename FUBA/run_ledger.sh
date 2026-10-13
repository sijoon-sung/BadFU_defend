#!/bin/bash
# 균형 장부 방어 (ledger_defense.py) — 논문 설정 6 궤적 + 요약. 궤적당 1–2 분 (순전파만). 결과: logs/ledger/
# 사용: bash run_ledger.sh              # iid + owner (있는 궤적만)
#       CONDS="iid" bash run_ledger.sh  # 한 조건만
#       EXTRA="--h0 0.9" bash run_ledger.sh   # 문턱 바꿔 재판정 (두 번째부터는 --from_cache 로 순전파 없이)
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
CONDS="${CONDS:-iid owner}"; SEEDS="${SEEDS:-0 1 2}"; TAG="${TAG:-_r100w5_k20paper}"; EXTRA="${EXTRA:-}"
DATASET="${DATASET:-mnist}"; PREFIX="${PREFIX:-bd}"   # cifar: DATASET=cifar10 TAG=_r100w5_k20paper_cifar; 공격 없음: PREFIX=clean TAG=_r100_k20paper CONDS="iid owner"
for c in $CONDS; do
  own=""; [ "$c" = "owner" ] && own="--owner_client 19"
  for s in $SEEDS; do
    n="${PREFIX}_${c}${TAG}_s$s"
    [ -f "checkpoints/global_net_model_${n}_round_1.pkl" ] || { echo "skip $n (없음)"; continue; }
    cache=""; [ -f "logs/ledger/${n}_push.npz" ] && cache="--from_cache"
    roles="--defender 4 --attackers 0 1 2 3 --target 8"; [ "$PREFIX" = "clean" ] && roles="--defender -1 --attackers --target -1"
    python ledger_defense.py --fmt fuba --dataset "$DATASET" --name "$n" --K 20 --rounds 100 $roles $own --cond "${PREFIX}_${c}${TAG}" $cache $EXTRA || exit 1
  done
  python ledger_defense.py --aggregate --cond "${PREFIX}_${c}${TAG}"
done
