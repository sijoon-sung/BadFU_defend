#!/bin/bash
# 정상 이탈 검증 일괄 실행: 궤적 생성(FUBA, mpiexec) -> 요청자별·FU별 언러닝 -> 집계
#
# 사용법 (FUBA 폴더에서):
#   bash run_benign_departure.sh                      # COND=owner, 시드 0 1 2
#   COND=iid   bash run_benign_departure.sh
#   COND=dir01 bash run_benign_departure.sh           # FUBA Table V 조건(Dirichlet α=0.1)
#   SKIP_ATTACK=1 bash run_benign_departure.sh        # 저장된 궤적이 있으면 공격 단계 생략
#   METHODS="subtract distillation fedEraser" SEEDS="0" bash run_benign_departure.sh
#
# 조건
#   iid    : 원래 det* 와 같은 IID 분할
#   owner  : 정상 클라 K-1 이 OWNER_CLASS 의 90% 를 독점 (우리 fp3/fp4 관찰 조건)
#   dir01  : Dirichlet α=0.1 (FUBA Table V 의 non-IID 조건)
#   dir05  : Dirichlet α=0.5
# 요청자: 4 (Adv-defender, 공격이 설계한 요청), K-1 (독점 정상 / iid 에서는 보통 정상), 5 (보통 정상)
set -e
cd "$(dirname "$0")"
[ -x ../.venv/Scripts/python.exe ] && export PATH="$(cd .. && pwd)/.venv/Scripts:$PATH"

COND="${COND:-owner}"
SEEDS="${SEEDS:-0 1 2}"
K="${K:-8}"
OWNER_CLASS="${OWNER_CLASS:-3}"
METHODS="${METHODS:-subtract distillation fedEraser retrain_benign retrain_attack}"
ROUNDS="${ROUNDS:-8}"

case "$COND" in
  iid)   NI="" ;;
  owner) NI="--non_iid --non_iid_type class_owner --owner_class $OWNER_CLASS --owner_frac 0.9 --owner_client -1" ;;
  dir01) NI="--non_iid --non_iid_type Dirichlet --alpha 0.1" ;;
  dir05) NI="--non_iid --non_iid_type Dirichlet --alpha 0.5" ;;
  *) echo "COND 는 iid|owner|dir01|dir05"; exit 1 ;;
esac

for s in $SEEDS; do
  NAME="bd_${COND}_s${s}"; SEED=$((522 + s))
  if [ "${SKIP_ATTACK:-0}" = "1" ] && ls checkpoints/local_net_model_${NAME}_round_*.pkl >/dev/null 2>&1; then
    echo "== [$NAME] 저장된 궤적 사용"
  else
    echo "== [$NAME] 1/2 궤적 생성 (FUBA, seed $SEED, $COND)"
    NB_CLIENTS="$K" EXTRA_ARGS="$NI --split_seed $SEED" bash run_real_fuba.sh "$NAME" "$SEED" > "logs_fuba_${NAME}.txt" 2>&1
    n=$(ls checkpoints/local_net_model_${NAME}_round_*.pkl 2>/dev/null | wc -l)
    [ "$n" -ge "$ROUNDS" ] || { echo "   궤적 생성 실패: 저장된 라운드 $n 개. logs_fuba_${NAME}.txt 확인"; exit 1; }
  fi
  echo "== [$NAME] 2/2 요청자별 언러닝 + 평가"
  PYTHONIOENCODING=utf-8 python benign_departure.py --name "$NAME" --K "$K" --rounds "$ROUNDS" \
    --seed "$SEED" --split_seed "$SEED" $NI \
    --requesters 4 $((K - 1)) 5 --methods $METHODS --resume | grep -v "^Files already"
done

echo "== 집계 ($COND)"
PYTHONIOENCODING=utf-8 python aggregate_benign_departure.py --cond "$COND"
